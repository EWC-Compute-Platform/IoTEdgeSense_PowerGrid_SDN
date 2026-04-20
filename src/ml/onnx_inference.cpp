/**
 * @file onnx_inference.cpp
 * @brief ONNX Runtime inference engine implementation
 *
 * Compile with -DONNX_STUB for stub mode (no ONNX Runtime needed).
 */

#include "onnx_inference.h"
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <numeric>

namespace ML {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

OnnxInferenceEngine::OnnxInferenceEngine(const std::string& modelPath,
                                          size_t             nFeatures,
                                          float              faultThreshold,
                                          int                numThreads)
    : mModelPath(modelPath)
    , mNFeatures(nFeatures)
    , mFaultThreshold(faultThreshold)
    , mNumThreads(numThreads)
{
#ifndef ONNX_STUB
    mEnv = std::make_unique<Ort::Env>(
        ORT_LOGGING_LEVEL_WARNING, "IoTEdgeSense_ML"
    );
#endif
    LOG_INFO("OnnxInference",
        "Engine created, model=" + modelPath
        + " features=" + std::to_string(nFeatures));
}

OnnxInferenceEngine::~OnnxInferenceEngine() {
    LOG_DEBUG("OnnxInference", "Engine destroyed");
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

bool OnnxInferenceEngine::loadModel() {
#ifdef ONNX_STUB
    mLoaded = true;
    LOG_INFO("OnnxInference", "STUB mode — model load simulated");
    return true;
#else
    try {
        mSessionOptions = std::make_unique<Ort::SessionOptions>();
        mSessionOptions->SetIntraOpNumThreads(mNumThreads);
        mSessionOptions->SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL
        );

        mSession = std::make_unique<Ort::Session>(
            *mEnv,
            mModelPath.c_str(),
            *mSessionOptions
        );

        // Cache input name
        size_t nInputs = mSession->GetInputCount();
        if (nInputs != 1) {
            LOG_ERROR("OnnxInference",
                "Expected 1 input, got " + std::to_string(nInputs));
            return false;
        }
        auto inputNamePtr = mSession->GetInputNameAllocated(0, mAllocator);
        mInputName = std::string(inputNamePtr.get());

        // Cache output names
        size_t nOutputs = mSession->GetOutputCount();
        if (nOutputs != 3) {
            LOG_ERROR("OnnxInference",
                "Expected 3 outputs, got " + std::to_string(nOutputs));
            return false;
        }
        auto o0 = mSession->GetOutputNameAllocated(0, mAllocator);
        auto o1 = mSession->GetOutputNameAllocated(1, mAllocator);
        auto o2 = mSession->GetOutputNameAllocated(2, mAllocator);
        mOutputNameRul    = std::string(o0.get());
        mOutputNameAgeing = std::string(o1.get());
        mOutputNameFault  = std::string(o2.get());

        // Verify input shape
        auto inputInfo = mSession->GetInputTypeInfo(0);
        auto shape = inputInfo.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() >= 2 && shape[1] != static_cast<int64_t>(-1)) {
            size_t modelFeatures = static_cast<size_t>(shape[1]);
            if (modelFeatures != mNFeatures) {
                LOG_ERROR("OnnxInference",
                    "Model expects " + std::to_string(modelFeatures)
                    + " features, engine configured for "
                    + std::to_string(mNFeatures));
                return false;
            }
        }

        mLoaded = true;
        LOG_INFO("OnnxInference",
            "Model loaded: input='" + mInputName + "'"
            + " outputs=[" + mOutputNameRul + ","
            + mOutputNameAgeing + "," + mOutputNameFault + "]");
        return true;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("OnnxInference",
            std::string("ONNX Runtime error: ") + e.what());
        System::ErrorHandler::getInstance().reportError(
            System::ErrorCode::NOT_INITIALIZED,
            System::ErrorSeverity::ERROR,
            System::ErrorSubsystem::SYSTEM,
            std::string("ONNX model load failed: ") + e.what(),
            "OnnxInferenceEngine::loadModel"
        );
        return false;
    }
#endif
}

// ---------------------------------------------------------------------------
// Single-sample inference
// ---------------------------------------------------------------------------

ThermalPrediction OnnxInferenceEngine::infer(const std::vector<float>& features) {
    if (!mLoaded) {
        LOG_WARN("OnnxInference", "Inference called before model loaded");
        return ThermalPrediction{};
    }
    if (features.size() != mNFeatures) {
        LOG_ERROR("OnnxInference",
            "Feature length mismatch: got " + std::to_string(features.size())
            + " expected " + std::to_string(mNFeatures));
        return ThermalPrediction{};
    }

    auto t0 = std::chrono::steady_clock::now();

#ifdef ONNX_STUB
    // Stub: return physics-based estimate from first few features
    // features[0] = K (load factor)
    // features[4] = theta_h_c (hot-spot temperature)
    float K       = features.size() > 0 ? features[0] : 1.0f;
    float theta_h = features.size() > 4 ? features[4] : 70.0f;
    float V       = features.size() > 7 ? features[7] : 1.0f;

    // Simple stub estimates
    float ageing    = std::min(1.0f, V * 0.05f);
    float rul       = std::max(0.0f, 14600.0f * (1.0f - ageing));
    float faultProb = (theta_h > 90.0f) ? 0.7f : (theta_h > 75.0f ? 0.3f : 0.1f);

    auto pred = makePrediction(rul, ageing, faultProb);
#else
    try {
        // Build input tensor
        std::array<int64_t, 2> inputShape = {1, static_cast<int64_t>(mNFeatures)};
        auto memInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault
        );
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo,
            const_cast<float*>(features.data()),
            features.size(),
            inputShape.data(),
            inputShape.size()
        );

        const char* inputNames[]  = {mInputName.c_str()};
        const char* outputNames[] = {
            mOutputNameRul.c_str(),
            mOutputNameAgeing.c_str(),
            mOutputNameFault.c_str()
        };

        auto outputs = mSession->Run(
            Ort::RunOptions{nullptr},
            inputNames,  &inputTensor,  1,
            outputNames, 3
        );

        float rul       = *outputs[0].GetTensorData<float>();
        float ageing    = *outputs[1].GetTensorData<float>();
        float faultProb = *outputs[2].GetTensorData<float>();

        auto pred = makePrediction(rul, ageing, faultProb);

    } catch (const Ort::Exception& e) {
        LOG_ERROR("OnnxInference",
            std::string("Inference failed: ") + e.what());
        return ThermalPrediction{};
    }
#endif

    auto t1 = std::chrono::steady_clock::now();
    double latMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    mTotalLatencyMs += latMs;
    ++mInferenceCount;

    LOG_DEBUG("OnnxInference",
        "Inference complete in " + std::to_string(latMs) + "ms"
        + " RUL=" + std::to_string(pred.rul_days) + "d"
        + " fault=" + pred.urgency());

#ifdef ONNX_STUB
    return pred;
#else
    return pred;   // pred defined in both branches above
#endif
}

// ---------------------------------------------------------------------------
// Batch inference
// ---------------------------------------------------------------------------

std::vector<ThermalPrediction>
OnnxInferenceEngine::inferBatch(const std::vector<std::vector<float>>& batch) {
    std::vector<ThermalPrediction> results;
    results.reserve(batch.size());
    for (const auto& row : batch) {
        results.push_back(infer(row));
    }
    return results;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

bool        OnnxInferenceEngine::isLoaded()       const { return mLoaded; }
size_t      OnnxInferenceEngine::nFeatures()       const { return mNFeatures; }
std::string OnnxInferenceEngine::modelPath()       const { return mModelPath; }
uint64_t    OnnxInferenceEngine::inferenceCount()  const { return mInferenceCount; }
float       OnnxInferenceEngine::avgLatencyMs()    const {
    return mInferenceCount > 0
        ? static_cast<float>(mTotalLatencyMs / mInferenceCount)
        : 0.0f;
}

void OnnxInferenceEngine::setFaultThreshold(float threshold) {
    mFaultThreshold = std::clamp(threshold, 0.0f, 1.0f);
    LOG_INFO("OnnxInference",
        "Fault threshold updated to " + std::to_string(mFaultThreshold));
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

ThermalPrediction
OnnxInferenceEngine::makePrediction(float rul, float ageing, float faultProb) const {
    ThermalPrediction p;
    p.rul_days       = std::max(0.0f, rul);
    p.ageing_state   = std::clamp(ageing, 0.0f, 1.0f);
    p.fault_prob     = std::clamp(faultProb, 0.0f, 1.0f);
    p.fault_imminent = (p.fault_prob >= mFaultThreshold);
    p.timestamp_ms   = nowMs();
    p.valid          = true;
    return p;
}

uint64_t OnnxInferenceEngine::nowMs() const {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<milliseconds>(
            steady_clock::now().time_since_epoch()).count());
}

} // namespace ML
