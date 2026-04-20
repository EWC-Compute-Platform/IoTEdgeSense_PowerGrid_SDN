/**
 * @file onnx_inference.h
 * @brief ONNX Runtime inference engine for edge ML models
 *
 * Wraps the ONNX Runtime C++ API into a clean interface that the
 * DataProcessor and PowerGridBridge can call without knowing anything
 * about ONNX internals.
 *
 * Build requirements:
 *   - ONNX Runtime >= 1.16  (https://github.com/microsoft/onnxruntime)
 *   - CMake: find_package(OnnxRuntime REQUIRED) or manual link
 *
 * Compile without ONNX Runtime (stub mode):
 *   -DONNX_STUB   — returns constant predictions, useful for CI
 */

#ifndef ONNX_INFERENCE_H
#define ONNX_INFERENCE_H

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <chrono>
#include "../system/error_handler.h"
#include "../system/logger.h"

#ifndef ONNX_STUB
  #include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#endif

namespace ML {

// ---------------------------------------------------------------------------
// Inference result
// ---------------------------------------------------------------------------

/**
 * @brief Output of a single transformer health inference pass
 */
struct ThermalPrediction {
    float    rul_days      = 0.0f;  ///< Remaining useful life in days
    float    ageing_state  = 0.0f;  ///< Normalised ageing [0=new, 1=EOL]
    float    fault_prob    = 0.0f;  ///< Probability of fault within 30 days
    bool     fault_imminent = false; ///< fault_prob > threshold
    uint64_t timestamp_ms  = 0;     ///< Inference timestamp
    bool     valid         = false;  ///< False if model not loaded or error

    // Derived human-readable fields
    std::string urgency() const {
        if (!valid)           return "UNKNOWN";
        if (fault_imminent)   return "CRITICAL";
        if (rul_days < 90)    return "WARNING";
        if (rul_days < 365)   return "MONITOR";
        return "NORMAL";
    }
};

// ---------------------------------------------------------------------------
// ONNX Inference Engine
// ---------------------------------------------------------------------------

/**
 * @brief Loads an ONNX model and runs inference on feature vectors.
 *
 * Thread-safety: each OnnxInferenceEngine instance is not thread-safe.
 * Create one per thread, or protect with a mutex.
 */
class OnnxInferenceEngine {
public:
    /**
     * @brief Constructor
     *
     * @param modelPath       Path to .onnx model file
     * @param nFeatures       Expected input feature vector length
     * @param faultThreshold  Probability above which fault_imminent=true
     * @param numThreads      ONNX Runtime intra-op thread count (1=serial)
     */
    explicit OnnxInferenceEngine(const std::string& modelPath,
                                  size_t             nFeatures      = 43,
                                  float              faultThreshold = 0.5f,
                                  int                numThreads     = 1);

    ~OnnxInferenceEngine();

    /**
     * @brief Load (or reload) the model from disk.
     * @return true on success
     */
    bool loadModel();

    /**
     * @brief Run inference on a single feature vector.
     *
     * @param features  Float vector of length nFeatures (must match model)
     * @return ThermalPrediction with valid=true on success
     */
    ThermalPrediction infer(const std::vector<float>& features);

    /**
     * @brief Run inference on a batch of feature vectors.
     *
     * @param batch  Row-major matrix: batch_size × nFeatures
     * @return Vector of ThermalPrediction, one per row
     */
    std::vector<ThermalPrediction> inferBatch(
        const std::vector<std::vector<float>>& batch
    );

    bool        isLoaded()     const;
    size_t      nFeatures()    const;
    std::string modelPath()    const;
    uint64_t    inferenceCount() const;
    float       avgLatencyMs() const;

    /**
     * @brief Update fault threshold at runtime (e.g. from command topic)
     */
    void setFaultThreshold(float threshold);

private:
    std::string  mModelPath;
    size_t       mNFeatures;
    float        mFaultThreshold;
    int          mNumThreads;
    bool         mLoaded = false;

    // Runtime statistics
    uint64_t     mInferenceCount = 0;
    double       mTotalLatencyMs = 0.0;

#ifndef ONNX_STUB
    std::unique_ptr<Ort::Env>            mEnv;
    std::unique_ptr<Ort::Session>        mSession;
    std::unique_ptr<Ort::SessionOptions> mSessionOptions;
    Ort::AllocatorWithDefaultOptions     mAllocator;

    // Cached input/output names (allocated once on load)
    std::string mInputName;
    std::string mOutputNameRul;
    std::string mOutputNameAgeing;
    std::string mOutputNameFault;
#endif

    ThermalPrediction makePrediction(float rul, float ageing, float faultProb) const;
    uint64_t          nowMs() const;
};

} // namespace ML

#endif // ONNX_INFERENCE_H
