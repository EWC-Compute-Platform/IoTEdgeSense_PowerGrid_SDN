/**
 * @file predictive_maintenance.cpp
 * @brief PredictiveMaintenance implementation
 */

#include "predictive_maintenance.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cmath>

namespace ML {

// ---------------------------------------------------------------------------
// MaintenanceAlert JSON serialisation
// ---------------------------------------------------------------------------

std::string MaintenanceAlert::toJson() const {
    std::ostringstream o;
    o << std::fixed << std::setprecision(2);
    auto sev = [this]() -> std::string {
        switch (severity) {
            case AlertSeverity::NORMAL:   return "NORMAL";
            case AlertSeverity::MONITOR:  return "MONITOR";
            case AlertSeverity::WARNING:  return "WARNING";
            case AlertSeverity::CRITICAL: return "CRITICAL";
            default:                      return "UNKNOWN";
        }
    };
    o << "{"
      << "\"ts\":"           << timestamp_ms        << ","
      << "\"node\":\""       << nodeTag              << "\","
      << "\"nodeId\":"       << static_cast<int>(nodeId) << ","
      << "\"severity\":\""   << sev()                << "\","
      << "\"rul_days\":"     << rulDays              << ","
      << "\"ageing_state\":" << ageingState          << ","
      << "\"fault_prob\":"   << faultProbability     << ","
      << "\"hotspot_c\":"    << hotspotTempC         << ","
      << "\"V\":"            << relativeAgeingRate   << ","
      << "\"cum_ageing_h\":" << cumulativeAgeingHours << ","
      << "\"action\":\""     << recommendedAction    << "\""
      << "}";
    return o.str();
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

PredictiveMaintenance::PredictiveMaintenance(const std::string& modelPath,
                                              size_t nFeatures)
    : mModelPath(modelPath)
    , mNFeatures(nFeatures)
{
    LOG_INFO("PredictiveMaintenance",
        "Module created, model=" + (modelPath.empty() ? "[stub]" : modelPath));
}

bool PredictiveMaintenance::initialize() {
    std::lock_guard<std::mutex> lock(mMutex);

    if (mModelPath.empty()) {
        // Stub mode — create engine without a real model path
        mEngine = std::make_unique<OnnxInferenceEngine>(
            "", mNFeatures, 0.5f, 1
        );
        mEngine->loadModel();   // loads in stub mode
        LOG_INFO("PredictiveMaintenance",
            "Initialised in stub mode (no ONNX model path provided)");
        return true;
    }

    mEngine = std::make_unique<OnnxInferenceEngine>(
        mModelPath, mNFeatures, 0.5f, 1
    );
    bool ok = mEngine->loadModel();
    if (!ok) {
        LOG_ERROR("PredictiveMaintenance", "Failed to load ONNX model");
        return false;
    }
    LOG_INFO("PredictiveMaintenance",
        "Initialised, model loaded: " + mModelPath);
    return true;
}

// ---------------------------------------------------------------------------
// Node registration
// ---------------------------------------------------------------------------

void PredictiveMaintenance::registerTransformer(const TransformerConfig& cfg) {
    std::lock_guard<std::mutex> lock(mMutex);
    mConfigs[cfg.nodeId] = cfg;
    mStates [cfg.nodeId] = NodeThermalState{};
    LOG_INFO("PredictiveMaintenance",
        "Registered transformer node '" + cfg.nodeTag + "' (id="
        + std::to_string(cfg.nodeId) + ")");
}

// ---------------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------------

void PredictiveMaintenance::ingest(const Grid::ThreePhaseMeasurement& m) {
    if (!m.valid) return;

    std::lock_guard<std::mutex> lock(mMutex);

    auto cfgIt = mConfigs.find(m.nodeId);
    if (cfgIt == mConfigs.end()) return;  // node not registered for ML

    const TransformerConfig& cfg = cfgIt->second;
    NodeThermalState& state      = mStates[m.nodeId];

    // ── Load factor from measured current ────────────────────────────────
    float Iavg = (m.currentA_A + m.currentB_A + m.currentC_A) / 3.0f;
    float K    = std::clamp(Iavg / cfg.ratedCurrentA, 0.0f, 2.0f);

    // ── Ambient temperature from measurement context ──────────────────────
    // Use a fixed default if not measured; in practice feed from ThermalSensor
    float ambientC = 20.0f;

    // ── Determine dt since last update ───────────────────────────────────
    float dtMinutes = 1.0f;  // default 1-minute step
    uint64_t now = nowMs();
    if (state.lastUpdateMs > 0) {
        dtMinutes = static_cast<float>(now - state.lastUpdateMs) / 60000.0f;
        dtMinutes = std::clamp(dtMinutes, 0.1f, 60.0f);
    }
    state.lastUpdateMs = now;

    // ── Update IEC 60076-7 thermal model ─────────────────────────────────
    updateThermalState(state, cfg, K, ambientC, dtMinutes);

    // ── Push to rolling buffers ───────────────────────────────────────────
    state.pushSample(K, state.thetaHC, state.V);

    // ── Check inference schedule ──────────────────────────────────────────
    uint64_t& lastInf = mLastInferenceMs[m.nodeId];
    uint64_t intervalMs = static_cast<uint64_t>(cfg.inferenceIntervalSec) * 1000;

    if (now - lastInf < intervalMs) return;  // not yet due
    lastInf = now;

    // ── Build feature vector and run inference ────────────────────────────
    if (!mEngine || !mEngine->isLoaded()) return;

    auto features = buildFeatureVector(state, cfg, m);
    ThermalPrediction pred = mEngine->infer(features);
    if (!pred.valid) return;

    mLastPredictions[m.nodeId] = pred;

    // ── Build and dispatch alert ──────────────────────────────────────────
    MaintenanceAlert alert = buildAlert(pred, state, cfg);
    if (alert.severity >= AlertSeverity::MONITOR) {
        dispatchAlert(alert);
    }
}

// ---------------------------------------------------------------------------
// IEC 60076-7 thermal state update
// ---------------------------------------------------------------------------

void PredictiveMaintenance::updateThermalState(NodeThermalState& state,
                                                const TransformerConfig& cfg,
                                                float K,
                                                float ambientC,
                                                float dtMinutes) {
    // Steady-state targets
    float dTto_ult = cfg.deltaTheta_to_r
                   * std::pow((K * K * cfg.R + 1.0f) / (cfg.R + 1.0f), cfg.n);
    float dTh_ult  = cfg.deltaTheta_h_r * std::pow(K, 2.0f * cfg.m);
    float thetaTo_ult = ambientC + dTto_ult;
    float thetaH_ult  = thetaTo_ult + dTh_ult;

    // Exponential approach (Euler step)
    float alphaTo = 1.0f - std::exp(-dtMinutes / cfg.tauToMin);
    float alphaH  = 1.0f - std::exp(-dtMinutes / cfg.tauHMin);

    state.thetaToC += alphaTo * (thetaTo_ult - state.thetaToC);
    state.thetaHC  += alphaH  * (thetaH_ult  - state.thetaHC);
    state.K         = K;

    // Relative ageing rate (Arrhenius)
    float thetaH_K = 273.0f + state.thetaHC;
    state.V = std::exp(cfg.arrheniusConst
                       * (1.0f / cfg.thetaRefK - 1.0f / thetaH_K));

    // Accumulate ageing
    state.cumulativeAging += state.V * (dtMinutes / 60.0f);
}

// ---------------------------------------------------------------------------
// Feature vector assembly
// ---------------------------------------------------------------------------

std::vector<float> PredictiveMaintenance::buildFeatureVector(
    const NodeThermalState&             state,
    const TransformerConfig&            cfg,
    const Grid::ThreePhaseMeasurement&  m) const
{
    // Sample counts for each window (assuming 1-min sample interval stored)
    // Must exactly match the Python feature extractor ordering
    auto wMean = [&](const std::deque<float>& buf, size_t n) {
        return windowMean(buf, n);
    };
    auto wMax = [&](const std::deque<float>& buf, size_t n) {
        return windowMax(buf, n);
    };
    auto wStd = [&](const std::deque<float>& buf, size_t n) {
        return windowStd(buf, n);
    };

    std::vector<float> f;
    f.reserve(47);

    // ── Physics features (15) ─────────────────────────────────────────────
    float ambientC = 20.0f;
    f.push_back(state.K);                                    // K
    f.push_back(state.K * state.K);                          // K_sq
    f.push_back(ambientC);                                   // theta_ambient_c
    f.push_back(state.thetaToC);                             // theta_to_c
    f.push_back(state.thetaHC);                              // theta_h_c
    f.push_back(state.thetaToC - ambientC);                  // delta_theta_to_c
    f.push_back(state.thetaHC  - state.thetaToC);            // delta_theta_h_c
    f.push_back(state.V);                                    // V
    f.push_back(state.cumulativeAging);                      // cumulative_ageing_h
    f.push_back(cfg.hotspotContinuousC - state.thetaHC);     // thermal_margin_c
    f.push_back(state.thetaHC / cfg.hotspotLimitC * 100.0f); // hotspot_to_limit_pct

    // Load trend: sign of change (simplified: current K vs 30-sample-ago K)
    float kTrend = 0.0f;
    if (state.kBuffer.size() >= 30) {
        kTrend = std::copysign(
            1.0f, state.K - state.kBuffer[state.kBuffer.size() - 30]
        );
    }
    f.push_back(kTrend);                                     // load_trend
    f.push_back(state.V);                                    // ageing_acceleration
    f.push_back(state.thetaHC > cfg.hotspotContinuousC ? 1.0f : 0.0f); // overtemp_flag
    f.push_back(state.K > 1.0f ? 1.0f : 0.0f);              // overload_flag
    // Total physics features: 15 — matches Python TransformerFeatureExtractor

    // ── Rolling window features (24: 6 stats × 4 windows) ────────────────
    // Windows: 30min=30, 1h=60, 6h=360, 24h=1440 samples
    for (size_t wSamples : {30u, 60u, 360u, 1440u}) {
        f.push_back(wMean(state.kBuffer,       wSamples));  // K_mean
        f.push_back(wMax (state.kBuffer,       wSamples));  // K_max
        f.push_back(wStd (state.kBuffer,       wSamples));  // K_std
        f.push_back(wMean(state.thetaHBuffer,  wSamples));  // theta_h_mean
        f.push_back(wMax (state.thetaHBuffer,  wSamples));  // theta_h_max
        f.push_back(wMean(state.vBuffer,       wSamples));  // V_mean
    }

    // ── Temporal features (4) ─────────────────────────────────────────────
    auto tp = std::chrono::system_clock::from_time_t(
        static_cast<time_t>(m.timestamp / 1000)
    );
    std::time_t tt = std::chrono::system_clock::to_time_t(tp);
    std::tm* lt = std::localtime(&tt);
    f.push_back(lt ? static_cast<float>(lt->tm_hour) : 12.0f);  // hour_of_day
    f.push_back(lt ? static_cast<float>(lt->tm_wday) : 0.0f);   // day_of_week
    f.push_back(lt ? static_cast<float>(lt->tm_mon + 1) : 6.0f);// month
    float hour = lt ? static_cast<float>(lt->tm_hour) : 12.0f;
    f.push_back((hour >= 7.0f && hour <= 22.0f) ? 1.0f : 0.0f); // is_peak_hours

    return f;
}

// ---------------------------------------------------------------------------
// Rolling window statistics
// ---------------------------------------------------------------------------

float PredictiveMaintenance::windowMean(const std::deque<float>& buf,
                                         size_t nSamples) {
    if (buf.empty()) return 0.0f;
    size_t n = std::min(nSamples, buf.size());
    auto it = buf.end() - static_cast<std::ptrdiff_t>(n);
    float sum = std::accumulate(it, buf.end(), 0.0f);
    return sum / static_cast<float>(n);
}

float PredictiveMaintenance::windowMax(const std::deque<float>& buf,
                                        size_t nSamples) {
    if (buf.empty()) return 0.0f;
    size_t n = std::min(nSamples, buf.size());
    auto it = buf.end() - static_cast<std::ptrdiff_t>(n);
    return *std::max_element(it, buf.end());
}

float PredictiveMaintenance::windowStd(const std::deque<float>& buf,
                                        size_t nSamples) {
    if (buf.size() < 2) return 0.0f;
    size_t n = std::min(nSamples, buf.size());
    auto it = buf.end() - static_cast<std::ptrdiff_t>(n);
    float mean = std::accumulate(it, buf.end(), 0.0f) / static_cast<float>(n);
    float var  = 0.0f;
    for (auto jt = it; jt != buf.end(); ++jt)
        var += (*jt - mean) * (*jt - mean);
    return std::sqrt(var / static_cast<float>(n));
}

// ---------------------------------------------------------------------------
// Alert building and dispatch
// ---------------------------------------------------------------------------

MaintenanceAlert PredictiveMaintenance::buildAlert(
    const ThermalPrediction&  pred,
    const NodeThermalState&   state,
    const TransformerConfig&  cfg) const
{
    MaintenanceAlert a;
    a.nodeId               = cfg.nodeId;
    a.nodeTag              = cfg.nodeTag;
    a.rulDays              = pred.rul_days;
    a.ageingState          = pred.ageing_state;
    a.faultProbability     = pred.fault_prob;
    a.hotspotTempC         = state.thetaHC;
    a.relativeAgeingRate   = state.V;
    a.cumulativeAgeingHours= state.cumulativeAging;
    a.timestamp_ms         = pred.timestamp_ms;
    a.valid                = true;

    // Determine severity
    if (pred.fault_imminent || pred.rul_days < 30.0f) {
        a.severity = AlertSeverity::CRITICAL;
        a.recommendedAction =
            "IMMEDIATE INSPECTION REQUIRED. "
            "Schedule replacement within 30 days. "
            "Reduce load to below 80% rated.";
    } else if (pred.rul_days < 90.0f || pred.fault_prob > 0.3f) {
        a.severity = AlertSeverity::WARNING;
        a.recommendedAction =
            "Schedule detailed inspection within 30 days. "
            "Review loading history and ambient conditions.";
    } else if (pred.rul_days < 365.0f || state.V > 2.0f) {
        a.severity = AlertSeverity::MONITOR;
        a.recommendedAction =
            "Increase monitoring frequency. "
            "Review at next scheduled maintenance.";
    } else {
        a.severity = AlertSeverity::NORMAL;
        a.recommendedAction = "No action required. Continue normal monitoring.";
    }

    return a;
}

void PredictiveMaintenance::dispatchAlert(const MaintenanceAlert& alert) {
    LOG_INFO("PredictiveMaintenance",
        "Alert [" + alert.nodeTag + "] "
        + (alert.severity == AlertSeverity::CRITICAL ? "CRITICAL" :
           alert.severity == AlertSeverity::WARNING  ? "WARNING"  :
           alert.severity == AlertSeverity::MONITOR  ? "MONITOR"  : "NORMAL")
        + " RUL=" + std::to_string(static_cast<int>(alert.rulDays)) + "d"
        + " fault_p=" + std::to_string(alert.faultProbability));

    for (auto& cb : mCallbacks) cb(alert);
}

// ---------------------------------------------------------------------------
// Query API
// ---------------------------------------------------------------------------

std::optional<ThermalPrediction>
PredictiveMaintenance::getLatestPrediction(uint8_t nodeId) const {
    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mLastPredictions.find(nodeId);
    if (it == mLastPredictions.end()) return std::nullopt;
    return it->second;
}

std::optional<NodeThermalState>
PredictiveMaintenance::getThermalState(uint8_t nodeId) const {
    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mStates.find(nodeId);
    if (it == mStates.end()) return std::nullopt;
    return it->second;
}

bool PredictiveMaintenance::isModelLoaded() const {
    return mEngine && mEngine->isLoaded();
}

void PredictiveMaintenance::onAlert(MaintenanceAlertCallback cb) {
    std::lock_guard<std::mutex> lock(mMutex);
    mCallbacks.push_back(std::move(cb));
}

uint64_t PredictiveMaintenance::nowMs() const {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<milliseconds>(
            steady_clock::now().time_since_epoch()).count());
}

} // namespace ML
