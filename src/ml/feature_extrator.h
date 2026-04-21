/**
 * @file feature_extractor.h
 * @brief C++ feature extraction for ML predictive maintenance
 *
 * Provides two feature extractors that mirror their Python counterparts
 * exactly — same feature order, same formulas, same constants — so that
 * models trained in Python deploy correctly at the C++ edge.
 *
 * ThermalFeatureExtractor
 *   Mirrors: ml/features/transformer_features.py :: TransformerFeatureExtractor
 *   Output:  43-element float vector
 *   Groups:  [0–14] physics (IEC 60076-7)
 *            [15–38] rolling window stats (6 stats × 4 windows)
 *            [39–42] temporal context
 *
 * AnomalyFeatureExtractor
 *   Mirrors: ml/models/anomaly_detector.py :: AnomalyFeatureExtractor
 *   Output:  30-element float vector
 *   Groups:  [0–5]  voltage, [6–10] current, [11–13] frequency,
 *            [14–18] power quality, [19–26] rolling stats, [27–29] interaction
 *
 * Both classes are designed to be called once per measurement cycle and
 * maintain minimal internal state (rolling buffers only).
 */

#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <vector>
#include <deque>
#include <string>
#include <array>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <chrono>

namespace ML {

// ---------------------------------------------------------------------------
// Shared constants — must match Python exactly
// ---------------------------------------------------------------------------

constexpr size_t N_THERMAL_FEATURES = 43;
constexpr size_t N_ANOMALY_FEATURES = 30;

// IEC 60076-7 Arrhenius constants
constexpr float ARRHENIUS_EA_OVER_KB = 15000.0f;   // E_a / k_B
constexpr float THETA_REF_K          = 383.0f;     // 110°C in Kelvin

// Feature group sizes (thermal model — must match ModelConfig in Python)
constexpr size_t N_THERMAL_PHYSICS   = 15;
constexpr size_t N_THERMAL_ROLLING   = 24;   // 6 stats × 4 windows
constexpr size_t N_THERMAL_TEMPORAL  = 4;

// Rolling window sizes in samples (at 1-minute resolution)
constexpr std::array<size_t, 4> THERMAL_WINDOWS = {30, 60, 360, 1440};

// Anomaly feature group sizes
constexpr size_t N_ANOMALY_VOLTAGE   = 6;
constexpr size_t N_ANOMALY_CURRENT   = 5;
constexpr size_t N_ANOMALY_FREQUENCY = 3;
constexpr size_t N_ANOMALY_POWER     = 5;
constexpr size_t N_ANOMALY_ROLLING   = 8;   // 4 stats × 2 windows (5-min, 30-min)
constexpr size_t N_ANOMALY_INTERACT  = 3;


// ---------------------------------------------------------------------------
// IEC 60076-7 thermal state — carried between calls
// ---------------------------------------------------------------------------

/**
 * @brief Persistent thermal state for a single transformer node.
 *
 * Maintained by ThermalFeatureExtractor across measurement calls.
 * One instance per monitored transformer.
 */
struct ThermalState {
    float    thetaToC        = 20.0f;  ///< Top-oil temperature (°C)
    float    thetaHC         = 20.0f;  ///< Hot-spot temperature (°C)
    float    K               = 0.0f;   ///< Last load factor
    float    V               = 1.0f;   ///< Relative ageing rate
    float    cumulativeAging = 0.0f;   ///< Cumulative ageing hours
    uint64_t lastUpdateMs    = 0;

    // Rolling sample buffers (1 sample per call → 1-min resolution assumed)
    std::deque<float> kBuf;     ///< Load factor history
    std::deque<float> thBuf;    ///< Hot-spot temperature history
    std::deque<float> vBuf;     ///< Relative ageing rate history

    static constexpr size_t MAX_BUF = 1440 * 2;  // 2 days

    void push(float k, float theta_h, float v) {
        auto pushOne = [](std::deque<float>& d, float x) {
            d.push_back(x);
            if (d.size() > MAX_BUF) d.pop_front();
        };
        pushOne(kBuf,  k);
        pushOne(thBuf, theta_h);
        pushOne(vBuf,  v);
    }
};


// ---------------------------------------------------------------------------
// Anomaly rolling state — carried between calls
// ---------------------------------------------------------------------------

/**
 * @brief Rolling buffers for the anomaly feature extractor.
 */
struct AnomalyState {
    std::deque<float> vaBuf;    ///< Phase-A voltage per-unit history
    std::deque<float> iAvgBuf;  ///< Average current per-unit history
    std::deque<float> freqBuf;  ///< Frequency history (60-sample window)
    float             lastVaPu  = 1.0f;

    static constexpr size_t MAX_BUF     = 1800;  // 30 min @ 1s
    static constexpr size_t FREQ_WINDOW = 60;

    void pushVoltage(float va_pu) {
        vaBuf.push_back(va_pu);
        if (vaBuf.size() > MAX_BUF) vaBuf.pop_front();
    }
    void pushCurrent(float i_avg_pu) {
        iAvgBuf.push_back(i_avg_pu);
        if (iAvgBuf.size() > MAX_BUF) iAvgBuf.pop_front();
    }
    void pushFreq(float freq_hz) {
        freqBuf.push_back(freq_hz);
        if (freqBuf.size() > FREQ_WINDOW) freqBuf.pop_front();
    }
};


// ---------------------------------------------------------------------------
// Raw measurement inputs (decoupled from grid_types.h for portability)
// ---------------------------------------------------------------------------

/**
 * @brief Minimal three-phase measurement struct for feature extraction.
 * Populated from GridNodeDescriptor + sensor readings.
 */
struct MeasurementInput {
    // Voltages (V RMS)
    float va = 230.0f, vb = 230.0f, vc = 230.0f;
    // Currents (A RMS)
    float ia = 0.0f,   ib = 0.0f,   ic = 0.0f;
    // Frequency (Hz) and rate-of-change (Hz/s)
    float freqHz   = 50.0f;
    float rocoFHz  = 0.0f;
    // Power
    float powerW   = 0.0f;
    float reactiveVar = 0.0f;
    float apparentVa  = 0.0f;
    float powerFactor = 0.95f;
    // Thermal (oil temperature °C, 0 = not measured)
    float topOilC  = 0.0f;
    // Ambient (°C)
    float ambientC = 20.0f;
    // Timestamp (ms since epoch)
    uint64_t timestampMs = 0;
};


// ---------------------------------------------------------------------------
// ThermalFeatureExtractor
// ---------------------------------------------------------------------------

/**
 * @brief Extracts the 43-element thermal feature vector used by
 *        TransformerThermalModel (IEC 60076-7 physics + rolling + temporal).
 *
 * Matches ml/features/transformer_features.py :: TransformerFeatureExtractor
 * exactly in feature order and computation.
 *
 * Usage:
 *   ThermalFeatureConfig cfg;
 *   cfg.ratedCurrentA = 1000.0f;
 *   ThermalFeatureExtractor ext(cfg);
 *   ThermalState state;
 *
 *   // Call once per measurement cycle
 *   auto features = ext.extract(state, meas, dt_minutes);
 */
struct ThermalFeatureConfig {
    float ratedCurrentA      = 1000.0f;
    float nominalVoltageV    = 230.0f;
    float deltaTheta_to_r    = 55.0f;
    float deltaTheta_h_r     = 23.0f;
    float R                  = 6.0f;
    float n                  = 0.8f;
    float m                  = 1.0f;
    float tauToMin           = 210.0f;
    float tauHMin            = 7.0f;
    float hotspotContinuousC = 98.0f;
    float hotspotLimitC      = 140.0f;
};

class ThermalFeatureExtractor {
public:
    explicit ThermalFeatureExtractor(const ThermalFeatureConfig& cfg = {});

    /**
     * @brief Extract 43 thermal features and advance thermal state.
     *
     * @param state     In/out: thermal state updated by this call
     * @param meas      Current measurement values
     * @param dtMinutes Time elapsed since last call (minutes)
     * @return          Feature vector of length N_THERMAL_FEATURES (43)
     */
    std::vector<float> extract(ThermalState&          state,
                                const MeasurementInput& meas,
                                float                   dtMinutes = 1.0f) const;

    /**
     * @brief Compute load factor from measured current.
     * Clipped to [0, 2] for numerical safety.
     */
    float loadFactor(float currentA) const;

    /**
     * @brief IEC 60076-7 relative ageing rate (Arrhenius).
     * V = 1.0 at 110°C. Uses class constants.
     */
    static float relativeAgeingRate(float thetaHC);

    /**
     * @brief Ordered feature names — matches Python get_feature_names().
     */
    static std::vector<std::string> featureNames();

private:
    ThermalFeatureConfig mCfg;

    // IEC 60076-7 dynamic thermal step (Euler integration)
    void thermalStep(ThermalState& state,
                     float K, float ambientC,
                     float dtMinutes) const;

    // Rolling window statistics over the last nSamples elements
    static float wMean(const std::deque<float>& buf, size_t nSamples);
    static float wMax (const std::deque<float>& buf, size_t nSamples);
    static float wStd (const std::deque<float>& buf, size_t nSamples);

    // Current wall-clock hour/day/month from timestamp
    static void wallTime(uint64_t tsMs,
                          int& hour, int& wday, int& month);
};


// ---------------------------------------------------------------------------
// AnomalyFeatureExtractor
// ---------------------------------------------------------------------------

/**
 * @brief Extracts the 30-element anomaly feature vector used by
 *        GridAutoencoder (voltage, current, frequency, power quality,
 *        rolling stats, interaction).
 *
 * Matches ml/models/anomaly_detector.py :: AnomalyFeatureExtractor
 * exactly in feature order and computation.
 *
 * Usage:
 *   AnomalyFeatureConfig cfg;
 *   AnomalyFeatureExtractor ext(cfg);
 *   AnomalyState state;
 *
 *   auto features = ext.extract(state, meas);
 */
struct AnomalyFeatureConfig {
    float ratedCurrentA  = 1000.0f;
    float ratedVoltageV  = 230.0f;
    float ratedPowerW    = 1000000.0f;   // 1 MVA base
    float nominalPF      = 0.95f;
    float nominalFreqHz  = 50.0f;
};

class AnomalyFeatureExtractor {
public:
    explicit AnomalyFeatureExtractor(const AnomalyFeatureConfig& cfg = {});

    /**
     * @brief Extract 30 anomaly features and update rolling state.
     *
     * @param state  In/out: rolling buffers updated by this call
     * @param meas   Current measurement values
     * @return       Feature vector of length N_ANOMALY_FEATURES (30)
     */
    std::vector<float> extract(AnomalyState&          state,
                                const MeasurementInput& meas) const;

    /**
     * @brief Ordered feature names — matches Python ANOMALY_FEATURE_NAMES.
     */
    static std::vector<std::string> featureNames();

private:
    AnomalyFeatureConfig mCfg;

    static float unbalancePct(float a, float b, float c);
    static float wMean(const std::deque<float>& buf, size_t n);
    static float wMax (const std::deque<float>& buf, size_t n);
    static float wStd (const std::deque<float>& buf, size_t n);
};


// ---------------------------------------------------------------------------
// FeatureExtractorFactory — convenience wrapper owning both extractors
// ---------------------------------------------------------------------------

/**
 * @brief Owns and coordinates ThermalFeatureExtractor + AnomalyFeatureExtractor.
 *
 * One instance per monitored node. Holds state between calls.
 * Called from PredictiveMaintenance::ingest() via the node state map.
 */
class FeatureExtractorFactory {
public:
    FeatureExtractorFactory(const ThermalFeatureConfig& thermalCfg = {},
                             const AnomalyFeatureConfig& anomalyCfg = {});

    /**
     * @brief Extract both feature vectors in one call.
     *
     * @param meas       Current measurement
     * @param dtMinutes  Time since last call (minutes)
     * @param thermal    Output: 43-element thermal vector
     * @param anomaly    Output: 30-element anomaly vector
     */
    void extractAll(const MeasurementInput& meas,
                    float                   dtMinutes,
                    std::vector<float>&     thermal,
                    std::vector<float>&     anomaly);

    const ThermalState& thermalState() const { return mThermalState; }
    const AnomalyState& anomalyState() const { return mAnomalyState; }
    void reset();

private:
    ThermalFeatureExtractor mThermalExt;
    AnomalyFeatureExtractor mAnomalyExt;
    ThermalState            mThermalState;
    AnomalyState            mAnomalyState;
};

} // namespace ML

#endif // FEATURE_EXTRACTOR_H

