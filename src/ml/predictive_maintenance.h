/**
 * @file predictive_maintenance.h
 * @brief Predictive maintenance module for the IoTEdgeSense platform
 *
 * PredictiveMaintenance is the integration point between the IoTEdgeSense
 * firmware pipeline and the ONNX ML inference engine.
 *
 * It sits between DataProcessor and CommManager in the data flow:
 *
 *   SensorReading (raw)
 *       → DataProcessor (filter pipeline)
 *       → PredictiveMaintenance (feature extraction + ONNX inference)
 *       → MaintenanceAlert (published via CommManager)
 *
 * The module maintains a rolling feature buffer per node, computes the
 * IEC 60076-7 physics features in C++ (matching the Python extractor),
 * invokes the ONNX model, and emits MaintenanceAlert events when
 * predictions cross configured thresholds.
 *
 * Build flags:
 *   -DONNX_STUB    Stub inference (no ONNX Runtime needed)
 */

#ifndef PREDICTIVE_MAINTENANCE_H
#define PREDICTIVE_MAINTENANCE_H

#include "onnx_inference.h"
#include "../sensors/sensor_base.h"
#include "../grid/grid_types.h"
#include "../system/error_handler.h"
#include "../system/logger.h"

#include <map>
#include <deque>
#include <vector>
#include <string>
#include <functional>
#include <mutex>
#include <memory>
#include <optional>
#include <cmath>

namespace ML {

// ---------------------------------------------------------------------------
// Transformer configuration (mirrors Python TransformerRating)
// ---------------------------------------------------------------------------

struct TransformerConfig {
    std::string nodeTag;              ///< Matches GridNodeDescriptor::nodeTag
    uint8_t     nodeId       = 0;

    // IEC 60076-7 thermal parameters
    float ratedCurrentA      = 1000.0f;
    float nominalVoltageV    = 11000.0f;
    float deltaTheta_to_r    = 55.0f;    ///< Rated top-oil rise (K)
    float deltaTheta_h_r     = 23.0f;    ///< Rated hot-spot rise over top-oil (K)
    float R                  = 6.0f;     ///< Load loss / no-load loss ratio
    float n                  = 0.8f;     ///< Top-oil thermal exponent
    float m                  = 1.0f;     ///< Winding exponent
    float tauToMin           = 210.0f;   ///< Top-oil time constant (min)
    float tauHMin            = 7.0f;     ///< Hot-spot time constant (min)
    float hotspotContinuousC = 98.0f;    ///< Continuous hot-spot limit (°C)
    float hotspotLimitC      = 140.0f;   ///< Absolute hot-spot limit (°C)
    float arrheniusConst     = 15000.0f; ///< E_a/k_B
    float thetaRefK          = 383.0f;   ///< Reference temperature (K)

    // Inference scheduling
    uint32_t inferenceIntervalSec = 900;  ///< Run ML every 15 min
};

// ---------------------------------------------------------------------------
// Maintenance alert
// ---------------------------------------------------------------------------

enum class AlertSeverity : uint8_t {
    NORMAL   = 0,
    MONITOR  = 1,
    WARNING  = 2,
    CRITICAL = 3
};

struct MaintenanceAlert {
    uint8_t      nodeId;
    std::string  nodeTag;
    AlertSeverity severity;
    float        rulDays;
    float        ageingState;
    float        faultProbability;
    float        hotspotTempC;
    float        relativeAgeingRate;    ///< V from IEC 60076-7
    float        cumulativeAgeingHours;
    uint64_t     timestamp_ms;
    std::string  recommendedAction;
    bool         valid = false;

    std::string toJson() const;
};

using MaintenanceAlertCallback = std::function<void(const MaintenanceAlert&)>;

// ---------------------------------------------------------------------------
// Node thermal state (maintained per transformer)
// ---------------------------------------------------------------------------

struct NodeThermalState {
    // Current IEC 60076-7 dynamic state
    float thetaToC        = 20.0f;   ///< Top-oil temperature (°C)
    float thetaHC         = 20.0f;   ///< Hot-spot temperature (°C)
    float cumulativeAging = 0.0f;    ///< Hours of ageing equivalent
    float K               = 0.0f;    ///< Current load factor
    float V               = 1.0f;    ///< Current relative ageing rate
    uint64_t lastUpdateMs = 0;

    // Rolling buffers for window features (30min → 24h)
    // Stored as circular buffers of (K, theta_h, V) tuples
    std::deque<float> kBuffer;        ///< Load factor history
    std::deque<float> thetaHBuffer;   ///< Hot-spot history
    std::deque<float> vBuffer;        ///< Ageing rate history

    static constexpr size_t MAX_BUFFER = 1440 * 4;  // 4 days @ 1min resolution

    void pushSample(float k, float thetaH, float v) {
        kBuffer.push_back(k);
        thetaHBuffer.push_back(thetaH);
        vBuffer.push_back(v);
        if (kBuffer.size()      > MAX_BUFFER) kBuffer.pop_front();
        if (thetaHBuffer.size() > MAX_BUFFER) thetaHBuffer.pop_front();
        if (vBuffer.size()      > MAX_BUFFER) vBuffer.pop_front();
    }
};

// ---------------------------------------------------------------------------
// PredictiveMaintenance
// ---------------------------------------------------------------------------

class PredictiveMaintenance {
public:
    /**
     * @brief Constructor
     *
     * @param modelPath    Path to exported ONNX model file
     * @param nFeatures    Feature vector length (must match Python export)
     */
    explicit PredictiveMaintenance(const std::string& modelPath = "",
                                    size_t nFeatures = 43);
    ~PredictiveMaintenance() = default;

    /**
     * @brief Initialise — load ONNX model.
     * @return true on success (or stub mode)
     */
    bool initialize();

    /**
     * @brief Register a transformer node for monitoring.
     */
    void registerTransformer(const TransformerConfig& cfg);

    /**
     * @brief Feed a new ThreePhaseMeasurement into the module.
     *
     * Called by PowerGridBridge on every assembled measurement.
     * Internally: updates thermal state, checks inference schedule,
     * runs ONNX if due, emits alerts if thresholds crossed.
     *
     * @param m   The assembled measurement
     */
    void ingest(const Grid::ThreePhaseMeasurement& m);

    /**
     * @brief Register a callback for maintenance alerts.
     */
    void onAlert(MaintenanceAlertCallback cb);

    /**
     * @brief Get latest prediction for a node (nullopt if never inferred).
     */
    std::optional<ThermalPrediction> getLatestPrediction(uint8_t nodeId) const;

    /**
     * @brief Get current thermal state for a node.
     */
    std::optional<NodeThermalState> getThermalState(uint8_t nodeId) const;

    bool isModelLoaded() const;

private:
    std::string            mModelPath;
    size_t                 mNFeatures;
    std::unique_ptr<OnnxInferenceEngine> mEngine;

    std::map<uint8_t, TransformerConfig>  mConfigs;
    std::map<uint8_t, NodeThermalState>   mStates;
    std::map<uint8_t, ThermalPrediction>  mLastPredictions;
    std::map<uint8_t, uint64_t>           mLastInferenceMs;

    std::vector<MaintenanceAlertCallback> mCallbacks;
    mutable std::mutex mMutex;

    // ── IEC 60076-7 thermal step (mirrors Python IEC60076_7_ThermalModel) ─
    void updateThermalState(NodeThermalState& state,
                             const TransformerConfig& cfg,
                             float K,
                             float ambientC,
                             float dtMinutes);

    // ── Feature vector assembly ──────────────────────────────────────────
    std::vector<float> buildFeatureVector(
        const NodeThermalState& state,
        const TransformerConfig& cfg,
        const Grid::ThreePhaseMeasurement& m
    ) const;

    // ── Rolling window statistics ────────────────────────────────────────
    static float windowMean(const std::deque<float>& buf, size_t nSamples);
    static float windowMax (const std::deque<float>& buf, size_t nSamples);
    static float windowStd (const std::deque<float>& buf, size_t nSamples);

    // ── Alert dispatch ───────────────────────────────────────────────────
    void dispatchAlert(const MaintenanceAlert& alert);
    MaintenanceAlert buildAlert(
        const ThermalPrediction&  pred,
        const NodeThermalState&   state,
        const TransformerConfig&  cfg
    ) const;

    uint64_t nowMs() const;
};

} // namespace ML

#endif // PREDICTIVE_MAINTENANCE_H
