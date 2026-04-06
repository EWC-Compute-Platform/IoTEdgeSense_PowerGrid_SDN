/**
 * @file fault_detector.h
 * @brief Real-time grid fault and anomaly detection
 *
 * FaultDetector receives ThreePhaseMeasurement and ScalarMeasurement objects
 * and evaluates them against the per-node GridConfig thresholds.
 * Detected events are stored internally and can be retrieved or
 * delivered via a registered callback.
 */

#ifndef FAULT_DETECTOR_H
#define FAULT_DETECTOR_H

#include "grid_types.h"
#include "grid_config.h"
#include <vector>
#include <functional>
#include <map>
#include <mutex>
#include <string>

namespace Grid {

/**
 * @brief Callback type for fault event notification
 */
using FaultCallback = std::function<void(const GridFaultEvent&)>;

/**
 * @brief Real-time fault detector for power grid measurements
 */
class FaultDetector {
public:
    /**
     * @brief Constructor
     * @param config  Grid operational thresholds
     */
    explicit FaultDetector(const GridConfig& config = GridConfig{});

    /**
     * @brief Evaluate a three-phase measurement snapshot
     *
     * Checks voltage, current, frequency, power factor and unbalance.
     * Any faults detected are stored and dispatched to callbacks.
     *
     * @param m   Measurement to evaluate
     * @param nd  Node descriptor supplying rated values
     * @return    Vector of newly raised fault events (may be empty)
     */
    std::vector<GridFaultEvent> evaluate(
        const ThreePhaseMeasurement& m,
        const GridNodeDescriptor&    nd);

    /**
     * @brief Evaluate a scalar measurement (temperature, SoC, tap…)
     *
     * @param s   Scalar measurement
     * @param nd  Node descriptor
     * @return    Vector of newly raised fault events
     */
    std::vector<GridFaultEvent> evaluateScalar(
        const ScalarMeasurement&  s,
        const GridNodeDescriptor& nd);

    /**
     * @brief Register a callback invoked for every new fault event
     */
    void registerCallback(FaultCallback cb);

    /**
     * @brief Get all active (not cleared) fault events
     */
    std::vector<GridFaultEvent> getActiveFaults() const;

    /**
     * @brief Get full fault history (including cleared events)
     */
    std::vector<GridFaultEvent> getFaultHistory() const;

    /**
     * @brief Mark a fault as cleared
     * @return true if the fault was found and cleared
     */
    bool clearFault(uint8_t nodeId, FaultType type);

    /**
     * @brief Clear all faults for a node (e.g. after reset / re-commission)
     */
    void clearAllFaults(uint8_t nodeId);

    /**
     * @brief Update the operational thresholds at runtime
     */
    void updateConfig(const GridConfig& config);

private:
    GridConfig mConfig;
    std::vector<GridFaultEvent>           mHistory;
    std::map<uint64_t, GridFaultEvent>    mActive;   // key = nodeId<<16 | FaultType
    std::vector<FaultCallback>            mCallbacks;
    mutable std::mutex                    mMutex;

    // ── Voltage checks ────────────────────────────────────────────────────
    void checkVoltage(const ThreePhaseMeasurement& m,
                      const GridNodeDescriptor& nd,
                      std::vector<GridFaultEvent>& out);

    void checkVoltageUnbalance(const ThreePhaseMeasurement& m,
                               const GridNodeDescriptor& nd,
                               std::vector<GridFaultEvent>& out);

    // ── Current checks ────────────────────────────────────────────────────
    void checkCurrent(const ThreePhaseMeasurement& m,
                      const GridNodeDescriptor& nd,
                      std::vector<GridFaultEvent>& out);

    void checkCurrentUnbalance(const ThreePhaseMeasurement& m,
                               const GridNodeDescriptor& nd,
                               std::vector<GridFaultEvent>& out);

    // ── Frequency checks ──────────────────────────────────────────────────
    void checkFrequency(const ThreePhaseMeasurement& m,
                        std::vector<GridFaultEvent>& out);

    // ── Power quality checks ──────────────────────────────────────────────
    void checkPowerFactor(const ThreePhaseMeasurement& m,
                          std::vector<GridFaultEvent>& out);

    // ── Thermal checks ────────────────────────────────────────────────────
    void checkTemperature(const ScalarMeasurement& s,
                          std::vector<GridFaultEvent>& out);

    // ── Helpers ───────────────────────────────────────────────────────────
    void raiseOrUpdate(const GridFaultEvent& ev,
                       std::vector<GridFaultEvent>& newEvents);

    GridFaultEvent makeFault(uint8_t nodeId,
                             const std::string& tag,
                             FaultType type,
                             FaultSeverity severity,
                             const std::string& description,
                             float measured,
                             float threshold) const;

    static uint64_t faultKey(uint8_t nodeId, FaultType type) {
        return (static_cast<uint64_t>(nodeId) << 16)
             | static_cast<uint64_t>(type);
    }

    static float voltageUnbalancePct(float va, float vb, float vc);
    static float currentUnbalancePct(float ia, float ib, float ic);
};

} // namespace Grid

#endif // FAULT_DETECTOR_H
