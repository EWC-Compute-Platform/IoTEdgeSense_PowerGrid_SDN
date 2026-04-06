/**
 * @file power_grid_bridge.h
 * @brief Bridge between IoTEdgeSense sensor readings and the power grid domain
 *
 * PowerGridBridge is the central integration class. It:
 *
 *  1. Accepts raw SensorReading objects from any IoTEdgeSense sensor
 *  2. Decodes them into typed ThreePhaseMeasurement / ScalarMeasurement
 *     objects using the sensor's known values[] layout contract
 *  3. Passes measurements through the FaultDetector
 *  4. Publishes structured grid telemetry to the SDN platform via
 *     CommManager (MQTT topics) or the GridTelemetryPublisher (REST/gRPC)
 *  5. Exposes a query API so upstream components can get the latest
 *     measurement snapshot per node
 */

#ifndef POWER_GRID_BRIDGE_H
#define POWER_GRID_BRIDGE_H

#include "sensors/sensor_base.h"
#include "comm/comm_manager.h"
#include "grid_types.h"
#include "grid_config.h"
#include "fault_detector.h"
#include <map>
#include <vector>
#include <memory>
#include <mutex>
#include <functional>

namespace Grid {

/**
 * @brief Sensor role within the bridge — determines how values[] is decoded
 */
enum class SensorRole : uint8_t {
    VOLTAGE,        ///< values[] = [Va, Vb, Vc, Vab, Vbc, Vca]
    CURRENT,        ///< values[] = [Ia, Ib, Ic]
    FREQUENCY,      ///< values[] = [Hz, dHz_dt]
    POWER_METER,    ///< values[] = [W, VAR, VA, PF, angA, angB, angC]
    THERMAL,        ///< values[] = [temp_C]
    GENERIC         ///< Pass-through, no grid decoding
};

/**
 * @brief Registration record binding a sensor to a grid node
 */
struct SensorRegistration {
    uint8_t     sensorId;
    SensorRole  role;
    uint8_t     gridNodeId;  ///< Which GridNodeDescriptor this sensor serves
};

/**
 * @brief Callback invoked when a new ThreePhaseMeasurement is assembled
 */
using MeasurementCallback =
    std::function<void(const ThreePhaseMeasurement&)>;

/**
 * @brief Callback invoked when a fault event is detected
 */
using GridFaultCallback =
    std::function<void(const GridFaultEvent&)>;

// ===========================================================================

class PowerGridBridge {
public:
    /**
     * @brief Constructor
     *
     * @param comm    Shared CommManager used for MQTT publishing
     * @param config  Grid operational thresholds
     */
    PowerGridBridge(std::shared_ptr<Communication::CommManager> comm,
                    const GridConfig& config = GridConfig{});

    ~PowerGridBridge() = default;

    // ── Node registration ──────────────────────────────────────────────────

    /**
     * @brief Register a grid node descriptor
     *
     * Must be called before registering sensors for that node.
     */
    void registerNode(const GridNodeDescriptor& nd);

    /**
     * @brief Register a sensor and its role for a given grid node
     *
     * @param sensorId    ID used in SensorReading::sensorId
     * @param role        How the values[] vector should be decoded
     * @param gridNodeId  Target GridNodeDescriptor nodeId
     */
    void registerSensor(uint8_t sensorId,
                        SensorRole role,
                        uint8_t gridNodeId);

    // ── Data ingestion ────────────────────────────────────────────────────

    /**
     * @brief Feed a single sensor reading into the bridge
     *
     * The bridge will decode it, update the measurement snapshot for
     * the associated node, run fault detection, and publish telemetry.
     *
     * @param reading   Raw reading from any registered sensor
     * @return true if the reading was processed (sensor was registered)
     */
    bool ingest(const Sensors::SensorReading& reading);

    /**
     * @brief Feed a batch of readings
     */
    void ingestBatch(const std::vector<Sensors::SensorReading>& readings);

    // ── Query API ─────────────────────────────────────────────────────────

    /**
     * @brief Get the latest assembled measurement for a node
     *
     * @param nodeId  Grid node ID
     * @return Measurement snapshot (valid=false if no data yet)
     */
    ThreePhaseMeasurement getLatestMeasurement(uint8_t nodeId) const;

    /**
     * @brief Get all active fault events across all nodes
     */
    std::vector<GridFaultEvent> getActiveFaults() const;

    /**
     * @brief Get fault history for a specific node
     */
    std::vector<GridFaultEvent> getFaultHistory(uint8_t nodeId) const;

    // ── Callbacks ─────────────────────────────────────────────────────────

    /**
     * @brief Register a callback for assembled measurements
     */
    void onMeasurement(MeasurementCallback cb);

    /**
     * @brief Register a callback for fault events
     */
    void onFault(GridFaultCallback cb);

    // ── Configuration ─────────────────────────────────────────────────────

    void updateConfig(const GridConfig& config);

    /**
     * @brief Enable or disable MQTT telemetry publishing
     */
    void setPublishingEnabled(bool enabled);

    /**
     * @brief Override the MQTT topic prefix (default: "grid/telemetry")
     */
    void setTopicPrefix(const std::string& prefix);

private:
    std::shared_ptr<Communication::CommManager> mComm;
    FaultDetector                               mFaultDetector;

    std::map<uint8_t, GridNodeDescriptor>     mNodes;       // nodeId → descriptor
    std::map<uint8_t, SensorRegistration>     mSensors;     // sensorId → registration
    std::map<uint8_t, ThreePhaseMeasurement>  mSnapshots;   // nodeId → latest measurement

    std::vector<MeasurementCallback> mMeasurementCbs;
    std::vector<GridFaultCallback>   mFaultCbs;

    mutable std::mutex mMutex;

    bool        mPublishEnabled = true;
    std::string mTopicPrefix    = "grid/telemetry";

    // ── Decode helpers ────────────────────────────────────────────────────

    void decodeVoltage   (const Sensors::SensorReading& r,
                          ThreePhaseMeasurement& m);
    void decodeCurrent   (const Sensors::SensorReading& r,
                          ThreePhaseMeasurement& m);
    void decodeFrequency (const Sensors::SensorReading& r,
                          ThreePhaseMeasurement& m);
    void decodePowerMeter(const Sensors::SensorReading& r,
                          ThreePhaseMeasurement& m);

    // ── Serialisation helpers ─────────────────────────────────────────────

    static std::string measurementToJson(const ThreePhaseMeasurement& m);
    static std::string faultToJson(const GridFaultEvent& ev);
    static std::string faultTypeString(FaultType t);
    static std::string faultSeverityString(FaultSeverity s);
    static std::string nodeTypeString(NodeType t);

    // ── Publishing ────────────────────────────────────────────────────────

    void publishMeasurement(const ThreePhaseMeasurement& m,
                            const GridNodeDescriptor& nd);
    void publishFault(const GridFaultEvent& ev);
};

} // namespace Grid

#endif // POWER_GRID_BRIDGE_H
