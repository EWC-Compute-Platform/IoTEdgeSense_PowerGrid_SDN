/**
 * @file power_grid_bridge.cpp
 * @brief PowerGridBridge implementation
 */

#include "power_grid_bridge.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace Grid {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

PowerGridBridge::PowerGridBridge(
    std::shared_ptr<Communication::CommManager> comm,
    const GridConfig& config)
    : mComm(std::move(comm))
    , mFaultDetector(config)
{
    // Forward fault events from the detector to our registered callbacks
    mFaultDetector.registerCallback([this](const GridFaultEvent& ev) {
        for (auto& cb : mFaultCbs) cb(ev);
        publishFault(ev);
    });
}

// ---------------------------------------------------------------------------
// Node and sensor registration
// ---------------------------------------------------------------------------

void PowerGridBridge::registerNode(const GridNodeDescriptor& nd) {
    std::lock_guard<std::mutex> lock(mMutex);
    mNodes[nd.nodeId] = nd;
    // Initialise an empty snapshot
    ThreePhaseMeasurement snap;
    snap.nodeId  = nd.nodeId;
    snap.nodeTag = nd.nodeTag;
    mSnapshots[nd.nodeId] = snap;
}

void PowerGridBridge::registerSensor(uint8_t sensorId,
                                      SensorRole role,
                                      uint8_t gridNodeId) {
    std::lock_guard<std::mutex> lock(mMutex);
    mSensors[sensorId] = { sensorId, role, gridNodeId };
}

// ---------------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------------

bool PowerGridBridge::ingest(const Sensors::SensorReading& reading) {
    if (!reading.valid) return false;

    std::lock_guard<std::mutex> lock(mMutex);

    auto sit = mSensors.find(reading.sensorId);
    if (sit == mSensors.end()) return false;  // Not registered

    const SensorRegistration& reg = sit->second;
    auto nit = mNodes.find(reg.gridNodeId);
    if (nit == mNodes.end()) return false;

    const GridNodeDescriptor& nd = nit->second;
    ThreePhaseMeasurement& snap  = mSnapshots[reg.gridNodeId];
    snap.nodeId  = nd.nodeId;
    snap.nodeTag = nd.nodeTag;
    snap.valid   = true;

    switch (reg.role) {
        case SensorRole::VOLTAGE:     decodeVoltage   (reading, snap); break;
        case SensorRole::CURRENT:     decodeCurrent   (reading, snap); break;
        case SensorRole::FREQUENCY:   decodeFrequency (reading, snap); break;
        case SensorRole::POWER_METER: decodePowerMeter(reading, snap); break;
        case SensorRole::THERMAL: {
            ScalarMeasurement sm;
            sm.nodeId       = nd.nodeId;
            sm.nodeTag      = nd.nodeTag;
            sm.timestamp    = reading.timestamp;
            sm.quantityName = "oil_temp";
            sm.value        = reading.values.empty() ? 0.0f : reading.values[0];
            sm.unit         = "degC";
            sm.valid        = true;
            auto faults = mFaultDetector.evaluateScalar(sm, nd);
            (void)faults;
            // Thermal readings don't update ThreePhaseMeasurement
            return true;
        }
        default: return false;
    }

    // Run fault detection on updated snapshot
    auto faults = mFaultDetector.evaluate(snap, nd);
    (void)faults;  // callbacks handle delivery

    // Notify measurement subscribers
    for (auto& cb : mMeasurementCbs) cb(snap);

    // Publish
    if (mPublishEnabled) publishMeasurement(snap, nd);

    return true;
}

void PowerGridBridge::ingestBatch(
    const std::vector<Sensors::SensorReading>& readings) {
    for (const auto& r : readings) ingest(r);
}

// ---------------------------------------------------------------------------
// Decode helpers
// ---------------------------------------------------------------------------

void PowerGridBridge::decodeVoltage(const Sensors::SensorReading& r,
                                     ThreePhaseMeasurement& m) {
    // Contract: [Va, Vb, Vc, Vab, Vbc, Vca]
    m.timestamp = r.timestamp;
    if (r.values.size() >= 1) { m.voltageA_V = r.values[0]; m.phaseAPresent = true; }
    if (r.values.size() >= 2) { m.voltageB_V = r.values[1]; m.phaseBPresent = true; }
    if (r.values.size() >= 3) { m.voltageC_V = r.values[2]; m.phaseCPresent = true; }
}

void PowerGridBridge::decodeCurrent(const Sensors::SensorReading& r,
                                     ThreePhaseMeasurement& m) {
    // Contract: [Ia, Ib, Ic]
    m.timestamp = r.timestamp;
    if (r.values.size() >= 1) m.currentA_A = r.values[0];
    if (r.values.size() >= 2) m.currentB_A = r.values[1];
    if (r.values.size() >= 3) m.currentC_A = r.values[2];
}

void PowerGridBridge::decodeFrequency(const Sensors::SensorReading& r,
                                       ThreePhaseMeasurement& m) {
    // Contract: [Hz, dHz_dt]
    m.timestamp = r.timestamp;
    if (r.values.size() >= 1) m.frequency_Hz = r.values[0];
}

void PowerGridBridge::decodePowerMeter(const Sensors::SensorReading& r,
                                        ThreePhaseMeasurement& m) {
    // Contract: [W, VAR, VA, PF, angA, angB, angC]
    m.timestamp = r.timestamp;
    if (r.values.size() >= 1) m.activePower_W     = r.values[0];
    if (r.values.size() >= 2) m.reactivePower_VAR = r.values[1];
    if (r.values.size() >= 3) m.apparentPower_VA  = r.values[2];
    if (r.values.size() >= 4) m.powerFactor       = r.values[3];
    if (r.values.size() >= 5) m.angleA_deg        = r.values[4];
    if (r.values.size() >= 6) m.angleB_deg        = r.values[5];
    if (r.values.size() >= 7) m.angleC_deg        = r.values[6];
}

// ---------------------------------------------------------------------------
// Query API
// ---------------------------------------------------------------------------

ThreePhaseMeasurement
PowerGridBridge::getLatestMeasurement(uint8_t nodeId) const {
    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mSnapshots.find(nodeId);
    if (it == mSnapshots.end()) return ThreePhaseMeasurement{};
    return it->second;
}

std::vector<GridFaultEvent> PowerGridBridge::getActiveFaults() const {
    return mFaultDetector.getActiveFaults();
}

std::vector<GridFaultEvent>
PowerGridBridge::getFaultHistory(uint8_t nodeId) const {
    auto all = mFaultDetector.getFaultHistory();
    std::vector<GridFaultEvent> filtered;
    for (const auto& ev : all)
        if (ev.nodeId == nodeId) filtered.push_back(ev);
    return filtered;
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void PowerGridBridge::onMeasurement(MeasurementCallback cb) {
    std::lock_guard<std::mutex> lock(mMutex);
    mMeasurementCbs.push_back(std::move(cb));
}

void PowerGridBridge::onFault(GridFaultCallback cb) {
    std::lock_guard<std::mutex> lock(mMutex);
    mFaultCbs.push_back(std::move(cb));
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

void PowerGridBridge::updateConfig(const GridConfig& config) {
    mFaultDetector.updateConfig(config);
}

void PowerGridBridge::setPublishingEnabled(bool enabled) {
    mPublishEnabled = enabled;
}

void PowerGridBridge::setTopicPrefix(const std::string& prefix) {
    mTopicPrefix = prefix;
}

// ---------------------------------------------------------------------------
// Publishing
// ---------------------------------------------------------------------------

void PowerGridBridge::publishMeasurement(const ThreePhaseMeasurement& m,
                                          const GridNodeDescriptor& nd) {
    if (!mComm || !mComm->isConnected()) return;

    std::string topic  = mTopicPrefix + "/" + nd.nodeTag + "/measurement";
    std::string payload = measurementToJson(m);
    mComm->publishStatus(payload);  // reuse status publish; override topic via direct API if needed
    // Note: for production, add a publishRaw(topic, payload) to CommManager
    (void)topic;
}

void PowerGridBridge::publishFault(const GridFaultEvent& ev) {
    if (!mComm || !mComm->isConnected()) return;

    std::string topic  = mTopicPrefix + "/" + ev.nodeTag + "/fault";
    std::string payload = faultToJson(ev);
    mComm->publishStatus(payload);
    (void)topic;
}

// ---------------------------------------------------------------------------
// JSON serialisation
// ---------------------------------------------------------------------------

std::string PowerGridBridge::measurementToJson(const ThreePhaseMeasurement& m) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(4);
    o << "{"
      << "\"ts\":"       << m.timestamp         << ","
      << "\"node\":\""   << m.nodeTag            << "\","
      << "\"nodeId\":"   << static_cast<int>(m.nodeId) << ","
      << "\"Va\":"       << m.voltageA_V         << ","
      << "\"Vb\":"       << m.voltageB_V         << ","
      << "\"Vc\":"       << m.voltageC_V         << ","
      << "\"Ia\":"       << m.currentA_A         << ","
      << "\"Ib\":"       << m.currentB_A         << ","
      << "\"Ic\":"       << m.currentC_A         << ","
      << "\"Hz\":"       << m.frequency_Hz       << ","
      << "\"W\":"        << m.activePower_W      << ","
      << "\"VAR\":"      << m.reactivePower_VAR  << ","
      << "\"VA\":"       << m.apparentPower_VA   << ","
      << "\"PF\":"       << m.powerFactor        << ","
      << "\"valid\":"    << (m.valid ? "true" : "false")
      << "}";
    return o.str();
}

std::string PowerGridBridge::faultToJson(const GridFaultEvent& ev) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(4);
    o << "{"
      << "\"ts\":"          << ev.timestamp                  << ","
      << "\"node\":\""      << ev.nodeTag                    << "\","
      << "\"nodeId\":"      << static_cast<int>(ev.nodeId)   << ","
      << "\"type\":\""      << faultTypeString(ev.type)      << "\","
      << "\"severity\":\""  << faultSeverityString(ev.severity) << "\","
      << "\"desc\":\""      << ev.description                << "\","
      << "\"measured\":"    << ev.measuredValue              << ","
      << "\"threshold\":"   << ev.threshold                  << ","
      << "\"cleared\":"     << (ev.cleared ? "true" : "false")
      << "}";
    return o.str();
}

std::string PowerGridBridge::faultTypeString(FaultType t) {
    switch (t) {
        case FaultType::OVER_VOLTAGE:         return "OVER_VOLTAGE";
        case FaultType::UNDER_VOLTAGE:        return "UNDER_VOLTAGE";
        case FaultType::VOLTAGE_UNBALANCE:    return "VOLTAGE_UNBALANCE";
        case FaultType::VOLTAGE_SAG:          return "VOLTAGE_SAG";
        case FaultType::VOLTAGE_SWELL:        return "VOLTAGE_SWELL";
        case FaultType::VOLTAGE_INTERRUPT:    return "VOLTAGE_INTERRUPT";
        case FaultType::OVERCURRENT:          return "OVERCURRENT";
        case FaultType::EARTH_FAULT:          return "EARTH_FAULT";
        case FaultType::SHORT_CIRCUIT:        return "SHORT_CIRCUIT";
        case FaultType::CURRENT_UNBALANCE:    return "CURRENT_UNBALANCE";
        case FaultType::OVER_FREQUENCY:       return "OVER_FREQUENCY";
        case FaultType::UNDER_FREQUENCY:      return "UNDER_FREQUENCY";
        case FaultType::FREQUENCY_INSTABLE:   return "FREQUENCY_INSTABLE";
        case FaultType::HIGH_HARMONICS:       return "HIGH_HARMONICS";
        case FaultType::LOW_POWER_FACTOR:     return "LOW_POWER_FACTOR";
        case FaultType::FLICKER:              return "FLICKER";
        case FaultType::TRANSFORMER_OVERHEAT: return "TRANSFORMER_OVERHEAT";
        case FaultType::CABLE_OVERHEAT:       return "CABLE_OVERHEAT";
        case FaultType::COMMUNICATION_LOSS:   return "COMMUNICATION_LOSS";
        case FaultType::SENSOR_FAILURE:       return "SENSOR_FAILURE";
        default:                              return "NONE";
    }
}

std::string PowerGridBridge::faultSeverityString(FaultSeverity s) {
    switch (s) {
        case FaultSeverity::INFO:     return "INFO";
        case FaultSeverity::WARNING:  return "WARNING";
        case FaultSeverity::ALARM:    return "ALARM";
        case FaultSeverity::CRITICAL: return "CRITICAL";
        default:                      return "UNKNOWN";
    }
}

std::string PowerGridBridge::nodeTypeString(NodeType t) {
    switch (t) {
        case NodeType::SUBSTATION:       return "SUBSTATION";
        case NodeType::DISTRIBUTION_BUS: return "DISTRIBUTION_BUS";
        case NodeType::FEEDER_LINE:      return "FEEDER_LINE";
        case NodeType::TRANSFORMER:      return "TRANSFORMER";
        case NodeType::LOAD_POINT:       return "LOAD_POINT";
        case NodeType::GENERATION:       return "GENERATION";
        case NodeType::STORAGE:          return "STORAGE";
        default:                         return "MEASUREMENT_ONLY";
    }
}

} // namespace Grid


