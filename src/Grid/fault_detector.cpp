/**
 * @file fault_detector.cpp
 * @brief FaultDetector implementation
 */

#include "fault_detector.h"
#include <cmath>
#include <algorithm>
#include <chrono>

namespace Grid {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static uint64_t tsNow() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<milliseconds>(
            steady_clock::now().time_since_epoch()).count());
}

float FaultDetector::voltageUnbalancePct(float va, float vb, float vc) {
    float avg = (va + vb + vc) / 3.0f;
    if (avg < 1e-6f) return 0.0f;
    float maxDev = std::max({std::fabs(va - avg),
                              std::fabs(vb - avg),
                              std::fabs(vc - avg)});
    return (maxDev / avg) * 100.0f;
}

float FaultDetector::currentUnbalancePct(float ia, float ib, float ic) {
    return voltageUnbalancePct(ia, ib, ic);   // same formula
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

FaultDetector::FaultDetector(const GridConfig& config)
    : mConfig(config)
{}

// ---------------------------------------------------------------------------
// Configuration update
// ---------------------------------------------------------------------------

void FaultDetector::updateConfig(const GridConfig& config) {
    std::lock_guard<std::mutex> lock(mMutex);
    mConfig = config;
}

// ---------------------------------------------------------------------------
// Callback registration
// ---------------------------------------------------------------------------

void FaultDetector::registerCallback(FaultCallback cb) {
    std::lock_guard<std::mutex> lock(mMutex);
    mCallbacks.push_back(std::move(cb));
}

// ---------------------------------------------------------------------------
// Main evaluation entry point
// ---------------------------------------------------------------------------

std::vector<GridFaultEvent>
FaultDetector::evaluate(const ThreePhaseMeasurement& m,
                        const GridNodeDescriptor& nd) {
    if (!m.valid) return {};

    std::vector<GridFaultEvent> newFaults;
    checkVoltage         (m, nd, newFaults);
    checkVoltageUnbalance(m, nd, newFaults);
    checkCurrent         (m, nd, newFaults);
    checkCurrentUnbalance(m, nd, newFaults);
    checkFrequency       (m,     newFaults);
    checkPowerFactor     (m,     newFaults);

    return newFaults;
}

std::vector<GridFaultEvent>
FaultDetector::evaluateScalar(const ScalarMeasurement& s,
                               const GridNodeDescriptor& nd) {
    (void)nd;
    std::vector<GridFaultEvent> newFaults;
    if (s.quantityName == "oil_temp" || s.quantityName == "cable_temp") {
        checkTemperature(s, newFaults);
    }
    return newFaults;
}

// ---------------------------------------------------------------------------
// Voltage checks
// ---------------------------------------------------------------------------

void FaultDetector::checkVoltage(const ThreePhaseMeasurement& m,
                                  const GridNodeDescriptor& nd,
                                  std::vector<GridFaultEvent>& out) {
    float nom = nd.nominalVoltage_V;
    if (nom < 1e-6f) return;

    auto check = [&](float v, const std::string& phase) {
        float pct = ((v - nom) / nom) * 100.0f;
        if (pct > mConfig.voltageOverPct) {
            out.push_back(makeFault(
                nd.nodeId, nd.nodeTag, FaultType::OVER_VOLTAGE,
                FaultSeverity::ALARM,
                "Over-voltage on phase " + phase + ": "
                    + std::to_string(v) + " V",
                v, nom * (1.0f + mConfig.voltageOverPct / 100.0f)));
        } else if (pct > mConfig.voltageHighPct) {
            out.push_back(makeFault(
                nd.nodeId, nd.nodeTag, FaultType::OVER_VOLTAGE,
                FaultSeverity::WARNING,
                "High voltage on phase " + phase + ": "
                    + std::to_string(v) + " V",
                v, nom * (1.0f + mConfig.voltageHighPct / 100.0f)));
        } else if (pct < -mConfig.voltageUnderPct) {
            out.push_back(makeFault(
                nd.nodeId, nd.nodeTag, FaultType::UNDER_VOLTAGE,
                FaultSeverity::ALARM,
                "Under-voltage on phase " + phase + ": "
                    + std::to_string(v) + " V",
                v, nom * (1.0f - mConfig.voltageUnderPct / 100.0f)));
        } else if (pct < -mConfig.voltageLowPct) {
            out.push_back(makeFault(
                nd.nodeId, nd.nodeTag, FaultType::UNDER_VOLTAGE,
                FaultSeverity::WARNING,
                "Low voltage on phase " + phase + ": "
                    + std::to_string(v) + " V",
                v, nom * (1.0f - mConfig.voltageLowPct / 100.0f)));
        }
    };

    if (m.phaseAPresent) check(m.voltageA_V, "A");
    if (m.phaseBPresent) check(m.voltageB_V, "B");
    if (m.phaseCPresent) check(m.voltageC_V, "C");

    for (auto& ev : out) raiseOrUpdate(ev, out);
}

void FaultDetector::checkVoltageUnbalance(const ThreePhaseMeasurement& m,
                                           const GridNodeDescriptor& nd,
                                           std::vector<GridFaultEvent>& out) {
    if (!m.phaseAPresent || !m.phaseBPresent || !m.phaseCPresent) return;
    float pct = voltageUnbalancePct(m.voltageA_V, m.voltageB_V, m.voltageC_V);
    if (pct > mConfig.voltageUnbalancePct) {
        auto ev = makeFault(nd.nodeId, nd.nodeTag,
            FaultType::VOLTAGE_UNBALANCE, FaultSeverity::WARNING,
            "Voltage unbalance " + std::to_string(pct) + " %",
            pct, mConfig.voltageUnbalancePct);
        raiseOrUpdate(ev, out);
    }
}

// ---------------------------------------------------------------------------
// Current checks
// ---------------------------------------------------------------------------

void FaultDetector::checkCurrent(const ThreePhaseMeasurement& m,
                                  const GridNodeDescriptor& nd,
                                  std::vector<GridFaultEvent>& out) {
    float rated = nd.ratedCurrent_A;
    if (rated < 1e-6f) return;

    auto check = [&](float i, const std::string& phase) {
        float pct = (i / rated) * 100.0f;
        if (pct > mConfig.overcurrentAlarmPct) {
            auto ev = makeFault(nd.nodeId, nd.nodeTag,
                FaultType::OVERCURRENT, FaultSeverity::ALARM,
                "Overcurrent phase " + phase + ": " + std::to_string(i) + " A",
                i, rated * mConfig.overcurrentAlarmPct / 100.0f);
            raiseOrUpdate(ev, out);
        } else if (pct > mConfig.overcurrentWarnPct) {
            auto ev = makeFault(nd.nodeId, nd.nodeTag,
                FaultType::OVERCURRENT, FaultSeverity::WARNING,
                "High current phase " + phase + ": " + std::to_string(i) + " A",
                i, rated);
            raiseOrUpdate(ev, out);
        }
    };

    check(m.currentA_A, "A");
    check(m.currentB_A, "B");
    check(m.currentC_A, "C");
}

void FaultDetector::checkCurrentUnbalance(const ThreePhaseMeasurement& m,
                                           const GridNodeDescriptor& nd,
                                           std::vector<GridFaultEvent>& out) {
    float pct = currentUnbalancePct(m.currentA_A, m.currentB_A, m.currentC_A);
    if (pct > mConfig.currentUnbalancePct) {
        auto ev = makeFault(nd.nodeId, nd.nodeTag,
            FaultType::CURRENT_UNBALANCE, FaultSeverity::WARNING,
            "Current unbalance " + std::to_string(pct) + " %",
            pct, mConfig.currentUnbalancePct);
        raiseOrUpdate(ev, out);
    }
}

// ---------------------------------------------------------------------------
// Frequency checks
// ---------------------------------------------------------------------------

void FaultDetector::checkFrequency(const ThreePhaseMeasurement& m,
                                    std::vector<GridFaultEvent>& out) {
    float f = m.frequency_Hz;
    if (f < 1.0f) return;   // Not measured

    if (f > mConfig.freqOverAlarm_Hz) {
        auto ev = makeFault(m.nodeId, m.nodeTag,
            FaultType::OVER_FREQUENCY, FaultSeverity::ALARM,
            "Over-frequency: " + std::to_string(f) + " Hz",
            f, mConfig.freqOverAlarm_Hz);
        raiseOrUpdate(ev, out);
    } else if (f > mConfig.freqOverWarn_Hz) {
        auto ev = makeFault(m.nodeId, m.nodeTag,
            FaultType::OVER_FREQUENCY, FaultSeverity::WARNING,
            "High frequency: " + std::to_string(f) + " Hz",
            f, mConfig.freqOverWarn_Hz);
        raiseOrUpdate(ev, out);
    } else if (f < mConfig.freqUnderAlarm_Hz) {
        auto ev = makeFault(m.nodeId, m.nodeTag,
            FaultType::UNDER_FREQUENCY, FaultSeverity::ALARM,
            "Under-frequency: " + std::to_string(f) + " Hz",
            f, mConfig.freqUnderAlarm_Hz);
        raiseOrUpdate(ev, out);
    } else if (f < mConfig.freqUnderWarn_Hz) {
        auto ev = makeFault(m.nodeId, m.nodeTag,
            FaultType::UNDER_FREQUENCY, FaultSeverity::WARNING,
            "Low frequency: " + std::to_string(f) + " Hz",
            f, mConfig.freqUnderWarn_Hz);
        raiseOrUpdate(ev, out);
    }
}

// ---------------------------------------------------------------------------
// Power quality checks
// ---------------------------------------------------------------------------

void FaultDetector::checkPowerFactor(const ThreePhaseMeasurement& m,
                                      std::vector<GridFaultEvent>& out) {
    float pf = std::fabs(m.powerFactor);
    if (pf < 1e-6f) return;

    if (pf < mConfig.powerFactorMinAlarm) {
        auto ev = makeFault(m.nodeId, m.nodeTag,
            FaultType::LOW_POWER_FACTOR, FaultSeverity::ALARM,
            "Low power factor: " + std::to_string(pf),
            pf, mConfig.powerFactorMinAlarm);
        raiseOrUpdate(ev, out);
    } else if (pf < mConfig.powerFactorMinWarn) {
        auto ev = makeFault(m.nodeId, m.nodeTag,
            FaultType::LOW_POWER_FACTOR, FaultSeverity::WARNING,
            "Poor power factor: " + std::to_string(pf),
            pf, mConfig.powerFactorMinWarn);
        raiseOrUpdate(ev, out);
    }
}

// ---------------------------------------------------------------------------
// Thermal checks
// ---------------------------------------------------------------------------

void FaultDetector::checkTemperature(const ScalarMeasurement& s,
                                      std::vector<GridFaultEvent>& out) {
    float temp = s.value;
    if (temp > mConfig.transformerTempAlarm_C) {
        auto ev = makeFault(s.nodeId, s.nodeTag,
            FaultType::TRANSFORMER_OVERHEAT, FaultSeverity::ALARM,
            "Transformer overheating: " + std::to_string(temp) + " °C",
            temp, mConfig.transformerTempAlarm_C);
        raiseOrUpdate(ev, out);
    } else if (temp > mConfig.transformerTempWarn_C) {
        auto ev = makeFault(s.nodeId, s.nodeTag,
            FaultType::TRANSFORMER_OVERHEAT, FaultSeverity::WARNING,
            "Transformer high temperature: " + std::to_string(temp) + " °C",
            temp, mConfig.transformerTempWarn_C);
        raiseOrUpdate(ev, out);
    }
}

// ---------------------------------------------------------------------------
// Active fault access
// ---------------------------------------------------------------------------

std::vector<GridFaultEvent> FaultDetector::getActiveFaults() const {
    std::lock_guard<std::mutex> lock(mMutex);
    std::vector<GridFaultEvent> result;
    for (const auto& [key, ev] : mActive)
        if (!ev.cleared) result.push_back(ev);
    return result;
}

std::vector<GridFaultEvent> FaultDetector::getFaultHistory() const {
    std::lock_guard<std::mutex> lock(mMutex);
    return mHistory;
}

bool FaultDetector::clearFault(uint8_t nodeId, FaultType type) {
    std::lock_guard<std::mutex> lock(mMutex);
    auto key = faultKey(nodeId, type);
    auto it  = mActive.find(key);
    if (it == mActive.end()) return false;
    it->second.cleared = true;
    return true;
}

void FaultDetector::clearAllFaults(uint8_t nodeId) {
    std::lock_guard<std::mutex> lock(mMutex);
    for (auto& [key, ev] : mActive)
        if (ev.nodeId == nodeId) ev.cleared = true;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

GridFaultEvent FaultDetector::makeFault(uint8_t nodeId,
                                         const std::string& tag,
                                         FaultType type,
                                         FaultSeverity severity,
                                         const std::string& description,
                                         float measured,
                                         float threshold) const {
    GridFaultEvent ev;
    ev.timestamp     = tsNow();
    ev.nodeId        = nodeId;
    ev.nodeTag       = tag;
    ev.type          = type;
    ev.severity      = severity;
    ev.description   = description;
    ev.measuredValue = measured;
    ev.threshold     = threshold;
    ev.cleared       = false;
    return ev;
}

void FaultDetector::raiseOrUpdate(const GridFaultEvent& ev,
                                   std::vector<GridFaultEvent>& newEvents) {
    std::lock_guard<std::mutex> lock(mMutex);
    auto key = faultKey(ev.nodeId, ev.type);

    bool isNew = (mActive.find(key) == mActive.end()) ||
                  mActive[key].cleared;

    mActive[key] = ev;
    mHistory.push_back(ev);

    if (isNew) {
        newEvents.push_back(ev);
        for (auto& cb : mCallbacks) cb(ev);
    }
}

} // namespace Grid



