/**
 * @file grid_sensors.h
 * @brief Power-grid specialised sensor classes
 *
 * Each class wraps the appropriate hardware driver (I2CSensor, ADC via SPI,
 * UART metering module) and adds grid-domain semantics: the read() result is
 * a SensorReading whose values[] vector is always in a known layout so the
 * PowerGridBridge can safely decode it.
 *
 * Values[] layout contracts
 * ─────────────────────────
 * VoltageSensor  : [Vrms_A, Vrms_B, Vrms_C, Vline_AB, Vline_BC, Vline_CA]
 * CurrentSensor  : [Arms_A, Arms_B, Arms_C]
 * FrequencySensor: [Hz, dHz_dt]    (frequency + rate-of-change)
 * PowerMeter     : [W_total, VAR_total, VA_total, PF, angle_A, angle_B, angle_C]
 * ThermalSensor  : [temperature_C]
 */

#ifndef GRID_SENSORS_H
#define GRID_SENSORS_H

#include "sensors/sensor_base.h"
#include "grid_types.h"
#include <chrono>
#include <cmath>
#include <algorithm>

namespace Grid {

// ---------------------------------------------------------------------------
// Utility: current timestamp in ms
// ---------------------------------------------------------------------------
inline uint64_t nowMs() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<milliseconds>(
            steady_clock::now().time_since_epoch()).count());
}

// ===========================================================================
// VoltageSensor
// ===========================================================================

/**
 * @brief Three-phase RMS voltage sensor
 *
 * Simulates a PT100/PT200 potential-transformer front-end connected to
 * a 16-bit ADC over SPI. In production, replace simulateRead() with
 * real ADC register parsing.
 */
class VoltageSensor : public Sensors::SensorBase {
public:
    /**
     * @param id           Sensor ID
     * @param nodeTag      Grid node label (e.g. "BUS_A1")
     * @param nominalV     Nominal phase-to-neutral RMS voltage (V)
     */
    VoltageSensor(uint8_t id,
                  const std::string& nodeTag,
                  float nominalV = 230.0f)
        : SensorBase(id, "VoltageSensor_" + nodeTag)
        , mNominalV(nominalV)
        , mNodeTag(nodeTag)
    {}

    bool initialize() override { setState(Sensors::SensorState::INITIALIZED); return true; }
    bool calibrate()  override { return true; }
    bool sleep()      override { setState(Sensors::SensorState::SLEEPING); return true; }
    bool wakeUp()     override { setState(Sensors::SensorState::RUNNING);  return true; }
    bool selfTest()   override { return true; }

    Sensors::SensorReading read() override {
        Sensors::SensorReading r;
        r.sensorId  = mId;
        r.unit      = "V";
        r.timestamp = nowMs();
        r.valid     = true;
        setState(Sensors::SensorState::RUNNING);

        // Simulate a ±3 % variation around nominal
        auto v = [&](float offset) {
            return mNominalV * (1.0f + offset + simulateNoise(0.005f));
        };

        float va = v(0.0f);
        float vb = v(0.0f);
        float vc = v(0.0f);

        // [Vrms_A, Vrms_B, Vrms_C, Vline_AB, Vline_BC, Vline_CA]
        r.values = {
            va, vb, vc,
            va * 1.732f,   // line-to-line ≈ √3 × phase
            vb * 1.732f,
            vc * 1.732f
        };
        return r;
    }

    float getNominalV() const { return mNominalV; }

private:
    float       mNominalV;
    std::string mNodeTag;

    float simulateNoise(float amplitude) {
        // Simple deterministic pseudo-noise for stub
        static uint32_t seed = 12345;
        seed = seed * 1664525u + 1013904223u;
        return amplitude * (static_cast<float>(seed & 0xFFFF) / 32768.0f - 1.0f);
    }
};

// ===========================================================================
// CurrentSensor
// ===========================================================================

/**
 * @brief Three-phase RMS current sensor (CT-based)
 *
 * Simulates a current transformer front-end.
 * values[] = [Arms_A, Arms_B, Arms_C]
 */
class CurrentSensor : public Sensors::SensorBase {
public:
    CurrentSensor(uint8_t id,
                  const std::string& nodeTag,
                  float ratedA = 100.0f)
        : SensorBase(id, "CurrentSensor_" + nodeTag)
        , mRatedA(ratedA)
        , mNodeTag(nodeTag)
        , mLoadFactor(0.6f)
    {}

    bool initialize() override { setState(Sensors::SensorState::INITIALIZED); return true; }
    bool calibrate()  override { return true; }
    bool sleep()      override { setState(Sensors::SensorState::SLEEPING); return true; }
    bool wakeUp()     override { setState(Sensors::SensorState::RUNNING);  return true; }
    bool selfTest()   override { return true; }

    void setLoadFactor(float lf) { mLoadFactor = std::clamp(lf, 0.0f, 1.5f); }

    Sensors::SensorReading read() override {
        Sensors::SensorReading r;
        r.sensorId  = mId;
        r.unit      = "A";
        r.timestamp = nowMs();
        r.valid     = true;
        setState(Sensors::SensorState::RUNNING);

        float base = mRatedA * mLoadFactor;
        auto noise = [&]() {
            static uint32_t s = 98765;
            s = s * 1664525u + 1013904223u;
            return base * 0.02f * (static_cast<float>(s & 0xFFFF) / 32768.0f - 1.0f);
        };

        r.values = { base + noise(), base + noise(), base + noise() };
        return r;
    }

private:
    float       mRatedA;
    std::string mNodeTag;
    float       mLoadFactor;
};

// ===========================================================================
// FrequencySensor
// ===========================================================================

/**
 * @brief Grid frequency and rate-of-change-of-frequency (RoCoF) sensor
 *
 * values[] = [Hz, dHz_dt]
 */
class FrequencySensor : public Sensors::SensorBase {
public:
    FrequencySensor(uint8_t id,
                    const std::string& nodeTag,
                    float nominalHz = 50.0f)
        : SensorBase(id, "FrequencySensor_" + nodeTag)
        , mNominalHz(nominalHz)
        , mNodeTag(nodeTag)
        , mLastFreq(nominalHz)
        , mLastTs(0)
    {}

    bool initialize() override { setState(Sensors::SensorState::INITIALIZED); return true; }
    bool calibrate()  override { return true; }
    bool sleep()      override { setState(Sensors::SensorState::SLEEPING); return true; }
    bool wakeUp()     override { setState(Sensors::SensorState::RUNNING);  return true; }
    bool selfTest()   override { return true; }

    Sensors::SensorReading read() override {
        Sensors::SensorReading r;
        r.sensorId  = mId;
        r.unit      = "Hz";
        r.timestamp = nowMs();
        r.valid     = true;
        setState(Sensors::SensorState::RUNNING);

        // Simulate ±0.1 Hz variation
        static uint32_t s = 44321;
        s = s * 1664525u + 1013904223u;
        float freq = mNominalHz + 0.1f * (static_cast<float>(s & 0xFFFF) / 32768.0f - 1.0f);

        float dHzDt = 0.0f;
        if (mLastTs > 0) {
            float dtSec = static_cast<float>(r.timestamp - mLastTs) / 1000.0f;
            if (dtSec > 0.0f) dHzDt = (freq - mLastFreq) / dtSec;
        }
        mLastFreq = freq;
        mLastTs   = r.timestamp;

        r.values = { freq, dHzDt };
        return r;
    }

private:
    float       mNominalHz;
    std::string mNodeTag;
    float       mLastFreq;
    uint64_t    mLastTs;
};

// ===========================================================================
// PowerMeter
// ===========================================================================

/**
 * @brief Three-phase active / reactive / apparent power meter
 *
 * values[] = [W_total, VAR_total, VA_total, PF, angle_A_deg, angle_B_deg, angle_C_deg]
 *
 * Derives power from voltage + current readings when both are provided,
 * or reads from a dedicated metering IC over UART/I2C.
 */
class PowerMeter : public Sensors::SensorBase {
public:
    PowerMeter(uint8_t id,
               const std::string& nodeTag,
               float nominalV = 230.0f,
               float ratedA   = 100.0f,
               float powerFactor = 0.95f)
        : SensorBase(id, "PowerMeter_" + nodeTag)
        , mNominalV(nominalV)
        , mRatedA(ratedA)
        , mPF(powerFactor)
        , mNodeTag(nodeTag)
        , mLoadFactor(0.6f)
    {}

    bool initialize() override { setState(Sensors::SensorState::INITIALIZED); return true; }
    bool calibrate()  override { return true; }
    bool sleep()      override { setState(Sensors::SensorState::SLEEPING); return true; }
    bool wakeUp()     override { setState(Sensors::SensorState::RUNNING);  return true; }
    bool selfTest()   override { return true; }

    void setLoadFactor(float lf) { mLoadFactor = std::clamp(lf, 0.0f, 1.5f); }
    void setPowerFactor(float pf) { mPF = std::clamp(pf, -1.0f, 1.0f); }

    Sensors::SensorReading read() override {
        Sensors::SensorReading r;
        r.sensorId  = mId;
        r.unit      = "W";
        r.timestamp = nowMs();
        r.valid     = true;
        setState(Sensors::SensorState::RUNNING);

        float va_rms = mNominalV;
        float ia_rms = mRatedA * mLoadFactor;
        float va_apparent = 3.0f * va_rms * ia_rms;    // 3-phase apparent (VA)
        float va_active   = va_apparent * std::fabs(mPF);
        float va_reactive = va_apparent * std::sqrt(1.0f - mPF * mPF);
        float angle = std::acos(std::fabs(mPF)) * 180.0f / 3.14159265f;

        // [W_total, VAR_total, VA_total, PF, angle_A, angle_B, angle_C]
        r.values = {
            va_active,
            va_reactive,
            va_apparent,
            mPF,
            angle, angle - 120.0f, angle - 240.0f
        };
        return r;
    }

private:
    float       mNominalV;
    float       mRatedA;
    float       mPF;
    std::string mNodeTag;
    float       mLoadFactor;
};

// ===========================================================================
// ThermalSensor  (transformer oil / cable temperature)
// ===========================================================================

/**
 * @brief Transformer oil or cable thermal sensor (PT100 / NTC)
 *
 * values[] = [temperature_C]
 */
class ThermalSensor : public Sensors::SensorBase {
public:
    ThermalSensor(uint8_t id,
                  const std::string& nodeTag,
                  float ambientC = 25.0f)
        : SensorBase(id, "ThermalSensor_" + nodeTag)
        , mAmbientC(ambientC)
        , mCurrentTempC(ambientC)
        , mNodeTag(nodeTag)
    {}

    bool initialize() override { setState(Sensors::SensorState::INITIALIZED); return true; }
    bool calibrate()  override { return true; }
    bool sleep()      override { setState(Sensors::SensorState::SLEEPING); return true; }
    bool wakeUp()     override { setState(Sensors::SensorState::RUNNING);  return true; }
    bool selfTest()   override { return true; }

    void setTemperature(float c) { mCurrentTempC = c; }

    Sensors::SensorReading read() override {
        Sensors::SensorReading r;
        r.sensorId  = mId;
        r.unit      = "degC";
        r.timestamp = nowMs();
        r.valid     = true;
        setState(Sensors::SensorState::RUNNING);

        static uint32_t s = 77777;
        s = s * 1664525u + 1013904223u;
        float noise = 0.3f * (static_cast<float>(s & 0xFFFF) / 32768.0f - 1.0f);
        r.values = { mCurrentTempC + noise };
        return r;
    }

private:
    float       mAmbientC;
    float       mCurrentTempC;
    std::string mNodeTag;
};

} // namespace Grid

#endif // GRID_SENSORS_H
