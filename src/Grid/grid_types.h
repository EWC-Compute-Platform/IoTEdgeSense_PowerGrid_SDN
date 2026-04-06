/**
 * @file grid_types.h
 * @brief Core power grid domain types
 *
 * All structures and enumerations shared across the grid sensing,
 * bridge, fault detection, and telemetry modules.
 *
 * Unit conventions used throughout:
 *   Voltage      → Volts (V)
 *   Current      → Amperes (A)
 *   Frequency    → Hertz (Hz)
 *   Power        → Watts (W) / VAR / VA
 *   Power factor → dimensionless [-1.0 .. 1.0]
 *   Temperature  → Celsius (°C)
 *   Angle        → degrees (°)
 *   Time         → milliseconds (uint64_t epoch)
 */

#ifndef GRID_TYPES_H
#define GRID_TYPES_H

#include <cstdint>
#include <string>
#include <vector>
#include <optional>

namespace Grid {

// ---------------------------------------------------------------------------
// Node classification
// ---------------------------------------------------------------------------

/**
 * @brief Classification of a grid node / measurement point
 */
enum class NodeType : uint8_t {
    SUBSTATION,         ///< High-voltage / medium-voltage substation
    DISTRIBUTION_BUS,   ///< Medium-voltage distribution bus
    FEEDER_LINE,        ///< Overhead or underground feeder segment
    TRANSFORMER,        ///< Power transformer (HV/MV or MV/LV)
    LOAD_POINT,         ///< Customer or industrial load connection
    GENERATION,         ///< Generation source (solar, wind, diesel)
    STORAGE,            ///< Battery or flywheel energy storage
    MEASUREMENT_ONLY    ///< Metering point with no switching capability
};

/**
 * @brief Voltage level category
 */
enum class VoltageLevel : uint8_t {
    EXTRA_HIGH   = 0,   ///< ≥ 220 kV
    HIGH         = 1,   ///< 36–220 kV
    MEDIUM       = 2,   ///< 1–35 kV
    LOW          = 3,   ///< < 1 kV (LV distribution)
    DC           = 4    ///< DC link
};

// ---------------------------------------------------------------------------
// Measurement types
// ---------------------------------------------------------------------------

/**
 * @brief Three-phase electrical measurement snapshot
 *
 * All three phases (A, B, C) plus calculated aggregates.
 * Optional fields are absent when the sensor does not measure that quantity.
 */
struct ThreePhaseMeasurement {
    uint64_t timestamp;         ///< Acquisition time (ms since epoch)
    uint8_t  nodeId;            ///< Originating node identifier
    std::string nodeTag;        ///< Human-readable node label

    // Per-phase RMS values
    float voltageA_V  = 0.0f;  ///< Phase A voltage (V RMS)
    float voltageB_V  = 0.0f;  ///< Phase B voltage (V RMS)
    float voltageC_V  = 0.0f;  ///< Phase C voltage (V RMS)

    float currentA_A  = 0.0f;  ///< Phase A current (A RMS)
    float currentB_A  = 0.0f;  ///< Phase B current (A RMS)
    float currentC_A  = 0.0f;  ///< Phase C current (A RMS)

    // Frequency (typically common to all phases)
    float frequency_Hz = 0.0f; ///< Fundamental frequency (Hz)

    // Power quantities
    float activePower_W    = 0.0f;  ///< Total 3-phase active power (W)
    float reactivePower_VAR= 0.0f;  ///< Total reactive power (VAR)
    float apparentPower_VA = 0.0f;  ///< Total apparent power (VA)
    float powerFactor      = 1.0f;  ///< Displacement power factor [-1..1]

    // Phase angles (degrees, relative to phase A as reference)
    float angleA_deg = 0.0f;
    float angleB_deg = 0.0f;
    float angleC_deg = 0.0f;

    // Quality flags
    bool valid          = false;
    bool phaseAPresent  = false;
    bool phaseBPresent  = false;
    bool phaseCPresent  = false;

    ThreePhaseMeasurement() : timestamp(0), nodeId(0) {}
};

/**
 * @brief Single-value scalar measurement (e.g. transformer temperature,
 *        battery state-of-charge, tap position)
 */
struct ScalarMeasurement {
    uint64_t    timestamp;
    uint8_t     nodeId;
    std::string nodeTag;
    std::string quantityName;   ///< e.g. "oil_temp", "soc", "tap_pos"
    float       value;
    std::string unit;
    bool        valid = false;

    ScalarMeasurement()
        : timestamp(0), nodeId(0), value(0.0f) {}
};

// ---------------------------------------------------------------------------
// Grid node descriptor
// ---------------------------------------------------------------------------

/**
 * @brief Static descriptor of a monitored grid node
 *
 * Written once at configuration time and used by the bridge to
 * contextualise raw sensor readings.
 */
struct GridNodeDescriptor {
    uint8_t     nodeId;
    std::string nodeTag;            ///< Unique human-readable tag
    NodeType    type;
    VoltageLevel voltageLevel;
    float       nominalVoltage_V;   ///< Nominal RMS line-to-neutral (V)
    float       nominalFrequency_Hz;///< Nominal frequency (Hz)
    float       ratedCurrent_A;     ///< Rated continuous current (A)

    // Operational thresholds (0 = use default from GridConfig)
    float voltageHighPct  = 0.0f;   ///< Over-voltage % above nominal
    float voltageLowPct   = 0.0f;   ///< Under-voltage % below nominal
    float freqHighHz      = 0.0f;   ///< Over-frequency limit
    float freqLowHz       = 0.0f;   ///< Under-frequency limit
    float currentMaxPct   = 0.0f;   ///< Overcurrent % of rated

    std::string location;           ///< Physical location label
    std::string sdnNodeId;          ///< Corresponding SDN topology node ID

    GridNodeDescriptor()
        : nodeId(0)
        , type(NodeType::MEASUREMENT_ONLY)
        , voltageLevel(VoltageLevel::LOW)
        , nominalVoltage_V(230.0f)
        , nominalFrequency_Hz(50.0f)
        , ratedCurrent_A(100.0f)
    {}
};

// ---------------------------------------------------------------------------
// Fault and event types
// ---------------------------------------------------------------------------

/**
 * @brief Classification of detected grid fault / anomaly
 */
enum class FaultType : uint16_t {
    // Voltage faults
    OVER_VOLTAGE        = 0x0100,
    UNDER_VOLTAGE       = 0x0101,
    VOLTAGE_UNBALANCE   = 0x0102,
    VOLTAGE_SAG         = 0x0103,
    VOLTAGE_SWELL       = 0x0104,
    VOLTAGE_INTERRUPT   = 0x0105,

    // Current / overcurrent
    OVERCURRENT         = 0x0200,
    EARTH_FAULT         = 0x0201,
    SHORT_CIRCUIT       = 0x0202,
    CURRENT_UNBALANCE   = 0x0203,

    // Frequency
    OVER_FREQUENCY      = 0x0300,
    UNDER_FREQUENCY     = 0x0301,
    FREQUENCY_INSTABLE  = 0x0302,

    // Power quality
    HIGH_HARMONICS      = 0x0400,
    LOW_POWER_FACTOR    = 0x0401,
    FLICKER             = 0x0402,

    // Thermal
    TRANSFORMER_OVERHEAT= 0x0500,
    CABLE_OVERHEAT      = 0x0501,

    // System
    COMMUNICATION_LOSS  = 0x0600,
    SENSOR_FAILURE      = 0x0601,

    NONE                = 0x0000
};

/**
 * @brief Severity level of a grid fault event
 */
enum class FaultSeverity : uint8_t {
    INFO        = 0,    ///< Advisory — within normal variation
    WARNING     = 1,    ///< Approaching limit — watchlist
    ALARM       = 2,    ///< Limit exceeded — action required
    CRITICAL    = 3     ///< Severe — immediate protective action
};

/**
 * @brief A detected grid fault event
 */
struct GridFaultEvent {
    uint64_t        timestamp;
    uint8_t         nodeId;
    std::string     nodeTag;
    FaultType       type;
    FaultSeverity   severity;
    std::string     description;
    float           measuredValue;  ///< The value that triggered the fault
    float           threshold;      ///< The threshold that was crossed
    bool            cleared = false;///< Set when the fault condition resolves

    GridFaultEvent()
        : timestamp(0), nodeId(0)
        , type(FaultType::NONE)
        , severity(FaultSeverity::INFO)
        , measuredValue(0.0f), threshold(0.0f)
    {}
};

} // namespace Grid

#endif // GRID_TYPES_H

