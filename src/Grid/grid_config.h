/**
 * @file grid_config.h
 * @brief Operational thresholds and default limits for grid monitoring
 *
 * Values follow IEC 60038 (voltage) and EN 50160 (power quality) standards
 * as defaults. Override per-node using GridNodeDescriptor fields.
 */

#ifndef GRID_CONFIG_H
#define GRID_CONFIG_H

#include <cstdint>

namespace Grid {

struct GridConfig {
    // ── Voltage thresholds (% of nominal) ────────────────────────────────
    float voltageOverPct         = 10.0f;  ///< +10 % → over-voltage ALARM
    float voltageHighPct         =  6.0f;  ///< +6  % → over-voltage WARNING
    float voltageLowPct          = 10.0f;  ///< -10 % → under-voltage WARNING
    float voltageUnderPct        = 15.0f;  ///< -15 % → under-voltage ALARM
    float voltageUnbalancePct    =  2.0f;  ///< >2  % unbalance → WARNING
    float voltageSagThresholdPct = 10.0f;  ///< Instantaneous sag detection

    // ── Frequency thresholds (Hz absolute) ───────────────────────────────
    float freqNominal_Hz         = 50.0f;  ///< System nominal (50 or 60 Hz)
    float freqOverAlarm_Hz       = 51.5f;  ///< Over-frequency ALARM
    float freqOverWarn_Hz        = 50.5f;  ///< Over-frequency WARNING
    float freqUnderWarn_Hz       = 49.5f;  ///< Under-frequency WARNING
    float freqUnderAlarm_Hz      = 48.5f;  ///< Under-frequency ALARM
    float freqInstableHz         =  0.5f;  ///< Rate-of-change threshold (Hz/s)

    // ── Current thresholds (% of rated) ──────────────────────────────────
    float overcurrentWarnPct     = 100.0f; ///< 100 % of rated → WARNING
    float overcurrentAlarmPct    = 120.0f; ///< 120 % of rated → ALARM
    float currentUnbalancePct    =  10.0f; ///< >10 % unbalance → WARNING

    // ── Power quality ─────────────────────────────────────────────────────
    float powerFactorMinWarn     =  0.90f; ///< PF < 0.90 → WARNING
    float powerFactorMinAlarm    =  0.85f; ///< PF < 0.85 → ALARM
    float harmonicTHD_pct        =  8.0f;  ///< Total harmonic distortion %

    // ── Thermal ───────────────────────────────────────────────────────────
    float transformerTempWarn_C  = 80.0f;  ///< °C WARNING
    float transformerTempAlarm_C = 98.0f;  ///< °C ALARM

    // ── Timing ────────────────────────────────────────────────────────────
    uint32_t faultConfirmMs      = 200;    ///< Sustain ms before raising fault
    uint32_t faultClearMs        = 1000;   ///< Clear delay ms after recovery
    uint32_t heartbeatIntervalMs = 30000;  ///< Telemetry heartbeat period (ms)

    // ── Serialisation ────────────────────────────────────────────────────
    /**
     * @brief Return a default 50 Hz European grid configuration
     */
    static GridConfig european50Hz() {
        GridConfig c;
        c.freqNominal_Hz = 50.0f;
        return c;
    }

    /**
     * @brief Return a default 60 Hz North American grid configuration
     */
    static GridConfig northAmerican60Hz() {
        GridConfig c;
        c.freqNominal_Hz    = 60.0f;
        c.freqOverAlarm_Hz  = 61.5f;
        c.freqOverWarn_Hz   = 60.5f;
        c.freqUnderWarn_Hz  = 59.5f;
        c.freqUnderAlarm_Hz = 58.5f;
        return c;
    }
};

} // namespace Grid

#endif // GRID_CONFIG_H

