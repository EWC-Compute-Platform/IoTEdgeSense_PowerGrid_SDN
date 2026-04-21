/**
 * @file feature_extractor.cpp
 * @brief Feature extractor implementation
 *
 * All formulas verified against:
 *   - IEC 60076-7:2018 (transformer thermal model)
 *   - ml/features/transformer_features.py (Python reference)
 *   - ml/models/anomaly_detector.py (Python reference)
 */

#include "feature_extractor.h"
#include <ctime>
#include <sstream>
#include <cassert>

namespace ML {

// ===========================================================================
// ThermalFeatureExtractor
// ===========================================================================

ThermalFeatureExtractor::ThermalFeatureExtractor(const ThermalFeatureConfig& cfg)
    : mCfg(cfg)
{}

// ---------------------------------------------------------------------------
// Main extraction entry point
// ---------------------------------------------------------------------------

std::vector<float>
ThermalFeatureExtractor::extract(ThermalState&           state,
                                  const MeasurementInput& meas,
                                  float                   dtMinutes) const {
    // ── Compute load factor ────────────────────────────────────────────
    float Iavg = (meas.ia + meas.ib + meas.ic) / 3.0f;
    float K    = loadFactor(Iavg);

    // ── Advance IEC 60076-7 thermal model ──────────────────────────────
    float ambientC = meas.ambientC;
    thermalStep(state, K, ambientC, dtMinutes);

    // If top-oil temperature is directly measured, override computed value
    // (more conservative — keeps computed hot-spot from model)
    if (meas.topOilC > 1.0f) {
        float dTh = mCfg.deltaTheta_h_r * std::pow(K, 2.0f * mCfg.m);
        state.thetaToC = meas.topOilC;
        state.thetaHC  = meas.topOilC + dTh;
        // Recompute V with corrected hot-spot
        state.V = relativeAgeingRate(state.thetaHC);
    }

    // Push to rolling buffers
    state.push(K, state.thetaHC, state.V);

    // ── Assemble feature vector ────────────────────────────────────────
    std::vector<float> f;
    f.reserve(N_THERMAL_FEATURES);

    // ── [0–14] Physics features (15) ──────────────────────────────────
    f.push_back(K);                                             // K
    f.push_back(K * K);                                         // K_sq
    f.push_back(ambientC);                                      // theta_ambient_c
    f.push_back(state.thetaToC);                                // theta_to_c
    f.push_back(state.thetaHC);                                 // theta_h_c
    f.push_back(state.thetaToC - ambientC);                     // delta_theta_to_c
    f.push_back(state.thetaHC  - state.thetaToC);               // delta_theta_h_c
    f.push_back(state.V);                                       // V
    f.push_back(state.cumulativeAging);                         // cumulative_ageing_h
    f.push_back(mCfg.hotspotContinuousC - state.thetaHC);      // thermal_margin_c
    f.push_back(state.thetaHC / mCfg.hotspotLimitC * 100.0f);  // hotspot_to_limit_pct

    // Load trend: sign of change vs 30 samples ago
    float kTrend = 0.0f;
    if (state.kBuf.size() >= 30) {
        kTrend = std::copysign(
            1.0f,
            K - state.kBuf[state.kBuf.size() - 30]
        );
    }
    f.push_back(kTrend);                                        // load_trend
    f.push_back(state.V);                                       // ageing_acceleration
    f.push_back(state.thetaHC > mCfg.hotspotContinuousC ? 1.0f : 0.0f); // overtemp_flag
    f.push_back(K > 1.0f ? 1.0f : 0.0f);                       // overload_flag

    // ── [15–38] Rolling window features (6 stats × 4 windows = 24) ───
    for (size_t wSamples : THERMAL_WINDOWS) {
        f.push_back(wMean(state.kBuf,  wSamples));              // K_mean
        f.push_back(wMax (state.kBuf,  wSamples));              // K_max
        f.push_back(wStd (state.kBuf,  wSamples));              // K_std
        f.push_back(wMean(state.thBuf, wSamples));              // theta_h_mean
        f.push_back(wMax (state.thBuf, wSamples));              // theta_h_max
        f.push_back(wMean(state.vBuf,  wSamples));              // V_mean
    }

    // ── [39–42] Temporal features (4) ─────────────────────────────────
    int hour = 12, wday = 0, month = 6;
    if (meas.timestampMs > 0)
        wallTime(meas.timestampMs, hour, wday, month);

    f.push_back(static_cast<float>(hour));                      // hour_of_day
    f.push_back(static_cast<float>(wday));                      // day_of_week
    f.push_back(static_cast<float>(month));                     // month
    f.push_back((hour >= 7 && hour <= 22) ? 1.0f : 0.0f);      // is_peak_hours

    assert(f.size() == N_THERMAL_FEATURES);
    return f;
}

// ---------------------------------------------------------------------------
// IEC 60076-7 Euler thermal step
// ---------------------------------------------------------------------------

void ThermalFeatureExtractor::thermalStep(ThermalState& state,
                                           float K,
                                           float ambientC,
                                           float dtMinutes) const {
    // Steady-state targets (IEC 60076-7 equations)
    float dTto_ult = mCfg.deltaTheta_to_r
                   * std::pow((K * K * mCfg.R + 1.0f) / (mCfg.R + 1.0f), mCfg.n);
    float dTh_ult  = mCfg.deltaTheta_h_r * std::pow(K, 2.0f * mCfg.m);

    float thetaTo_ult = ambientC + dTto_ult;
    float thetaH_ult  = thetaTo_ult + dTh_ult;

    // Exponential approach to steady state
    float alphaTo = 1.0f - std::exp(-dtMinutes / mCfg.tauToMin);
    float alphaH  = 1.0f - std::exp(-dtMinutes / mCfg.tauHMin);

    state.thetaToC += alphaTo * (thetaTo_ult - state.thetaToC);
    state.thetaHC  += alphaH  * (thetaH_ult  - state.thetaHC);
    state.K         = K;

    // Arrhenius relative ageing rate
    state.V = relativeAgeingRate(state.thetaHC);

    // Accumulate ageing (hours)
    state.cumulativeAging += state.V * (dtMinutes / 60.0f);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

float ThermalFeatureExtractor::loadFactor(float currentA) const {
    float K = currentA / mCfg.ratedCurrentA;
    return std::clamp(K, 0.0f, 2.0f);
}

float ThermalFeatureExtractor::relativeAgeingRate(float thetaHC) {
    float thetaH_K = 273.0f + thetaHC;
    return std::exp(ARRHENIUS_EA_OVER_KB * (1.0f / THETA_REF_K - 1.0f / thetaH_K));
}

float ThermalFeatureExtractor::wMean(const std::deque<float>& buf, size_t n) {
    if (buf.empty()) return 0.0f;
    size_t cnt = std::min(n, buf.size());
    auto it    = buf.end() - static_cast<std::ptrdiff_t>(cnt);
    return std::accumulate(it, buf.end(), 0.0f) / static_cast<float>(cnt);
}

float ThermalFeatureExtractor::wMax(const std::deque<float>& buf, size_t n) {
    if (buf.empty()) return 0.0f;
    size_t cnt = std::min(n, buf.size());
    auto it    = buf.end() - static_cast<std::ptrdiff_t>(cnt);
    return *std::max_element(it, buf.end());
}

float ThermalFeatureExtractor::wStd(const std::deque<float>& buf, size_t n) {
    if (buf.size() < 2) return 0.0f;
    size_t cnt  = std::min(n, buf.size());
    auto it     = buf.end() - static_cast<std::ptrdiff_t>(cnt);
    float mean  = std::accumulate(it, buf.end(), 0.0f) / static_cast<float>(cnt);
    float var   = 0.0f;
    for (auto jt = it; jt != buf.end(); ++jt)
        var += (*jt - mean) * (*jt - mean);
    return std::sqrt(var / static_cast<float>(cnt));
}

void ThermalFeatureExtractor::wallTime(uint64_t tsMs,
                                        int& hour, int& wday, int& month) {
    std::time_t tt = static_cast<std::time_t>(tsMs / 1000);
    std::tm* lt    = std::localtime(&tt);
    if (lt) {
        hour  = lt->tm_hour;
        wday  = lt->tm_wday;
        month = lt->tm_mon + 1;
    }
}

std::vector<std::string> ThermalFeatureExtractor::featureNames() {
    std::vector<std::string> names = {
        // Physics (15)
        "K", "K_sq",
        "theta_ambient_c", "theta_to_c", "theta_h_c",
        "delta_theta_to_c", "delta_theta_h_c",
        "V", "cumulative_ageing_h",
        "thermal_margin_c", "hotspot_to_limit_pct",
        "load_trend", "ageing_acceleration",
        "overtemp_flag", "overload_flag",
    };
    // Rolling (24 = 6 stats × 4 windows)
    const std::array<std::string, 4> labels = {"30min", "1h", "6h", "24h"};
    for (const auto& label : labels) {
        names.push_back("K_mean_"       + label);
        names.push_back("K_max_"        + label);
        names.push_back("K_std_"        + label);
        names.push_back("theta_h_mean_" + label);
        names.push_back("theta_h_max_"  + label);
        names.push_back("V_mean_"       + label);
    }
    // Temporal (4)
    names.push_back("hour_of_day");
    names.push_back("day_of_week");
    names.push_back("month");
    names.push_back("is_peak_hours");
    return names;
}


// ===========================================================================
// AnomalyFeatureExtractor
// ===========================================================================

AnomalyFeatureExtractor::AnomalyFeatureExtractor(const AnomalyFeatureConfig& cfg)
    : mCfg(cfg)
{}

std::vector<float>
AnomalyFeatureExtractor::extract(AnomalyState&          state,
                                  const MeasurementInput& meas) const {
    // Per-unit conversions
    float va_pu = meas.va / mCfg.ratedVoltageV;
    float vb_pu = meas.vb / mCfg.ratedVoltageV;
    float vc_pu = meas.vc / mCfg.ratedVoltageV;
    float ia_pu = meas.ia / mCfg.ratedCurrentA;
    float ib_pu = meas.ib / mCfg.ratedCurrentA;
    float ic_pu = meas.ic / mCfg.ratedCurrentA;

    float p_pu  = meas.powerW      / mCfg.ratedPowerW;
    float q_pu  = meas.reactiveVar / mCfg.ratedPowerW;
    float s_pu  = meas.apparentVa  / mCfg.ratedPowerW;
    float pf    = std::clamp(meas.powerFactor, -1.0f, 1.0f);

    // Derived voltage quantities
    float v_avg       = (va_pu + vb_pu + vc_pu) / 3.0f;
    float v_unbalance = unbalancePct(va_pu, vb_pu, vc_pu);
    float v_deviation = std::abs(v_avg - 1.0f);
    float v_rocov     = va_pu - state.lastVaPu;       // rate of change per sample
    state.lastVaPu    = va_pu;

    // Derived current quantities
    float i_unbalance = unbalancePct(ia_pu, ib_pu, ic_pu);
    float i_neutral   = std::abs(ia_pu + ib_pu + ic_pu);

    // Frequency quantities
    float freq_dev    = meas.freqHz - mCfg.nominalFreqHz;
    float rocof       = meas.rocoFHz;
    state.pushFreq(meas.freqHz);
    float freq_stability = 0.0f;
    if (state.freqBuf.size() >= 2) {
        freq_stability = wStd(state.freqBuf, state.freqBuf.size());
    }

    // Power quality
    float pf_deviation = std::abs(pf - mCfg.nominalPF);

    // Update rolling buffers
    state.pushVoltage(va_pu);
    float i_avg_pu = (std::abs(ia_pu) + std::abs(ib_pu) + std::abs(ic_pu)) / 3.0f;
    state.pushCurrent(i_avg_pu);

    // Rolling window statistics (5-min = 300 samples, 30-min = 1800 samples at 1s)
    size_t n5  = std::min(state.vaBuf.size(),   static_cast<size_t>(300));
    size_t n30 = std::min(state.vaBuf.size(),   static_cast<size_t>(1800));

    // Interaction features
    float vi_coherence   = std::abs(pf);
    float load_asym      = v_unbalance * i_unbalance / 100.0f;
    float pf_x_unbalance = pf_deviation * v_unbalance;

    // ── Assemble feature vector ────────────────────────────────────────
    std::vector<float> f;
    f.reserve(N_ANOMALY_FEATURES);

    // Voltage (6)
    f.push_back(va_pu);           // Va_pu
    f.push_back(vb_pu);           // Vb_pu
    f.push_back(vc_pu);           // Vc_pu
    f.push_back(v_unbalance);     // V_unbalance_pct
    f.push_back(v_deviation);     // V_deviation_from_nominal
    f.push_back(v_rocov);         // V_rocov

    // Current (5)
    f.push_back(ia_pu);           // Ia_pu
    f.push_back(ib_pu);           // Ib_pu
    f.push_back(ic_pu);           // Ic_pu
    f.push_back(i_unbalance);     // I_unbalance_pct
    f.push_back(i_neutral);       // I_neutral_pu

    // Frequency (3)
    f.push_back(freq_dev);        // freq_deviation_hz
    f.push_back(rocof);           // rocof_hz_per_s
    f.push_back(freq_stability);  // freq_stability

    // Power quality (5)
    f.push_back(pf);              // power_factor
    f.push_back(p_pu);            // active_power_pu
    f.push_back(q_pu);            // reactive_power_pu
    f.push_back(s_pu);            // apparent_power_pu
    f.push_back(pf_deviation);    // pf_deviation

    // Rolling 5-min (4)
    f.push_back(wMean(state.vaBuf,    n5));  // Va_mean_5min
    f.push_back(wStd (state.vaBuf,    n5));  // Va_std_5min
    f.push_back(wMax (state.iAvgBuf,  n5));  // I_max_5min
    f.push_back(wStd (state.iAvgBuf,  n5));  // I_std_5min

    // Rolling 30-min (4)
    f.push_back(wMean(state.vaBuf,    n30)); // Va_mean_30min
    f.push_back(wStd (state.vaBuf,    n30)); // Va_std_30min
    f.push_back(wMax (state.iAvgBuf,  n30)); // I_max_30min
    f.push_back(wStd (state.iAvgBuf,  n30)); // I_std_30min

    // Interaction (3)
    f.push_back(vi_coherence);    // VI_phase_coherence
    f.push_back(load_asym);       // load_asymmetry
    f.push_back(pf_x_unbalance);  // power_factor_x_unbalance

    assert(f.size() == N_ANOMALY_FEATURES);
    return f;
}

float AnomalyFeatureExtractor::unbalancePct(float a, float b, float c) {
    float avg = (std::abs(a) + std::abs(b) + std::abs(c)) / 3.0f;
    if (avg < 1e-9f) return 0.0f;
    float dev = std::max({std::abs(std::abs(a) - avg),
                           std::abs(std::abs(b) - avg),
                           std::abs(std::abs(c) - avg)});
    return (dev / avg) * 100.0f;
}

float AnomalyFeatureExtractor::wMean(const std::deque<float>& buf, size_t n) {
    if (buf.empty() || n == 0) return 0.0f;
    size_t cnt = std::min(n, buf.size());
    auto it    = buf.end() - static_cast<std::ptrdiff_t>(cnt);
    return std::accumulate(it, buf.end(), 0.0f) / static_cast<float>(cnt);
}

float AnomalyFeatureExtractor::wMax(const std::deque<float>& buf, size_t n) {
    if (buf.empty() || n == 0) return 0.0f;
    size_t cnt = std::min(n, buf.size());
    auto it    = buf.end() - static_cast<std::ptrdiff_t>(cnt);
    return *std::max_element(it, buf.end());
}

float AnomalyFeatureExtractor::wStd(const std::deque<float>& buf, size_t n) {
    if (buf.size() < 2 || n < 2) return 0.0f;
    size_t cnt  = std::min(n, buf.size());
    auto it     = buf.end() - static_cast<std::ptrdiff_t>(cnt);
    float mean  = std::accumulate(it, buf.end(), 0.0f) / static_cast<float>(cnt);
    float var   = 0.0f;
    for (auto jt = it; jt != buf.end(); ++jt)
        var += (*jt - mean) * (*jt - mean);
    return std::sqrt(var / static_cast<float>(cnt));
}

std::vector<std::string> AnomalyFeatureExtractor::featureNames() {
    return {
        "Va_pu", "Vb_pu", "Vc_pu",
        "V_unbalance_pct", "V_deviation_from_nominal", "V_rocov",
        "Ia_pu", "Ib_pu", "Ic_pu",
        "I_unbalance_pct", "I_neutral_pu",
        "freq_deviation_hz", "rocof_hz_per_s", "freq_stability",
        "power_factor", "active_power_pu", "reactive_power_pu",
        "apparent_power_pu", "pf_deviation",
        "Va_mean_5min", "Va_std_5min", "I_max_5min", "I_std_5min",
        "Va_mean_30min", "Va_std_30min", "I_max_30min", "I_std_30min",
        "VI_phase_coherence", "load_asymmetry", "power_factor_x_unbalance",
    };
}


// ===========================================================================
// FeatureExtractorFactory
// ===========================================================================

FeatureExtractorFactory::FeatureExtractorFactory(
    const ThermalFeatureConfig& thermalCfg,
    const AnomalyFeatureConfig& anomalyCfg)
    : mThermalExt(thermalCfg)
    , mAnomalyExt(anomalyCfg)
{}

void FeatureExtractorFactory::extractAll(
    const MeasurementInput& meas,
    float                   dtMinutes,
    std::vector<float>&     thermal,
    std::vector<float>&     anomaly)
{
    thermal = mThermalExt.extract(mThermalState, meas, dtMinutes);
    anomaly = mAnomalyExt.extract(mAnomalyState, meas);
}

void FeatureExtractorFactory::reset() {
    mThermalState = ThermalState{};
    mAnomalyState = AnomalyState{};
}

} // namespace ML

