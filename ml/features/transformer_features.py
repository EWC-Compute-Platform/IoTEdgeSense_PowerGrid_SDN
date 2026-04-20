"""
transformer_features.py
=======================
Physics-informed feature extraction for power transformer
predictive maintenance, grounded in IEC 60076-7:2018
"Loading guide for oil-immersed power transformers".

IEC 60076-7 Thermal Model
--------------------------
The standard defines a two-layer thermal model:

  Layer 1 — Top-oil temperature rise:
    Δθ_TO = Δθ_TO,R × [(K²R + 1) / (R + 1)]^n

  Layer 2 — Hot-spot temperature rise over top-oil:
    Δθ_H  = Δθ_H,R × K^(2m)

  Hot-spot temperature:
    θ_H = θ_a + Δθ_TO + Δθ_H

  Relative ageing rate (Arrhenius equation):
    V = exp[ (E_a / k_B) × (1/383 - 1/(273 + θ_H)) ]
      = exp[ 15000 × (1/383 - 1/(273 + θ_H)) ]
      (reference temperature: 110°C = 383 K)

  Loss of life (per time step dt):
    L += V × dt

Typical IEC 60076-7 default parameters (ONAN cooling):
  Δθ_TO,R = 55 K  (rated top-oil rise over ambient)
  Δθ_H,R  = 23 K  (rated hot-spot rise over top-oil)
  R       = 6     (ratio load losses / no-load losses)
  n       = 0.8   (top-oil thermal exponent)
  m       = 1.0   (winding exponent)
  τ_TO    = 210 min (top-oil thermal time constant)
  τ_H     = 7 min   (hot-spot thermal time constant)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# IEC 60076-7 Transformer Rating Dataclass
# ---------------------------------------------------------------------------

@dataclass
class TransformerRating:
    """
    Nameplate and thermal parameters for a specific transformer.
    Defaults are IEC 60076-7 typical values for ONAN cooling (Table 3).
    """
    # Identity
    transformer_id: str = "TX_DEFAULT"
    rated_power_kva: float = 1000.0
    rated_voltage_hv_kv: float = 11.0
    rated_voltage_lv_kv: float = 0.415

    # Cooling mode: ONAN, ONAF, OFAF, ODAF
    cooling_mode: str = "ONAN"

    # IEC 60076-7 thermal parameters
    delta_theta_to_r: float = 55.0   # K — rated top-oil temperature rise
    delta_theta_h_r: float  = 23.0   # K — rated hot-spot rise over top-oil
    R: float = 6.0                   # ratio load losses / no-load losses
    n: float = 0.8                   # top-oil thermal exponent
    m: float = 1.0                   # winding exponent

    # Thermal time constants (minutes)
    tau_to_min: float = 210.0        # top-oil time constant
    tau_h_min: float  = 7.0          # hot-spot time constant

    # Reference temperatures
    theta_ref_k: float = 383.0       # 110°C reference in Kelvin (IEC 60076-7 eq.)
    arrhenius_ea_over_kb: float = 15000.0  # E_a/k_B constant for normal paper

    # Design limits
    max_load_factor: float = 1.5     # IEC guideline maximum K for short-term
    oil_temp_limit_c: float = 105.0  # maximum top-oil temperature
    hotspot_limit_c: float = 140.0   # maximum hot-spot (normal paper)
    hotspot_continuous_c: float = 98.0  # continuous hot-spot design limit

    # Rated load losses / no-load losses ratio (for K calculation)
    rated_current_a: float = 1000.0  # rated HV side current (A)


# ---------------------------------------------------------------------------
# IEC 60076-7 Thermal State Calculator
# ---------------------------------------------------------------------------

class IEC60076_7_ThermalModel:
    """
    Implements the IEC 60076-7 differential thermal model.
    
    Maintains internal state (θ_TO, θ_H) and integrates forward in time
    using a simple Euler step, suitable for real-time edge computation.
    
    For batch processing of historical data, use
    TransformerFeatureExtractor.compute_from_timeseries().
    """

    def __init__(self, rating: TransformerRating, ambient_c: float = 20.0):
        self.r = rating
        # Initialise thermal state at ambient + rated rise (steady state at K=1)
        self.theta_ambient_c = ambient_c
        self.theta_to_c = ambient_c + rating.delta_theta_to_r   # top-oil
        self.theta_h_c  = (ambient_c
                           + rating.delta_theta_to_r
                           + rating.delta_theta_h_r)             # hot-spot
        self._cumulative_ageing = 0.0   # dimensionless (hours equivalent)
        self._elapsed_hours = 0.0

    # ── Steady-state calculations (no dynamics) ──────────────────────────

    def steady_state_top_oil_rise(self, K: float) -> float:
        """Δθ_TO — steady-state top-oil rise at load factor K."""
        return self.r.delta_theta_to_r * ((K**2 * self.r.R + 1) / (self.r.R + 1)) ** self.r.n

    def steady_state_hotspot_rise_over_top_oil(self, K: float) -> float:
        """Δθ_H — steady-state hot-spot rise over top-oil at load factor K."""
        return self.r.delta_theta_h_r * (K ** (2 * self.r.m))

    def steady_state_hotspot(self, K: float, theta_ambient: float) -> float:
        """θ_H — absolute hot-spot temperature (°C) at steady state."""
        return (theta_ambient
                + self.steady_state_top_oil_rise(K)
                + self.steady_state_hotspot_rise_over_top_oil(K))

    # ── Relative ageing rate ─────────────────────────────────────────────

    def relative_ageing_rate(self, theta_h_c: float) -> float:
        """
        V — relative ageing rate (dimensionless).
        V=1.0 at 110°C (normal reference), V>1 above, V<1 below.
        Uses Arrhenius equation per IEC 60076-7 Section 6.3.
        """
        theta_h_k = 273.0 + theta_h_c
        return float(np.exp(
            self.r.arrhenius_ea_over_kb * (1.0 / self.r.theta_ref_k - 1.0 / theta_h_k)
        ))

    # ── Dynamic step (Euler integration) ────────────────────────────────

    def step(self, K: float, theta_ambient_c: float, dt_minutes: float) -> dict:
        """
        Advance thermal state by dt_minutes at load factor K.
        
        Returns a dict of all computed quantities for this time step,
        ready to be used as a feature row.
        """
        self.theta_ambient_c = theta_ambient_c

        # Steady-state targets
        dtheta_to_ult = self.steady_state_top_oil_rise(K)
        dtheta_h_ult  = self.steady_state_hotspot_rise_over_top_oil(K)
        theta_to_ult  = theta_ambient_c + dtheta_to_ult
        theta_h_ult   = theta_to_ult + dtheta_h_ult

        # Exponential approach to steady state (IEC 60076-7 Annex G)
        alpha_to = 1.0 - np.exp(-dt_minutes / self.r.tau_to_min)
        alpha_h  = 1.0 - np.exp(-dt_minutes / self.r.tau_h_min)

        self.theta_to_c += alpha_to * (theta_to_ult - self.theta_to_c)
        self.theta_h_c  += alpha_h  * (theta_h_ult  - self.theta_h_c)

        V = self.relative_ageing_rate(self.theta_h_c)
        dt_hours = dt_minutes / 60.0
        self.theta_h_c  # hot-spot
        self._cumulative_ageing += V * dt_hours
        self._elapsed_hours     += dt_hours

        return {
            "K":                    K,
            "theta_ambient_c":      theta_ambient_c,
            "theta_to_c":           self.theta_to_c,
            "theta_h_c":            self.theta_h_c,
            "delta_theta_to_c":     self.theta_to_c - theta_ambient_c,
            "delta_theta_h_c":      self.theta_h_c  - self.theta_to_c,
            "V":                    V,
            "cumulative_ageing_h":  self._cumulative_ageing,
            "elapsed_h":            self._elapsed_hours,
            "overtemp_flag":        int(self.theta_h_c > self.r.hotspot_continuous_c),
            "overload_flag":        int(K > 1.0),
        }

    @property
    def cumulative_ageing_hours(self) -> float:
        return self._cumulative_ageing

    def reset(self, ambient_c: float = 20.0):
        self.theta_ambient_c = ambient_c
        self.theta_to_c = ambient_c + self.r.delta_theta_to_r
        self.theta_h_c  = ambient_c + self.r.delta_theta_to_r + self.r.delta_theta_h_r
        self._cumulative_ageing = 0.0
        self._elapsed_hours = 0.0


# ---------------------------------------------------------------------------
# Feature Extractor — batch processing + window features
# ---------------------------------------------------------------------------

class TransformerFeatureExtractor:
    """
    Converts raw time-series sensor readings into a rich feature matrix
    suitable for ML model training and inference.

    Input DataFrame columns expected:
        timestamp       — pd.Timestamp or unix ms
        load_current_a  — measured load current (A)
        top_oil_temp_c  — measured top-oil temperature (°C)
        ambient_temp_c  — measured ambient temperature (°C)
        voltage_v       — measured terminal voltage (V) [optional]
        power_factor    — measured power factor [optional]

    Output feature columns (per row):
        Physics features (IEC 60076-7):
            K, K_sq, theta_to_c, theta_h_c (computed or measured),
            delta_theta_to_c, delta_theta_h_c, V, cumulative_ageing_h

        Rolling window features (1h, 6h, 24h):
            K_mean_1h, K_max_1h, K_std_1h
            theta_h_mean_6h, theta_h_max_6h
            V_mean_24h (average ageing rate over 24 h)
            ... etc.

        Derived indicators:
            thermal_margin_c  — hotspot limit minus current hotspot
            load_trend        — sign of derivative of K over past 30 min
            overtemp_flag     — binary: hotspot > continuous limit
            overload_flag     — binary: K > 1.0
    """

    # Window sizes in minutes for rolling feature computation
    WINDOWS = {
        "30min": 30,
        "1h":    60,
        "6h":    360,
        "24h":   1440,
    }

    def __init__(self, rating: TransformerRating, sample_interval_min: float = 1.0):
        self.rating = rating
        self.sample_interval_min = sample_interval_min
        self.thermal_model = IEC60076_7_ThermalModel(rating)

    def compute_load_factor(self, current_a: np.ndarray) -> np.ndarray:
        """K = I / I_rated — clipped to [0, 2.0] for numerical stability."""
        K = current_a / self.rating.rated_current_a
        return np.clip(K, 0.0, 2.0)

    def compute_from_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature engineering pass over a raw sensor DataFrame.
        
        Runs the IEC 60076-7 dynamic model forward in time to compute
        hot-spot temperature and ageing, then appends rolling features
        and derived indicators.
        
        Returns a new DataFrame aligned with the input index.
        """
        df = df.copy().sort_values("timestamp").reset_index(drop=True)
        n = len(df)

        # ── Step 1: load factor ─────────────────────────────────────────
        df["K"] = self.compute_load_factor(df["load_current_a"].values)
        df["K_sq"] = df["K"] ** 2

        # ── Step 2: IEC 60076-7 dynamic model ──────────────────────────
        self.thermal_model.reset(
            ambient_c=float(df["ambient_temp_c"].iloc[0])
        )

        thermal_rows = []
        for i in range(n):
            row = self.thermal_model.step(
                K=float(df["K"].iloc[i]),
                theta_ambient_c=float(df["ambient_temp_c"].iloc[i]),
                dt_minutes=self.sample_interval_min
            )
            thermal_rows.append(row)

        thermal_df = pd.DataFrame(thermal_rows, index=df.index)

        # If measured top-oil temperature is available, override the computed value
        # but keep the computed hot-spot (more conservative for safety)
        if "top_oil_temp_c" in df.columns:
            thermal_df["theta_to_c"] = df["top_oil_temp_c"].values
            # Recompute hot-spot from measured top-oil + model rise
            dtheta_h = np.array([
                self.thermal_model.steady_state_hotspot_rise_over_top_oil(K)
                for K in df["K"].values
            ])
            thermal_df["theta_h_c"] = thermal_df["theta_to_c"] + dtheta_h
            thermal_df["delta_theta_to_c"] = (
                thermal_df["theta_to_c"] - df["ambient_temp_c"].values
            )
            thermal_df["delta_theta_h_c"] = dtheta_h
            # Recompute ageing with corrected hot-spot
            thermal_df["V"] = thermal_df["theta_h_c"].apply(
                self.thermal_model.relative_ageing_rate
            )

        # ── Step 3: merge base features ─────────────────────────────────
        # Drop columns already in df before concat to avoid duplicates
        thermal_drop = [c for c in thermal_df.columns if c in df.columns]
        features = pd.concat([df, thermal_df.drop(columns=thermal_drop)], axis=1)

        # ── Step 4: rolling window statistics ───────────────────────────
        freq_min = int(self.sample_interval_min)
        freq_str = f"{freq_min}min"
        if "timestamp" in features.columns:
            features = features.set_index(
                pd.to_datetime(features["timestamp"])
            )

        for label, window_min in self.WINDOWS.items():
            n_samples = max(1, int(window_min / self.sample_interval_min))
            roll = features[["K", "theta_h_c", "V", "theta_to_c"]].rolling(
                window=n_samples, min_periods=1
            )
            features[f"K_mean_{label}"]        = roll["K"].mean()
            features[f"K_max_{label}"]         = roll["K"].max()
            features[f"K_std_{label}"]         = roll["K"].std().fillna(0)
            features[f"theta_h_mean_{label}"]  = roll["theta_h_c"].mean()
            features[f"theta_h_max_{label}"]   = roll["theta_h_c"].max()
            features[f"V_mean_{label}"]        = roll["V"].mean()

        # ── Step 5: derived indicators ───────────────────────────────────
        features["thermal_margin_c"] = (
            self.rating.hotspot_continuous_c - features["theta_h_c"]
        )
        features["hotspot_to_limit_pct"] = (
            features["theta_h_c"] / self.rating.hotspot_limit_c * 100.0
        )

        # Load trend: positive derivative of K over last 30 min
        k_shift = max(1, int(30 / self.sample_interval_min))
        features["load_trend"] = np.sign(
            features["K"] - features["K"].shift(k_shift).fillna(features["K"])
        )

        # Ageing acceleration: V / 1.0 (how much faster than normal ageing)
        features["ageing_acceleration"] = features["V"]

        # Time-of-day and seasonal context
        if hasattr(features.index, "hour"):
            features["hour_of_day"]   = features.index.hour
            features["day_of_week"]   = features.index.dayofweek
            features["month"]         = features.index.month
            features["is_peak_hours"] = features["hour_of_day"].between(
                7, 22
            ).astype(int)

        features = features.reset_index(drop=True)
        return features

    def get_feature_names(self) -> list:
        """
        Returns the ordered list of feature column names that the
        ML model expects as input — used for ONNX input validation.
        """
        base = [
            "K", "K_sq",
            "theta_ambient_c", "theta_to_c", "theta_h_c",
            "delta_theta_to_c", "delta_theta_h_c",
            "V", "cumulative_ageing_h",
            "thermal_margin_c", "hotspot_to_limit_pct",
            "load_trend", "ageing_acceleration",
            "overtemp_flag", "overload_flag",
        ]
        rolling = []
        for label in self.WINDOWS:
            rolling += [
                f"K_mean_{label}", f"K_max_{label}", f"K_std_{label}",
                f"theta_h_mean_{label}", f"theta_h_max_{label}",
                f"V_mean_{label}",
            ]
        temporal = [
            "hour_of_day", "day_of_week", "month", "is_peak_hours"
        ]
        return base + rolling + temporal


# ---------------------------------------------------------------------------
# Synthetic Data Generator (for training without real sensor data)
# ---------------------------------------------------------------------------

class TransformerDataGenerator:
    """
    Generates realistic synthetic transformer operating history for
    model training and testing.

    Simulates:
      - Daily load cycles (morning ramp, afternoon peak, overnight trough)
      - Seasonal ambient temperature variation
      - Random overload events
      - Progressive insulation ageing (degrades thermal performance slightly)
      - Injected fault precursors near end-of-life (rising hot-spot baseline)
    """

    def __init__(self,
                 rating: TransformerRating,
                 seed: int = 42):
        self.rating = rating
        self.rng    = np.random.default_rng(seed)

    def generate(self,
                 n_days: int = 365,
                 sample_interval_min: float = 15.0,
                 age_at_start_years: float = 0.0) -> pd.DataFrame:
        """
        Generate n_days of synthetic operating history.

        Returns a raw sensor DataFrame (before feature extraction) with
        added ground-truth labels:
            rul_days         — remaining useful life in days
            ageing_state     — float in [0,1]: 0=new, 1=end of life
            fault_imminent   — binary: 1 if fault within 30 days
        """
        n_samples = int(n_days * 24 * 60 / sample_interval_min)
        timestamps = pd.date_range(
            start="2024-01-01",
            periods=n_samples,
            freq=f"{int(sample_interval_min)}min"
        )

        # Normal transformer paper life: ~40 years = 350,400 hours
        # Ageing consumes this at a rate driven by hot-spot temperature
        design_life_hours = 40 * 365 * 24
        start_ageing_hours = age_at_start_years * 365 * 24

        hours = np.arange(n_samples) * (sample_interval_min / 60.0)
        day_of_year = (hours / 24.0) % 365.0

        # ── Ambient temperature (seasonal, daily variation) ───────────
        seasonal = 10.0 * np.sin(2 * np.pi * day_of_year / 365.0 - np.pi / 2)
        daily    = 5.0  * np.sin(2 * np.pi * (hours % 24) / 24.0 - np.pi / 2)
        noise    = self.rng.normal(0, 0.5, n_samples)
        ambient  = 15.0 + seasonal + daily + noise

        # ── Load profile (daily pattern + overload events) ────────────
        hour_of_day = (hours % 24)
        # Base load follows a realistic daily pattern
        base_load = (
            0.4                                                   # overnight minimum
            + 0.3 * np.clip((hour_of_day - 6) / 3, 0, 1)        # morning ramp
            + 0.15 * np.clip((hour_of_day - 17) / 2, 0, 1)      # evening peak
            - 0.15 * np.clip((hour_of_day - 21) / 2, 0, 1)      # evening trough
        )
        base_load += self.rng.normal(0, 0.03, n_samples)

        # Inject random overload events (3% of samples)
        overload_mask = self.rng.random(n_samples) < 0.03
        overload_K = self.rng.uniform(1.05, 1.4, n_samples) * overload_mask
        K = np.clip(base_load + overload_K, 0.1, 1.8)

        # ── Progressive fault precursor ────────────────────────────────
        # As transformer ages, insulation degrades → hot-spot rises faster
        # In the last 10% of life, inject a linear drift upward
        ageing_fraction = (start_ageing_hours + hours / design_life_hours)
        ageing_fraction = np.clip(ageing_fraction, 0, 1)
        fault_precursor = np.where(
            ageing_fraction > 0.9,
            (ageing_fraction - 0.9) * 50.0,  # up to +5°C drift near EOL
            0.0
        )

        # ── Convert load to current ────────────────────────────────────
        current_a = K * self.rating.rated_current_a

        # ── Measurement noise ──────────────────────────────────────────
        current_a += self.rng.normal(0, 2.0, n_samples)

        # ── Build raw DataFrame ────────────────────────────────────────
        df = pd.DataFrame({
            "timestamp":       timestamps,
            "load_current_a":  np.clip(current_a, 0, None),
            "ambient_temp_c":  ambient,
        })

        # ── Run IEC thermal model to get top-oil temperature ───────────
        model = IEC60076_7_ThermalModel(self.rating, ambient_c=float(ambient[0]))
        top_oil = []
        for i in range(n_samples):
            state = model.step(
                K=float(K[i]),
                theta_ambient_c=float(ambient[i]),
                dt_minutes=sample_interval_min
            )
            top_oil.append(state["theta_to_c"] + fault_precursor[i]
                           + self.rng.normal(0, 0.3))

        df["top_oil_temp_c"] = top_oil

        # ── Ground-truth labels ────────────────────────────────────────
        # Cumulative ageing in hours (recomputed from thermal model output)
        thermal_extractor = TransformerFeatureExtractor(
            self.rating, sample_interval_min
        )
        features = thermal_extractor.compute_from_timeseries(df)
        cumulative_h = features["cumulative_ageing_h"].values + start_ageing_hours

        rul_hours = np.maximum(0.0, design_life_hours - cumulative_h)
        rul_days  = rul_hours / 24.0

        df["rul_days"]       = rul_days
        df["ageing_state"]   = np.clip(cumulative_h / design_life_hours, 0, 1)
        df["fault_imminent"] = (rul_days < 30).astype(int)

        return df

                   
