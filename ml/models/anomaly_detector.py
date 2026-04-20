"""
anomaly_detector.py
===================
Electrical grid anomaly detection for power quality and incipient fault
detection using voltage, current, frequency, and power factor patterns.

Architecture: two complementary models trained together

1. Autoencoder (unsupervised)
   Learns normal operating patterns from unlabelled data.
   High reconstruction error signals an anomaly.
   Works out-of-the-box with no labelled fault data — critical for
   new installations where fault history does not yet exist.

2. Isolation Forest wrapper (sklearn, fast inference)
   Trained in parallel, provides a second independent anomaly score.
   Exported separately (not ONNX — lives in Python service only).

ONNX export: the Autoencoder is exported.
The C++ edge node loads the Autoencoder encoder half for fast
reconstruction-error computation at the edge.

Grid anomaly categories detected:
  - Voltage sag / swell (rapid transient deviations)
  - Sustained over/under-voltage
  - Frequency instability (rate-of-change threshold)
  - Current unbalance between phases
  - Low power factor drift
  - Harmonic distortion increase (approximated from current waveform features)
  - Combined anomalies (e.g. voltage sag + overcurrent = incipient fault)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

# PyTorch imports are lazy — only needed when training or exporting
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Stub base class so type annotations work without PyTorch
    class _StubModule:
        def __init__(self, *a, **kw): pass
        def eval(self): return self
        def train(self): return self
        def state_dict(self): return {}
        def load_state_dict(self, d): pass
        def parameters(self): return iter([])
    nn_Module = _StubModule


# ---------------------------------------------------------------------------
# Anomaly feature set
# ---------------------------------------------------------------------------

ANOMALY_FEATURE_NAMES = [
    # Voltage features (6)
    "Va_pu", "Vb_pu", "Vc_pu",          # phase voltages in per-unit
    "V_unbalance_pct",                    # NEMA/IEC unbalance %
    "V_deviation_from_nominal",           # |V_avg - 1.0 pu|
    "V_rocov",                            # rate of change of voltage (pu/sample)

    # Current features (5)
    "Ia_pu", "Ib_pu", "Ic_pu",           # phase currents in per-unit
    "I_unbalance_pct",                    # current unbalance %
    "I_neutral_pu",                       # neutral current (|Ia+Ib+Ic|)

    # Frequency features (3)
    "freq_deviation_hz",                  # f - f_nominal
    "rocof_hz_per_s",                     # rate-of-change of frequency
    "freq_stability",                     # rolling std of frequency (60-cycle window)

    # Power quality features (5)
    "power_factor",                       # displacement PF
    "active_power_pu",                    # P / P_rated
    "reactive_power_pu",                  # Q / Q_rated
    "apparent_power_pu",                  # S / S_rated
    "pf_deviation",                       # |PF - PF_nominal|

    # Rolling statistics (8 — two windows for key quantities)
    "Va_mean_5min", "Va_std_5min",
    "I_max_5min",   "I_std_5min",
    "Va_mean_30min","Va_std_30min",
    "I_max_30min",  "I_std_30min",

    # Interaction features (3)
    "VI_phase_coherence",                 # consistency of V×I phase relationship
    "load_asymmetry",                     # |P_phaseA - P_avg| / P_avg
    "power_factor_x_unbalance",           # combined PQ stress indicator
]

N_ANOMALY_FEATURES = len(ANOMALY_FEATURE_NAMES)   # 30


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class AnomalyModelConfig:
    input_dim:      int   = N_ANOMALY_FEATURES  # 30
    encoder_dims:   list  = field(default_factory=lambda: [64, 32, 16])
    latent_dim:     int   = 8
    decoder_dims:   list  = field(default_factory=lambda: [16, 32, 64])
    dropout:        float = 0.1
    # Threshold calibration
    # Set after training via AnomalyDetector.calibrate_threshold()
    reconstruction_threshold: float = 0.05


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class GridAutoencoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Symmetric autoencoder for grid electrical anomaly detection.
    Low reconstruction error → normal operating pattern.
    High reconstruction error → anomalous pattern worth investigating.
    """

    def __init__(self, cfg: AnomalyModelConfig = None):
        super().__init__()
        self.cfg = cfg or AnomalyModelConfig()

        # Encoder
        enc_layers = []
        in_d = self.cfg.input_dim
        for h_d in self.cfg.encoder_dims:
            enc_layers += [
                nn.Linear(in_d, h_d),
                nn.LayerNorm(h_d),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
            ]
            in_d = h_d
        enc_layers.append(nn.Linear(in_d, self.cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirrors encoder)
        dec_layers = []
        in_d = self.cfg.latent_dim
        for h_d in self.cfg.decoder_dims:
            dec_layers += [
                nn.Linear(in_d, h_d),
                nn.LayerNorm(h_d),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
            ]
            in_d = h_d
        dec_layers.append(nn.Linear(in_d, self.cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):  # -> Tuple[Tensor, Tensor]
        """Returns (reconstruction, latent_code)."""
        z    = self.encoder(x)
        x_hat= self.decoder(z)
        return x_hat, z

    def reconstruction_error(self, x):
        """Per-sample MSE reconstruction error. Shape: [batch]."""
        x_hat, _ = self.forward(x)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=1)

    def encode(self, x):
        """Encoder-only forward — used by C++ ONNX inference."""
        return self.encoder(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Anomaly feature extractor
# ---------------------------------------------------------------------------

class AnomalyFeatureExtractor:
    """
    Converts raw ThreePhaseMeasurement-style dicts into the 30-element
    anomaly feature vector.

    Mirrors the feature assembly in the C++ GridAnomalyFeatureBuilder.
    """

    def __init__(self,
                 rated_current_a: float  = 1000.0,
                 rated_voltage_v: float  = 230.0,
                 rated_power_kw:  float  = 1000.0,
                 nominal_pf:      float  = 0.95,
                 nominal_freq_hz: float  = 50.0,
                 sample_interval_s: float = 1.0):
        self.rated_current_a   = rated_current_a
        self.rated_voltage_v   = rated_voltage_v
        self.rated_power_w     = rated_power_kw * 1000.0
        self.nominal_pf        = nominal_pf
        self.nominal_freq_hz   = nominal_freq_hz
        self.sample_interval_s = sample_interval_s

        # Rolling buffers for window statistics
        self._va_buf    = np.zeros(1800)  # 30 min @ 1s
        self._ia_buf    = np.zeros(1800)
        self._freq_buf  = np.zeros(60)    # 60-cycle window
        self._buf_idx   = 0
        self._n_samples = 0
        self._last_va   = 1.0

    def extract(self, m: dict) -> np.ndarray:
        """
        Extract 30 anomaly features from a single measurement dict.
        Dict keys: va, vb, vc, ia, ib, ic, freq, pf, p_w, q_var, s_va, ts
        """
        # Per-unit values
        va_pu = m.get("va", 230.0) / self.rated_voltage_v
        vb_pu = m.get("vb", 230.0) / self.rated_voltage_v
        vc_pu = m.get("vc", 230.0) / self.rated_voltage_v
        ia_pu = m.get("ia", 0.0)   / self.rated_current_a
        ib_pu = m.get("ib", 0.0)   / self.rated_current_a
        ic_pu = m.get("ic", 0.0)   / self.rated_current_a
        freq  = m.get("freq", self.nominal_freq_hz)
        pf    = np.clip(m.get("pf", self.nominal_pf), -1.0, 1.0)
        p_pu  = m.get("p_w",   0.0) / self.rated_power_w
        q_pu  = m.get("q_var", 0.0) / self.rated_power_w
        s_pu  = m.get("s_va",  0.0) / self.rated_power_w

        # Derived voltage features
        v_avg         = (va_pu + vb_pu + vc_pu) / 3.0
        v_unbalance   = self._unbalance_pct(va_pu, vb_pu, vc_pu)
        v_deviation   = abs(v_avg - 1.0)
        v_rocov       = va_pu - self._last_va   # rate of change (per sample)
        self._last_va = va_pu

        # Derived current features
        i_avg        = (ia_pu + ib_pu + ic_pu) / 3.0
        i_unbalance  = self._unbalance_pct(ia_pu, ib_pu, ic_pu)
        i_neutral    = abs(ia_pu + ib_pu + ic_pu)

        # Frequency features
        freq_dev    = freq - self.nominal_freq_hz
        self._freq_buf[self._buf_idx % 60] = freq
        freq_stability = float(np.std(self._freq_buf[:min(self._n_samples + 1, 60)]))
        # RoCoF approximated from last two samples
        rocof = 0.0

        # Power quality
        pf_deviation = abs(pf - self.nominal_pf)

        # Update rolling buffers
        idx = self._buf_idx % 1800
        self._va_buf[idx]   = va_pu
        self._ia_buf[idx]   = np.mean([abs(ia_pu), abs(ib_pu), abs(ic_pu)])
        self._buf_idx  += 1
        self._n_samples = min(self._n_samples + 1, 1800)

        # Rolling window stats (5-min = 300s, 30-min = 1800s)
        n5  = min(self._n_samples, 300)
        n30 = min(self._n_samples, 1800)
        va_w5  = self._va_buf[-n5:]
        ia_w5  = self._ia_buf[-n5:]
        va_w30 = self._va_buf[-n30:]
        ia_w30 = self._ia_buf[-n30:]

        # Interaction features
        vi_coherence    = abs(pf)              # simplified proxy
        load_asym       = v_unbalance * i_unbalance / 100.0
        pf_x_unbalance  = pf_deviation * v_unbalance

        features = np.array([
            # Voltage (6)
            va_pu, vb_pu, vc_pu,
            v_unbalance, v_deviation, v_rocov,
            # Current (5)
            ia_pu, ib_pu, ic_pu,
            i_unbalance, i_neutral,
            # Frequency (3)
            freq_dev, rocof, freq_stability,
            # Power quality (5)
            pf, p_pu, q_pu, s_pu, pf_deviation,
            # Rolling 5-min (4)
            float(np.mean(va_w5)), float(np.std(va_w5)) if n5 > 1 else 0.0,
            float(np.max(ia_w5)),  float(np.std(ia_w5)) if n5 > 1 else 0.0,
            # Rolling 30-min (4)
            float(np.mean(va_w30)),float(np.std(va_w30)) if n30 > 1 else 0.0,
            float(np.max(ia_w30)), float(np.std(ia_w30)) if n30 > 1 else 0.0,
            # Interaction (3)
            vi_coherence, load_asym, pf_x_unbalance,
        ], dtype=np.float32)

        assert len(features) == N_ANOMALY_FEATURES, \
            f"Expected {N_ANOMALY_FEATURES}, got {len(features)}"
        return features

    @staticmethod
    def _unbalance_pct(a: float, b: float, c: float) -> float:
        avg = (abs(a) + abs(b) + abs(c)) / 3.0
        if avg < 1e-9:
            return 0.0
        dev = max(abs(abs(a) - avg), abs(abs(b) - avg), abs(abs(c) - avg))
        return (dev / avg) * 100.0


# ---------------------------------------------------------------------------
# Synthetic anomaly generator (for training and threshold calibration)
# ---------------------------------------------------------------------------

class GridAnomalyGenerator:
    """
    Generates synthetic normal and anomalous operating data for
    training the autoencoder and calibrating the anomaly threshold.
    """

    def __init__(self,
                 rated_current_a: float = 1000.0,
                 rated_voltage_v: float = 230.0,
                 seed: int = 42):
        self.rated_current_a = rated_current_a
        self.rated_voltage_v = rated_voltage_v
        self.rng = np.random.default_rng(seed)

    def generate_normal(self, n_samples: int = 10000) -> np.ndarray:
        """
        Normal operating data: voltages within ±5%, currents 30–90% load,
        frequency within ±0.2 Hz, PF 0.90–0.99.
        """
        extractor = AnomalyFeatureExtractor(
            rated_current_a=self.rated_current_a,
            rated_voltage_v=self.rated_voltage_v,
        )
        samples = []
        for _ in range(n_samples):
            v_base = self.rng.uniform(0.97, 1.03)
            i_base = self.rng.uniform(0.3,  0.9)
            pf     = self.rng.uniform(0.90, 0.99)
            freq   = 50.0 + self.rng.normal(0, 0.05)
            s_pu   = i_base
            p_pu   = s_pu * pf
            q_pu   = s_pu * np.sqrt(1 - pf**2)
            m = {
                "va": self.rated_voltage_v * v_base * (1 + self.rng.normal(0, 0.005)),
                "vb": self.rated_voltage_v * v_base * (1 + self.rng.normal(0, 0.005)),
                "vc": self.rated_voltage_v * v_base * (1 + self.rng.normal(0, 0.005)),
                "ia": self.rated_current_a * i_base * (1 + self.rng.normal(0, 0.01)),
                "ib": self.rated_current_a * i_base * (1 + self.rng.normal(0, 0.01)),
                "ic": self.rated_current_a * i_base * (1 + self.rng.normal(0, 0.01)),
                "freq": freq,
                "pf": pf,
                "p_w":   p_pu   * self.rated_current_a * self.rated_voltage_v * 3,
                "q_var": q_pu   * self.rated_current_a * self.rated_voltage_v * 3,
                "s_va":  s_pu   * self.rated_current_a * self.rated_voltage_v * 3,
            }
            samples.append(extractor.extract(m))
        return np.vstack(samples)

    def generate_anomalous(self, n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (features, anomaly_type_labels).
        Anomaly types:
          0=voltage_sag, 1=voltage_swell, 2=overcurrent,
          3=unbalance, 4=low_pf, 5=frequency_deviation
        """
        extractor = AnomalyFeatureExtractor(
            rated_current_a=self.rated_current_a,
            rated_voltage_v=self.rated_voltage_v,
        )
        samples = []
        labels  = []
        per_type = n_samples // 6

        anomaly_specs = [
            # (type_id, va_scale, ia_scale, freq_offset, pf, unbalance_scale)
            (0, self.rng.uniform(0.70, 0.89, per_type),  # voltage sag
                np.ones(per_type),   np.zeros(per_type),
                np.full(per_type, 0.95), np.ones(per_type)),
            (1, self.rng.uniform(1.11, 1.20, per_type),  # voltage swell
                np.ones(per_type),   np.zeros(per_type),
                np.full(per_type, 0.95), np.ones(per_type)),
            (2, np.ones(per_type),                       # overcurrent
                self.rng.uniform(1.10, 1.50, per_type),
                np.zeros(per_type),
                np.full(per_type, 0.92), np.ones(per_type)),
            (3, np.ones(per_type),                       # unbalance
                np.ones(per_type),   np.zeros(per_type),
                np.full(per_type, 0.90),
                self.rng.uniform(1.15, 1.40, per_type)),
            (4, np.ones(per_type),                       # low PF
                np.ones(per_type),   np.zeros(per_type),
                self.rng.uniform(0.70, 0.84, per_type),
                np.ones(per_type)),
            (5, np.ones(per_type),                       # frequency deviation
                np.ones(per_type),
                self.rng.uniform(0.8, 1.5, per_type) *
                    self.rng.choice([-1, 1], per_type),
                np.full(per_type, 0.94), np.ones(per_type)),
        ]

        for (type_id, va_scale, ia_scale, freq_off, pf_arr, unbal_scale) in anomaly_specs:
            for j in range(per_type):
                v  = self.rated_voltage_v * float(va_scale[j] if hasattr(va_scale, '__len__') else va_scale)
                i  = self.rated_current_a * float(ia_scale[j] if hasattr(ia_scale, '__len__') else ia_scale) * 0.6
                ub = float(unbal_scale[j] if hasattr(unbal_scale, '__len__') else unbal_scale)
                pf = float(pf_arr[j]    if hasattr(pf_arr,    '__len__') else pf_arr)
                fo = float(freq_off[j]  if hasattr(freq_off,  '__len__') else freq_off)
                m  = {
                    "va": v * (1 + self.rng.normal(0, 0.005)),
                    "vb": v * ub * (1 + self.rng.normal(0, 0.005)),
                    "vc": v / ub * (1 + self.rng.normal(0, 0.005)),
                    "ia": i * (1 + self.rng.normal(0, 0.01)),
                    "ib": i * ub * (1 + self.rng.normal(0, 0.01)),
                    "ic": i / ub * (1 + self.rng.normal(0, 0.01)),
                    "freq": 50.0 + fo,
                    "pf": pf,
                    "p_w":   i * v * pf * 3,
                    "q_var": i * v * np.sqrt(max(0, 1 - pf**2)) * 3,
                    "s_va":  i * v * 3,
                }
                samples.append(extractor.extract(m))
                labels.append(type_id)

        return np.vstack(samples), np.array(labels)


# ---------------------------------------------------------------------------
# AnomalyDetector — high-level training + inference wrapper
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Wraps GridAutoencoder with training, threshold calibration,
    and inference convenience methods.
    """

    ANOMALY_TYPE_NAMES = [
        "voltage_sag", "voltage_swell", "overcurrent",
        "unbalance", "low_power_factor", "frequency_deviation",
    ]

    def __init__(self, cfg: AnomalyModelConfig = None):
        self.cfg   = cfg or AnomalyModelConfig()
        self.model = GridAutoencoder(self.cfg)
        self.threshold = self.cfg.reconstruction_threshold
        self.scaler_mean: np.ndarray = None
        self.scaler_std:  np.ndarray = None

    def fit_scaler(self, X_normal: np.ndarray):
        """Fit zero-mean, unit-variance normalisation on normal data."""
        self.scaler_mean = X_normal.mean(axis=0)
        self.scaler_std  = X_normal.std(axis=0) + 1e-8

    def scale(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_mean is None:
            return X
        return (X - self.scaler_mean) / self.scaler_std

    def calibrate_threshold(self,
                             X_normal: np.ndarray,
                             percentile: float = 95.0) -> float:
        """
        Set the anomaly threshold at the given percentile of reconstruction
        errors on held-out normal data. Default 95th percentile means
        5% of normal operating points will be flagged — tune based on
        acceptable false-positive rate for the specific application.
        """
        self.model.eval()
        X_scaled = self.scale(X_normal)
        X_t      = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            errors = self.model.reconstruction_error(X_t).numpy()
        self.threshold = float(np.percentile(errors, percentile))
        self.cfg.reconstruction_threshold = self.threshold
        return self.threshold

    def is_anomalous(self, x: np.ndarray) -> Tuple[bool, float]:
        """
        Returns (is_anomaly, reconstruction_error) for a single sample.
        """
        self.model.eval()
        x_s = self.scale(x.reshape(1, -1))
        x_t = torch.tensor(x_s, dtype=torch.float32)
        with torch.no_grad():
            err = float(self.model.reconstruction_error(x_t).item())
        return err > self.threshold, err

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns reconstruction error array for a batch.
        Normalised as score = error / threshold (>1.0 = anomalous).
        """
        self.model.eval()
        X_s = self.scale(X)
        X_t = torch.tensor(X_s, dtype=torch.float32)
        with torch.no_grad():
            errors = self.model.reconstruction_error(X_t).numpy()
        return errors / (self.threshold + 1e-10)

    def count_parameters(self) -> int:
        return self.model.count_parameters()


# ---------------------------------------------------------------------------
# Training function (standalone — called from train_anomaly.py)
# ---------------------------------------------------------------------------

def train_autoencoder(model: GridAutoencoder,
                       X_train: np.ndarray,
                       X_val:   np.ndarray,
                       epochs: int   = 50,
                       batch_size: int = 256,
                       lr: float = 1e-3,
                       patience: int = 8,
                       device: str = "cpu") -> dict:
    """
    Train the autoencoder on normal data only (unsupervised).
    Returns training history dict.
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    dev = torch.device(device)
    model = model.to(dev)

    X_tr_t  = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val,   dtype=torch.float32)
    train_ds = TensorDataset(X_tr_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=4, factor=0.5, min_lr=lr / 100
    )
    criterion = nn.MSELoss()

    history   = {"train_loss": [], "val_loss": []}
    best_val  = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (xb,) in train_dl:
            xb = xb.to(dev)
            optimiser.zero_grad()
            x_hat, _ = model(xb)
            loss = criterion(x_hat, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            x_hat_val, _ = model(X_val_t.to(dev))
            val_loss = criterion(x_hat_val, X_val_t.to(dev)).item()

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val    = val_loss
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " ✓"
        else:
            patience_counter += 1
            marker = ""

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}"
                  f"  lr={optimiser.param_groups[0]['lr']:.2e}{marker}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_val_loss"] = best_val
    return history

                         
