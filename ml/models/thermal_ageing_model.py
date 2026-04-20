"""
thermal_ageing_model.py
=======================
PyTorch model for transformer thermal ageing and remaining useful
life (RUL) prediction.

Architecture
------------
We use a hybrid model combining:

1. A shallow physics residual branch — takes the IEC 60076-7 computed
   features directly (V, θ_H, cumulative ageing) and learns a small
   correction to the physics model rather than replacing it.

2. A temporal context branch — a 1D CNN that sees a window of rolling
   statistics (K_mean, θ_H_max over multiple horizons) and extracts
   temporal patterns that the steady-state physics model misses.

3. A fusion MLP — merges both branches and outputs:
     - rul_pred_days   (regression head)
     - ageing_state    (regression head, [0,1])
     - fault_imminent  (binary classification head, sigmoid)

This architecture is deliberately modest in size (~50K parameters) so
it can be exported to ONNX and run on an edge device alongside the C++
firmware using ONNX Runtime Lite.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # Input dimensions
    n_physics_features: int = 15   # IEC 60076-7 physics features
    n_rolling_features: int = 24   # rolling window statistics
    n_temporal_features: int = 4   # hour, day, month, is_peak
    input_dim: int = 43            # total (computed in __post_init__)

    # Physics branch
    physics_hidden: int = 64
    physics_layers: int = 2

    # Rolling feature CNN
    cnn_channels: int = 32
    cnn_kernel:   int = 3

    # Fusion MLP
    fusion_hidden: int = 128
    fusion_layers: int = 3
    dropout_rate:  float = 0.25

    # Output
    rul_max_days: float = 14600.0   # 40 years — normalisation constant

    def __post_init__(self):
        self.input_dim = (self.n_physics_features
                          + self.n_rolling_features
                          + self.n_temporal_features)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class PhysicsResidualBranch(nn.Module):
    """
    Small MLP that learns residuals on top of the IEC physics model.
    Takes the 19 physics features and outputs a 64-dim embedding.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        layers = []
        in_dim = cfg.n_physics_features
        for _ in range(cfg.physics_layers):
            layers += [
                nn.Linear(in_dim, cfg.physics_hidden),
                nn.LayerNorm(cfg.physics_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout_rate / 2),
            ]
            in_dim = cfg.physics_hidden
        self.net = nn.Sequential(*layers)
        self.out_dim = cfg.physics_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalContextBranch(nn.Module):
    """
    1D CNN over the rolling window features.
    Each feature across its 4 window sizes is treated as a channel,
    allowing the CNN to learn temporal trend patterns.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # Reshape rolling features: 6 stats × 4 windows → (6, 4) pseudo-sequence
        self.n_stats   = 6    # K_mean, K_max, K_std, θ_H_mean, θ_H_max, V_mean
        self.n_windows = 4    # 30min, 1h, 6h, 24h

        self.conv1 = nn.Conv1d(
            in_channels=self.n_stats,
            out_channels=cfg.cnn_channels,
            kernel_size=cfg.cnn_kernel,
            padding=cfg.cnn_kernel // 2
        )
        self.conv2 = nn.Conv1d(
            in_channels=cfg.cnn_channels,
            out_channels=cfg.cnn_channels * 2,
            kernel_size=2,
            padding=0
        )
        self.norm1 = nn.BatchNorm1d(cfg.cnn_channels)
        self.norm2 = nn.BatchNorm1d(cfg.cnn_channels * 2)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.out_dim = cfg.cnn_channels * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_rolling_features=24)
        # Reshape to (batch, n_stats=6, n_windows=4)
        b = x.size(0)
        x = x.view(b, self.n_stats, self.n_windows)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)   # (batch, out_dim)
        return x


class FusionMLP(nn.Module):
    """
    Merges physics and temporal embeddings, produces three outputs.
    """

    def __init__(self, in_dim: int, cfg: ModelConfig):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(cfg.fusion_layers):
            layers += [
                nn.Linear(d, cfg.fusion_hidden),
                nn.LayerNorm(cfg.fusion_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout_rate),
            ]
            d = cfg.fusion_hidden
        self.trunk = nn.Sequential(*layers)

        # Three separate output heads
        self.head_rul     = nn.Linear(d, 1)   # RUL in days (normalised)
        self.head_ageing  = nn.Linear(d, 1)   # ageing state [0,1]
        self.head_fault   = nn.Linear(d, 1)   # fault imminent (logit)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        rul    = F.softplus(self.head_rul(h))    # strictly positive
        ageing = torch.sigmoid(self.head_ageing(h))
        fault  = self.head_fault(h)              # raw logit (BCEWithLogitsLoss)
        return rul, ageing, fault


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class TransformerThermalModel(nn.Module):
    """
    Full hybrid model for transformer thermal ageing prediction.

    Inputs (all as a single flat tensor of shape [batch, input_dim]):
        [:n_physics]   — physics features (IEC 60076-7 derived)
        [n_physics:n_physics+n_rolling] — rolling window statistics
        [-n_temporal:] — temporal context (hour, day, month, peak)

    Outputs:
        rul_days       — remaining useful life in days (un-normalised)
        ageing_state   — float in [0, 1]
        fault_logit    — raw logit for fault_imminent classification
    """

    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()

        self.physics_branch  = PhysicsResidualBranch(self.cfg)
        self.temporal_branch = TemporalContextBranch(self.cfg)

        # Temporal context (hour, day, month, peak) — simple embedding
        self.temporal_proj = nn.Sequential(
            nn.Linear(self.cfg.n_temporal_features, 16),
            nn.GELU(),
        )

        fusion_in = (
            self.physics_branch.out_dim
            + self.temporal_branch.out_dim
            + 16  # temporal projection
        )
        self.fusion = FusionMLP(fusion_in, self.cfg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        np = self.cfg.n_physics_features
        nr = self.cfg.n_rolling_features

        x_phys    = x[:, :np]
        x_roll    = x[:, np:np + nr]
        x_temp    = x[:, np + nr:]

        h_phys = self.physics_branch(x_phys)
        h_roll = self.temporal_branch(x_roll)
        h_temp = self.temporal_proj(x_temp)

        h = torch.cat([h_phys, h_roll, h_temp], dim=1)
        rul_norm, ageing, fault_logit = self.fusion(h)

        # Un-normalise RUL
        rul_days = rul_norm * self.cfg.rul_max_days

        return rul_days, ageing, fault_logit

    def predict(self, x: torch.Tensor) -> dict:
        """
        Convenience method for inference — returns a dict with named outputs
        and fault probability (sigmoid of logit).
        """
        self.eval()
        with torch.no_grad():
            rul, ageing, fault_logit = self.forward(x)
            fault_prob = torch.sigmoid(fault_logit)
        return {
            "rul_days":      rul.squeeze(-1).numpy(),
            "ageing_state":  ageing.squeeze(-1).numpy(),
            "fault_prob":    fault_prob.squeeze(-1).numpy(),
            "fault_imminent": (fault_prob.squeeze(-1) > 0.5).numpy().astype(int),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class TransformerThermalLoss(nn.Module):
    """
    Multi-task loss combining:
      - Huber loss for RUL regression   (robust to outliers)
      - MSE loss for ageing state
      - BCE with logits for fault detection
    
    Weights balance the three tasks — fault detection is up-weighted
    because false negatives (missing a fault) are far more costly than
    false positives.
    """

    def __init__(self,
                 w_rul: float   = 1.0,
                 w_ageing: float = 0.5,
                 w_fault: float  = 3.0):
        super().__init__()
        self.w_rul    = w_rul
        self.w_ageing = w_ageing
        self.w_fault  = w_fault

        self.huber = nn.HuberLoss(delta=100.0)   # delta in days
        self.mse   = nn.MSELoss()
        self.bce   = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([5.0])       # fault class is rare
        )

    def forward(self,
                rul_pred: torch.Tensor,     ageing_pred: torch.Tensor,
                fault_logit: torch.Tensor,
                rul_true: torch.Tensor,     ageing_true: torch.Tensor,
                fault_true: torch.Tensor) -> Tuple[torch.Tensor, dict]:

        loss_rul    = self.huber(rul_pred.squeeze(-1),    rul_true)
        loss_ageing = self.mse  (ageing_pred.squeeze(-1), ageing_true)
        loss_fault  = self.bce  (fault_logit.squeeze(-1), fault_true.float())

        total = (self.w_rul    * loss_rul
               + self.w_ageing * loss_ageing
               + self.w_fault  * loss_fault)

        return total, {
            "loss_rul":    loss_rul.item(),
            "loss_ageing": loss_ageing.item(),
            "loss_fault":  loss_fault.item(),
            "loss_total":  total.item(),
        }

                  
