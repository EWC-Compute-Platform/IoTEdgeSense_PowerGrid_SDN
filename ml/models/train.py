"""
train.py
========
Training pipeline for the TransformerThermalModel.

Usage:
    python ml/models/train.py                          # default: 1 transformer, 2 years
    python ml/models/train.py --days 730 --n-tx 5     # 5 transformers, 730 days each
    python ml/models/train.py --checkpoint best.pt    # resume from checkpoint

The script:
  1. Generates synthetic operating history using TransformerDataGenerator
  2. Extracts IEC 60076-7 physics features via TransformerFeatureExtractor
  3. Builds a PyTorch Dataset with configurable sequence length
  4. Trains TransformerThermalModel with early stopping and LR scheduling
  5. Saves the best checkpoint and a training metrics JSON
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.features.transformer_features import (
    TransformerRating,
    TransformerFeatureExtractor,
    TransformerDataGenerator,
)
from ml.models.thermal_ageing_model import (
    ModelConfig,
    TransformerThermalModel,
    TransformerThermalLoss,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TransformerDataset(Dataset):
    """
    Converts feature DataFrame + labels into (X, y) tensors.
    Each sample is a single time step (not a sequence) — the temporal
    context is encoded in the rolling window features already computed
    by TransformerFeatureExtractor.
    """

    FEATURE_COLS = None   # set in __init__ from extractor

    def __init__(self,
                 features_df: pd.DataFrame,
                 feature_names: list,
                 scaler=None,
                 fit_scaler: bool = False):

        self.feature_names = feature_names

        # Drop rows with NaN (from rolling windows at the start)
        df = features_df[feature_names + ["rul_days", "ageing_state", "fault_imminent"]]
        df = df.dropna().reset_index(drop=True)

        X = df[feature_names].values.astype(np.float32)
        y_rul    = df["rul_days"].values.astype(np.float32)
        y_ageing = df["ageing_state"].values.astype(np.float32)
        y_fault  = df["fault_imminent"].values.astype(np.float32)

        # Feature normalisation
        if scaler is None:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        if fit_scaler:
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)
        self.scaler = scaler

        self.X       = torch.tensor(X,        dtype=torch.float32)
        self.y_rul   = torch.tensor(y_rul,    dtype=torch.float32)
        self.y_aging = torch.tensor(y_ageing, dtype=torch.float32)
        self.y_fault = torch.tensor(y_fault,  dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y_rul[idx],
            self.y_aging[idx],
            self.y_fault[idx],
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    totals = {"loss_total": 0, "loss_rul": 0, "loss_ageing": 0, "loss_fault": 0}
    n = 0
    for X, y_rul, y_ageing, y_fault in loader:
        X, y_rul, y_ageing, y_fault = (
            t.to(device) for t in (X, y_rul, y_ageing, y_fault)
        )
        optimiser.zero_grad()
        rul_pred, ageing_pred, fault_logit = model(X)
        loss, metrics = criterion(
            rul_pred, ageing_pred, fault_logit,
            y_rul, y_ageing, y_fault
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        for k in totals:
            totals[k] += metrics[k] * len(X)
        n += len(X)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    totals = {"loss_total": 0, "loss_rul": 0, "loss_ageing": 0, "loss_fault": 0}
    rul_errors = []
    fault_correct = 0
    n = 0
    for X, y_rul, y_ageing, y_fault in loader:
        X, y_rul, y_ageing, y_fault = (
            t.to(device) for t in (X, y_rul, y_ageing, y_fault)
        )
        rul_pred, ageing_pred, fault_logit = model(X)
        loss, metrics = criterion(
            rul_pred, ageing_pred, fault_logit,
            y_rul, y_ageing, y_fault
        )
        for k in totals:
            totals[k] += metrics[k] * len(X)
        rul_errors.append(
            torch.abs(rul_pred.squeeze() - y_rul).cpu().numpy()
        )
        fault_pred = (torch.sigmoid(fault_logit.squeeze()) > 0.5).long()
        fault_correct += (fault_pred == y_fault.long()).sum().item()
        n += len(X)

    mae_rul  = np.concatenate(rul_errors).mean()
    fault_acc = fault_correct / n
    result = {k: v / n for k, v in totals.items()}
    result["mae_rul_days"] = float(mae_rul)
    result["fault_accuracy"] = float(fault_acc)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # ── 1. Generate synthetic data ────────────────────────────────────────
    print(f"[train] Generating {args.n_transformers} transformer(s) × {args.days} days ...")

    rating = TransformerRating(
        transformer_id="TX_TRAIN",
        rated_power_kva=1000.0,
        rated_current_a=1000.0,
    )
    extractor  = TransformerFeatureExtractor(rating, sample_interval_min=15.0)
    generator  = TransformerDataGenerator(rating, seed=args.seed)
    feature_names = extractor.get_feature_names()

    all_features = []
    for i in range(args.n_transformers):
        # Vary age at start to get diverse RUL distributions
        age_years = (i / max(1, args.n_transformers - 1)) * 20.0
        raw_df = generator.generate(
            n_days=args.days,
            sample_interval_min=15.0,
            age_at_start_years=age_years,
        )
        feat_df = extractor.compute_from_timeseries(raw_df)
        # Bring labels back
        for col in ["rul_days", "ageing_state", "fault_imminent"]:
            feat_df[col] = raw_df[col].values[:len(feat_df)]
        all_features.append(feat_df)
        print(f"  TX {i+1}/{args.n_transformers}: {len(feat_df):,} samples, "
              f"age_start={age_years:.0f}yr, "
              f"fault_rate={feat_df['fault_imminent'].mean():.3f}")

    combined_df = pd.concat(all_features, ignore_index=True)
    print(f"[train] Total samples: {len(combined_df):,}")

    # ── 2. Build datasets ─────────────────────────────────────────────────
    # Fit scaler on full dataset, then split
    full_dataset = TransformerDataset(
        combined_df, feature_names, scaler=None, fit_scaler=True
    )
    scaler = full_dataset.scaler

    # Save scaler for inference service
    import joblib
    joblib.dump(scaler, out_dir / "feature_scaler.pkl")
    print(f"[train] Feature scaler saved → {out_dir}/feature_scaler.pkl")

    n_total = len(full_dataset)
    n_val   = int(n_total * 0.15)
    n_test  = int(n_total * 0.10)
    n_train = n_total - n_val - n_test

    torch.manual_seed(args.seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test]
    )
    print(f"[train] Split: train={n_train:,}  val={n_val:,}  test={n_test:,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ── 3. Model, optimiser, scheduler ───────────────────────────────────
    cfg   = ModelConfig()
    model = TransformerThermalModel(cfg).to(device)
    print(f"[train] Model parameters: {model.count_parameters():,}")

    criterion = TransformerThermalLoss(w_rul=1.0, w_ageing=0.5, w_fault=3.0)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr / 100
    )

    # ── 4. Training loop ──────────────────────────────────────────────────
    history = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt_path = out_dir / "best_model.pt"

    print(f"[train] Starting training: {args.epochs} epochs, "
          f"batch={args.batch_size}, lr={args.lr}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimiser, criterion, device)
        val_metrics   = eval_epoch (model, val_loader,              criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        row = {
            "epoch":        epoch,
            "lr":           lr_now,
            "elapsed_s":    elapsed,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        }
        history.append(row)

        # Checkpoint on improvement
        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            patience_counter = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "cfg":          cfg,
                "val_metrics":  val_metrics,
                "feature_names": feature_names,
            }, best_ckpt_path)
            ckpt_marker = " ✓"
        else:
            patience_counter += 1
            ckpt_marker = ""

        print(
            f"  Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_metrics['loss_total']:.4f}  "
            f"val_loss={val_metrics['loss_total']:.4f}  "
            f"MAE_RUL={val_metrics['mae_rul_days']:.1f}d  "
            f"fault_acc={val_metrics['fault_accuracy']:.3f}  "
            f"lr={lr_now:.2e}  "
            f"[{elapsed:.1f}s]{ckpt_marker}"
        )

        # Early stopping
        if patience_counter >= args.patience:
            print(f"[train] Early stopping triggered at epoch {epoch}")
            break

    # ── 5. Test set evaluation ────────────────────────────────────────────
    print("\n[train] Loading best checkpoint for test evaluation...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = eval_epoch(model, test_loader, criterion, device)
    print(f"[train] Test results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── 6. Save artefacts ─────────────────────────────────────────────────
    history_path = out_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    meta = {
        "model_config":   cfg.__dict__,
        "feature_names":  feature_names,
        "n_features":     len(feature_names),
        "test_metrics":   test_metrics,
        "best_val_loss":  best_val_loss,
        "transformer_rating": {
            "rated_power_kva":  rating.rated_power_kva,
            "rated_current_a":  rating.rated_current_a,
            "delta_theta_to_r": rating.delta_theta_to_r,
            "delta_theta_h_r":  rating.delta_theta_h_r,
        },
    }
    with open(out_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[train] Artefacts saved to {out_dir}/")
    print(f"  best_model.pt          — model weights")
    print(f"  feature_scaler.pkl     — sklearn StandardScaler")
    print(f"  training_history.json  — per-epoch metrics")
    print(f"  model_meta.json        — config + feature names")
    print("\n[train] Run export: python ml/export/export_to_onnx.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TransformerThermalModel for predictive maintenance"
    )
    parser.add_argument("--days",         type=int,   default=730,
                        help="Days of synthetic history per transformer")
    parser.add_argument("--n-transformers", type=int, default=5,
                        help="Number of synthetic transformers to simulate")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=512)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--patience",     type=int,   default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--output-dir",   type=str,   default="ml/data/checkpoints",
                        help="Directory for saved model and artefacts")
    parser.add_argument("--checkpoint",   type=str,   default=None,
                        help="Resume training from checkpoint path")

    args = parser.parse_args()
    main(args)

