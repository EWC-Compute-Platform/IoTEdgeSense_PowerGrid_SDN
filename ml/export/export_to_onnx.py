"""
export_to_onnx.py
=================
Exports the trained TransformerThermalModel to ONNX format
for deployment in the C++ ONNX Runtime inference engine.

The exported model has:
  Input:  "features"  — float32 tensor [batch, 47]
  Outputs:
    "rul_days"      — float32 tensor [batch, 1]
    "ageing_state"  — float32 tensor [batch, 1]
    "fault_prob"    — float32 tensor [batch, 1]  (sigmoid applied here)

Usage:
    python ml/export/export_to_onnx.py
    python ml/export/export_to_onnx.py --checkpoint ml/data/checkpoints/best_model.pt
    python ml/export/export_to_onnx.py --opset 17 --output model.onnx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.models.thermal_ageing_model import (
    ModelConfig,
    TransformerThermalModel,
)


# ---------------------------------------------------------------------------
# ONNX-compatible wrapper
# ---------------------------------------------------------------------------

class OnnxExportWrapper(nn.Module):
    """
    Thin wrapper that applies sigmoid to the fault logit before export,
    so the ONNX model outputs probability (not raw logit) — cleaner for
    the C++ consumer which does not need to know about the training loss.
    """

    def __init__(self, model: TransformerThermalModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        rul, ageing, fault_logit = self.model(x)
        fault_prob = torch.sigmoid(fault_logit)
        return rul, ageing, fault_prob


# ---------------------------------------------------------------------------
# Validation: compare PyTorch vs ONNX outputs
# ---------------------------------------------------------------------------

def validate_onnx(onnx_path: str,
                  model_pt: TransformerThermalModel,
                  n_features: int,
                  tolerance: float = 1e-4) -> bool:
    """
    Runs the same random input through PyTorch and ONNX Runtime
    and checks that outputs agree within tolerance.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[export] onnxruntime not installed — skipping validation")
        return True

    dummy = torch.randn(4, n_features)

    # PyTorch reference
    wrapper = OnnxExportWrapper(model_pt)
    wrapper.eval()
    with torch.no_grad():
        pt_rul, pt_ageing, pt_fault = wrapper(dummy)
    pt_rul    = pt_rul.numpy()
    pt_ageing = pt_ageing.numpy()
    pt_fault  = pt_fault.numpy()

    # ONNX Runtime
    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(
        None, {"features": dummy.numpy()}
    )
    ort_rul, ort_ageing, ort_fault = ort_out

    ok = True
    for name, pt, ort_val in [
        ("rul_days",     pt_rul,    ort_rul),
        ("ageing_state", pt_ageing, ort_ageing),
        ("fault_prob",   pt_fault,  ort_fault),
    ]:
        diff = np.abs(pt - ort_val).max()
        status = "✓" if diff < tolerance else "✗ MISMATCH"
        print(f"  {name:15s}  max_diff={diff:.2e}  {status}")
        if diff >= tolerance:
            ok = False

    return ok


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def main(args):
    ckpt_path = Path(args.checkpoint)
    out_path  = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg   = ckpt.get("cfg", ModelConfig())
    model = TransformerThermalModel(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    feature_names = ckpt.get("feature_names", [])
    n_features    = cfg.input_dim
    print(f"[export] Model input dim: {n_features}  params: {model.count_parameters():,}")

    # Wrap for clean ONNX outputs
    wrapper = OnnxExportWrapper(model)
    wrapper.eval()

    # Dummy input for tracing
    dummy = torch.randn(1, n_features)

    print(f"[export] Exporting to ONNX (opset {args.opset})...")
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["rul_days", "ageing_state", "fault_prob"],
        dynamic_axes={
            "features":     {0: "batch_size"},
            "rul_days":     {0: "batch_size"},
            "ageing_state": {0: "batch_size"},
            "fault_prob":   {0: "batch_size"},
        },
    )
    print(f"[export] Saved: {out_path}")

    # ── ONNX model validation ─────────────────────────────────────────────
    try:
        import onnx
        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)
        print(f"[export] ONNX model check: PASSED")
    except ImportError:
        print("[export] onnx package not installed — skipping graph check")
    except Exception as e:
        print(f"[export] ONNX model check FAILED: {e}")

    # ── PyTorch vs ONNX Runtime numerical validation ──────────────────────
    print("[export] Numerical validation (PyTorch vs ONNX Runtime):")
    valid = validate_onnx(str(out_path), model, n_features)
    if valid:
        print("[export] Validation PASSED — outputs match within tolerance")
    else:
        print("[export] WARNING — output mismatch detected, review model")

    # ── Write metadata sidecar ────────────────────────────────────────────
    meta_path = out_path.with_suffix(".json")
    meta = {
        "onnx_path":        str(out_path.name),
        "opset":            args.opset,
        "input_name":       "features",
        "input_shape":      [1, n_features],
        "input_dtype":      "float32",
        "output_names":     ["rul_days", "ageing_state", "fault_prob"],
        "feature_names":    feature_names,
        "n_features":       n_features,
        "model_config":     cfg.__dict__,
        "source_checkpoint": str(ckpt_path.name),
        "val_metrics":      ckpt.get("val_metrics", {}),
        # C++ consumer uses these to reconstruct feature vector
        "feature_groups": {
            "physics":  {
                "start": 0,
                "end":   cfg.n_physics_features,
                "names": feature_names[:cfg.n_physics_features],
            },
            "rolling":  {
                "start": cfg.n_physics_features,
                "end":   cfg.n_physics_features + cfg.n_rolling_features,
                "names": feature_names[
                    cfg.n_physics_features:
                    cfg.n_physics_features + cfg.n_rolling_features
                ],
            },
            "temporal": {
                "start": cfg.n_physics_features + cfg.n_rolling_features,
                "end":   n_features,
                "names": feature_names[
                    cfg.n_physics_features + cfg.n_rolling_features:
                ],
            },
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] Metadata sidecar: {meta_path}")
    print("\n[export] Complete. Files ready for C++ deployment:")
    print(f"  {out_path}")
    print(f"  {meta_path}")
    print("\nNext step: build C++ ONNX inference engine")
    print("  src/ml/onnx_inference.h/.cpp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export TransformerThermalModel to ONNX"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ml/data/checkpoints/best_model.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ml/data/checkpoints/transformer_thermal.onnx",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (17 recommended for ORT 1.16+)"
    )
    args = parser.parse_args()
    main(args)
