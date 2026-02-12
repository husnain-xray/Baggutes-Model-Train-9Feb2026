#!/usr/bin/env python3
"""
convert_pth_to_onnx.py

Exports the trained checkpoint from train_good_bad_prepared.py to ONNX.

Run:
python convert_pth_to_onnx.py `
  --checkpoint "runs_good_bad_prepared\\best.pth" `
  --output "runs_good_bad_prepared\\best_opset18.onnx" `
  --model resnet18 `
  --num-classes 2 `
  --input-size 224 `
  --opset 18 `
  --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def build_model(model_name: str, num_classes: int):
    import torch.nn as nn
    from torchvision import models

    name = model_name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m

    raise ValueError(f"Unsupported model: {model_name}")


def extract_state_dict(ckpt):
    # Your trainer saves weights under "model_state"
    if isinstance(ckpt, dict):
        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            return ckpt["model_state"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]

        # Sometimes ckpt itself is a pure state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt

    # Rare: ckpt IS the state_dict directly
    if isinstance(ckpt, dict) is False:
        return ckpt

    raise ValueError("Could not find model weights in checkpoint (expected model_state/state_dict).")


def strip_module_prefix_if_needed(state: dict) -> dict:
    # ✅ ONLY strip when key starts with "module."
    if not isinstance(state, dict):
        return state
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True, choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] checkpoint not found: {ckpt_path}")
        sys.exit(2)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    print("[INFO] Building model...")
    model = build_model(args.model, args.num_classes).to(device).eval()

    print("[INFO] Loading checkpoint...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = extract_state_dict(ckpt)
    state = strip_module_prefix_if_needed(state)

    # strict=True because you want exact match
    model.load_state_dict(state, strict=True)

    dummy = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device, dtype=torch.float32)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Exporting ONNX -> {out_path} (opset={args.opset}, device={device.type})")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
        verbose=args.verbose,
    )

    print("[OK] Export completed:", out_path.resolve())


if __name__ == "__main__":
    main()