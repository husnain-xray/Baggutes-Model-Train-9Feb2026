#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


# ----------------------------
# Display helpers
# ----------------------------
def fit_to_screen(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Resize an image to fit within max_w x max_h (keeps aspect). Never upscales."""
    h, w = img.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return img


# ----------------------------
# ROI helpers
# ----------------------------
def _as_int(v) -> int:
    return int(round(float(v)))


def load_roi_any(roi_path: Path) -> Tuple[int, int, int, int]:
    data = json.loads(roi_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        cand = data
        for k in ("roi", "ROI", "tray_roi", "rect", "bbox"):
            if k in data and isinstance(data[k], dict):
                cand = data[k]
                break

        keys = set(cand.keys())
        if {"x1", "y1", "x2", "y2"}.issubset(keys):
            x1, y1, x2, y2 = map(_as_int, (cand["x1"], cand["y1"], cand["x2"], cand["y2"]))
            return x1, y1, x2, y2

        if {"x", "y", "w", "h"}.issubset(keys):
            x = _as_int(cand["x"])
            y = _as_int(cand["y"])
            w = _as_int(cand["w"])
            h = _as_int(cand["h"])
            return x, y, x + w, y + h

    raise ValueError("Unsupported ROI JSON format")


def clamp_roi(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


# ----------------------------
# Model
# ----------------------------
def build_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    raise ValueError(f"Unsupported --arch: {arch}")


def load_checkpoint(pth_path: Path, model: nn.Module) -> dict:
    ckpt = torch.load(str(pth_path), map_location="cpu")

    # Your trainer saves weights under "model_state"
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        meta = ckpt
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        meta = ckpt
    else:
        state = ckpt
        meta = ckpt if isinstance(ckpt, dict) else {}

    # strip "module." if any
    new_state = {}
    for k, v in state.items():
        new_state[k[7:]] = v if k.startswith("module.") else v
        if not k.startswith("module."):
            new_state[k] = v

    # load strict (you want to detect mismatch early)
    model.load_state_dict(state, strict=True)
    return meta


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    s = float(np.sum(e))
    return e / (s if s > 0 else 1.0)


def set_cpu_threads(cpu_pct: int, interop: int):
    cpu_count = os.cpu_count() or 1
    cpu_pct = max(1, min(100, int(cpu_pct)))
    threads = max(1, int(round(cpu_count * (cpu_pct / 100.0))))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, int(interop)))
    torch.backends.mkldnn.enabled = True
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    return cpu_count, threads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--pth", required=True)
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--num-classes", type=int, default=2)

    ap.add_argument("--good-index", type=int, default=-1, help="If -1: auto from checkpoint class_to_idx if possible")
    ap.add_argument("--good-name", type=str, default="good")

    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--min-gap-sec", type=float, default=1.0)

    ap.add_argument("--cpu-pct", type=int, default=100)
    ap.add_argument("--interop", type=int, default=2)

    ap.add_argument("--save-dir", type=str, default="")
    ap.add_argument("--save-full-frame", action="store_true")
    ap.add_argument("--no-gui", action="store_true")

    # screen fit
    ap.add_argument("--max-w", type=int, default=1280)
    ap.add_argument("--max-h", type=int, default=720)
    ap.add_argument("--window", type=str, default="PTH ROI Preview")

    args = ap.parse_args()

    video_path = Path(args.video)
    roi_path = Path(args.roi)
    pth_path = Path(args.pth)

    if not video_path.exists():
        raise SystemExit(f"[ERROR] video not found: {video_path}")
    if not roi_path.exists():
        raise SystemExit(f"[ERROR] roi not found: {roi_path}")
    if not pth_path.exists():
        raise SystemExit(f"[ERROR] pth not found: {pth_path}")

    cpu_count, threads = set_cpu_threads(args.cpu_pct, args.interop)
    print(f"[CPU] cpu_count={cpu_count} threads={threads} interop={args.interop} mkldnn={torch.backends.mkldnn.enabled}")

    x1, y1, x2, y2 = load_roi_any(roi_path)

    # IMPORTANT: must match training normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    model = build_model(args.arch, args.num_classes)
    meta = load_checkpoint(pth_path, model)
    model.eval()

    # good index auto
    good_index = args.good_index
    if good_index == -1 and isinstance(meta, dict):
        c2i = meta.get("class_to_idx", None)
        if isinstance(c2i, dict) and args.good_name in c2i:
            good_index = int(c2i[args.good_name])
        else:
            good_index = 1  # fallback: bad=0 good=1
    bad_index = 0 if good_index != 0 else 1

    print(f"[INFO] good_index={good_index} bad_index={bad_index} class_to_idx={meta.get('class_to_idx', None) if isinstance(meta, dict) else None}")
    print("KEYS: space=pause | q/esc=quit | +/- threshold | s toggle saving")

    save_enabled = bool(args.save_dir)
    save_dir = Path(args.save_dir) if save_enabled else None
    if save_enabled:
        save_dir.mkdir(parents=True, exist_ok=True)

    out_csv = (save_dir / "video_results.csv") if save_enabled else Path("video_results.csv")
    csv_f = out_csv.open("w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["time_sec", "frame_idx", "pred", "pg", "pb", "saved_path"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit("[ERROR] cannot open video")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30.0
    step_frames = max(1, int(round(video_fps / max(0.01, args.fps))))

    paused = False
    last_save_t = -1e9
    frame_idx = 0
    last_frame = None
    do_save = save_enabled

    if not args.no_gui:
        cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            last_frame = frame
            frame_idx += 1
        else:
            if last_frame is None:
                ok, frame = cap.read()
                if not ok:
                    break
                last_frame = frame
            frame = last_frame

        if (frame_idx % step_frames) != 0 and not paused:
            continue

        time_sec = frame_idx / video_fps
        fh, fw = frame.shape[:2]
        rx1, ry1, rx2, ry2 = clamp_roi(x1, y1, x2, y2, fw, fh)
        roi_bgr = frame[ry1:ry2, rx1:rx2].copy()

        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi_rgb = cv2.resize(roi_rgb, (args.img, args.img), interpolation=cv2.INTER_AREA)
        x = tfm(roi_rgb).unsqueeze(0)

        with torch.inference_mode():
            logits = model(x).cpu().numpy().reshape(-1)

        probs = softmax_np(logits)
        pg = float(probs[good_index]) if good_index < len(probs) else float("nan")
        pb = float(probs[bad_index]) if bad_index < len(probs) else float("nan")
        pred = int(np.argmax(probs)) if len(probs) else -1

        is_good = (pred == good_index) and (pg >= args.threshold)

        saved_path = ""
        if do_save and save_enabled and is_good and (time_sec - last_save_t) >= args.min_gap_sec:
            name = f"GOOD_t{time_sec:08.2f}_f{frame_idx:07d}_pg{pg:.3f}.jpg"
            out_path = save_dir / name
            if args.save_full_frame:
                tmp = frame.copy()
                cv2.rectangle(tmp, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
                cv2.putText(tmp, f"GOOD pg={pg:.3f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imwrite(str(out_path), tmp)
            else:
                cv2.imwrite(str(out_path), roi_bgr)
            saved_path = str(out_path)
            last_save_t = time_sec

        csv_w.writerow([f"{time_sec:.3f}", frame_idx, pred, f"{pg:.4f}", f"{pb:.4f}", saved_path])

        if args.no_gui:
            print(f"t={time_sec:7.2f}s f={frame_idx:7d} pred={'GOOD' if pred==good_index else 'BAD '} pg={pg:.3f} pb={pb:.3f} save={'Y' if saved_path else 'N'}")
            continue

        overlay = frame.copy()
        color = (0, 255, 0) if is_good else (0, 0, 255)
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), color, 3)

        label = "GOOD" if pred == good_index else "BAD"
        cv2.putText(
            overlay,
            f"t={time_sec:.2f}s f={frame_idx} pred={label} pg={pg:.3f} thr={args.threshold:.2f} save={'Y' if saved_path else 'N'}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

        # small ROI inset (size adapts)
        inset_w = min(360, max(200, overlay.shape[1] // 5))
        inset_h = int(inset_w * 9 / 16)
        inset = cv2.resize(roi_bgr, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
        y0, x0 = 60, 20
        if y0 + inset_h < overlay.shape[0] and x0 + inset_w < overlay.shape[1]:
            overlay[y0:y0+inset_h, x0:x0+inset_w] = inset
            cv2.rectangle(overlay, (x0, y0), (x0+inset_w, y0+inset_h), color, 2)

        # FIT TO SCREEN HERE
        show = fit_to_screen(overlay, args.max_w, args.max_h)

        cv2.imshow(args.window, show)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        if key in (ord("+"), ord("=")):
            args.threshold = min(0.99, args.threshold + 0.02)
        if key in (ord("-"), ord("_")):
            args.threshold = max(0.01, args.threshold - 0.02)
        if key in (ord("s"), ord("S")):
            do_save = not do_save

    csv_f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] CSV -> {out_csv}")


if __name__ == "__main__":
    main()