#!/usr/bin/env python3
"""
prepare_color_aug_split.py

INPUT:
  dataset/
    good/
    bad/

OUTPUT:
  dataset_prepared/
    train/good, train/bad   (train gets color-aug copies if needed)
    val/good, val/bad       (VAL = originals only, NO augmentation)

Color-only augmentation:
  - brightness/contrast
  - gamma (exposure)
  - saturation
  - hue shift (tiny)
  - white balance (tiny)
  - optional CLAHE on luminance (very mild)

Usage examples (see bottom).
"""

from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
    out = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def imread_bgr(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {p}")
    return img


def save_jpg(path: Path, bgr: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    else:
        cv2.imwrite(str(path), bgr)


# -------------------------
# COLOR AUGMENT (only)
# -------------------------
def aug_color_only(bgr: np.ndarray, rng: np.random.Generator, use_clahe: bool) -> Tuple[np.ndarray, Dict[str, float]]:
    params: Dict[str, float] = {}

    img = bgr.astype(np.float32)

    # brightness/contrast
    alpha = float(rng.uniform(0.85, 1.15))   # contrast
    beta  = float(rng.uniform(-20, 20))      # brightness shift
    img = img * alpha + beta
    params["contrast_alpha"] = alpha
    params["brightness_beta"] = beta

    img = np.clip(img, 0, 255).astype(np.uint8)

    # saturation + hue (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_mul = float(rng.uniform(0.80, 1.25))
    hue_add = float(rng.uniform(-4, 4))      # tiny hue shift (OpenCV hue is 0..179)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_mul, 0, 255)
    hsv[..., 0] = (hsv[..., 0] + hue_add) % 180.0
    params["sat_mul"] = sat_mul
    params["hue_add"] = hue_add
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # gamma (exposure)
    gamma = float(rng.uniform(0.85, 1.20))
    inv = 1.0 / gamma
    table = (np.linspace(0, 1, 256) ** inv) * 255.0
    table = np.clip(table, 0, 255).astype(np.uint8)
    img = cv2.LUT(img, table)
    params["gamma"] = gamma

    # tiny white balance (channel gains)
    b_gain = float(rng.uniform(0.95, 1.05))
    r_gain = float(rng.uniform(0.95, 1.05))
    wb = img.astype(np.float32)
    wb[..., 0] *= b_gain
    wb[..., 2] *= r_gain
    img = np.clip(wb, 0, 255).astype(np.uint8)
    params["b_gain"] = b_gain
    params["r_gain"] = r_gain

    # optional mild CLAHE on luminance (helps lighting variations)
    if use_clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
        params["clahe"] = 1.0
    else:
        params["clahe"] = 0.0

    return img, params


# -------------------------
# MAIN
# -------------------------
def split_indices(n: int, val_ratio: float, rng: np.random.Generator) -> Tuple[List[int], List[int]]:
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_ratio))
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    return train_idx, val_idx


def copy_files(paths: List[Path], out_dir: Path) -> None:
    safe_mkdir(out_dir)
    for p in paths:
        shutil.copy2(str(p), str(out_dir / p.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="dataset folder that contains good/ and bad/")
    ap.add_argument("--output", default="dataset_prepared", help="output folder")
    ap.add_argument("--val-ratio", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)

    # choose how much to expand TRAIN:
    ap.add_argument("--good-target-train", type=int, default=0, help="target #train images for good (0 = no target)")
    ap.add_argument("--bad-target-train", type=int, default=0, help="target #train images for bad  (0 = no target)")
    ap.add_argument("--good-mult-train", type=float, default=0.0, help="multiplier on good TRAIN (0 = ignore)")
    ap.add_argument("--bad-mult-train", type=float, default=0.0, help="multiplier on bad TRAIN  (0 = ignore)")

    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--use-clahe", action="store_true", help="enable mild CLAHE on luminance (still color/lighting)")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    good_dir = inp / "good"
    bad_dir = inp / "bad"
    if not good_dir.exists() or not bad_dir.exists():
        raise SystemExit(f"[ERROR] Expected folders: {good_dir} and {bad_dir}")

    rng = np.random.default_rng(args.seed)

    good_imgs = list_images(good_dir)
    bad_imgs = list_images(bad_dir)

    if len(good_imgs) == 0 or len(bad_imgs) == 0:
        raise SystemExit("[ERROR] No images found in good/ or bad/")

    print("==============================================")
    print("DATASET PREP (split originals + color aug TRAIN)")
    print("==============================================")
    print(f"[INFO] input   : {inp.resolve()}")
    print(f"[INFO] output  : {out.resolve()}")
    print(f"[INFO] counts  : good={len(good_imgs)} bad={len(bad_imgs)}")
    print(f"[INFO] val_ratio={args.val_ratio} seed={args.seed} use_clahe={args.use_clahe}")
    print("==============================================")

    # Split per class
    good_train_idx, good_val_idx = split_indices(len(good_imgs), args.val_ratio, rng)
    bad_train_idx, bad_val_idx = split_indices(len(bad_imgs), args.val_ratio, rng)

    good_train = [good_imgs[i] for i in good_train_idx]
    good_val   = [good_imgs[i] for i in good_val_idx]
    bad_train  = [bad_imgs[i] for i in bad_train_idx]
    bad_val    = [bad_imgs[i] for i in bad_val_idx]

    # Output folders
    train_good_out = out / "train" / "good"
    train_bad_out  = out / "train" / "bad"
    val_good_out   = out / "val" / "good"
    val_bad_out    = out / "val" / "bad"
    for d in [train_good_out, train_bad_out, val_good_out, val_bad_out]:
        safe_mkdir(d)

    # Copy originals
    copy_files(good_train, train_good_out)
    copy_files(bad_train,  train_bad_out)
    copy_files(good_val,   val_good_out)
    copy_files(bad_val,    val_bad_out)

    print(f"[SPLIT] train_good={len(good_train)} train_bad={len(bad_train)} val_good={len(good_val)} val_bad={len(bad_val)}")

    def desired_train_count(cur: int, target: int, mult: float) -> int:
        if target and target > 0:
            return max(cur, int(target))
        if mult and mult > 0:
            return max(cur, int(round(cur * mult)))
        return cur

    # Decide targets
    want_good = desired_train_count(len(good_train), args.good_target_train, args.good_mult_train)
    want_bad  = desired_train_count(len(bad_train),  args.bad_target_train,  args.bad_mult_train)

    # If user didn't specify anything, default behavior: balance GOOD up to BAD
    if args.good_target_train == 0 and args.good_mult_train == 0.0:
        want_good = max(len(good_train), len(bad_train))
    if args.bad_target_train == 0 and args.bad_mult_train == 0.0:
        want_bad = len(bad_train)  # keep bad as-is by default

    print(f"[TARGET] train_good target={want_good} | train_bad target={want_bad}")

    def augment_to_target(src_list: List[Path], out_dir: Path, target_total: int, prefix: str):
        cur = len(src_list)
        need = max(0, target_total - cur)
        if need == 0:
            print(f"[AUG] {prefix}: no augmentation needed.")
            return

        print(f"[AUG] {prefix}: generating {need} augmented images...")
        for k in range(need):
            src = src_list[k % cur]
            img = imread_bgr(src)
            aug, params = aug_color_only(img, rng, use_clahe=args.use_clahe)
            out_name = f"{src.stem}_aug{k:05d}.jpg"
            save_jpg(out_dir / out_name, aug, quality=args.jpeg_quality)

            if (k + 1) % 50 == 0 or (k + 1) == need:
                print(f"  {prefix}: {k+1}/{need}  last={out_name}  params={params}")

    augment_to_target(good_train, train_good_out, want_good, "GOOD")
    augment_to_target(bad_train,  train_bad_out,  want_bad,  "BAD")

    # Final counts
    final_train_good = len(list_images(train_good_out))
    final_train_bad  = len(list_images(train_bad_out))
    final_val_good   = len(list_images(val_good_out))
    final_val_bad    = len(list_images(val_bad_out))

    print("==============================================")
    print("[DONE]")
    print(f"TRAIN: good={final_train_good} bad={final_train_bad}")
    print(f"VAL  : good={final_val_good} bad={final_val_bad}  (NO AUG)")
    print(f"Output folder: {out.resolve()}")
    print("==============================================")


if __name__ == "__main__":
    main()