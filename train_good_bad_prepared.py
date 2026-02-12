#!/usr/bin/env python3
"""
train_good_bad_prepared.py

Trains a 2-class classifier (bad/good) using an already prepared dataset:

dataset_prepared/
  train/
    bad/
    good/
  val/
    bad/
    good/

Key points:
- NO leakage: train and val are separate folders
- Train uses augmentation (flip + color jitter)
- Val uses deterministic transform only
- Uses ImageNet normalization (IMPORTANT if using pretrained ResNet/MobileNet)
- Saves: last.pth every epoch, best.pth when val acc improves
- Writes: meta.json, metrics.csv, train.log
"""

from __future__ import annotations

import os
import time
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_model(model_name: str, num_classes: int = 2):
    model_name = model_name.lower()
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unknown model_name: {model_name}")


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"[LOG] Writing logs to: {log_path}")
    return logger


def fmt_hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def configure_cpu(logger: logging.Logger, cpu_pct: int, interop_threads: int):
    cpu_count = os.cpu_count() or 1
    cpu_pct = max(1, min(100, int(cpu_pct)))
    threads = max(1, int(round(cpu_count * (cpu_pct / 100.0))))

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, int(interop_threads)))
    torch.backends.mkldnn.enabled = True

    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)

    logger.info(
        f"[CPU] cpu_count={cpu_count} | cpu_pct={cpu_pct}% | torch_threads={threads} | "
        f"interop_threads={interop_threads} | mkldnn={torch.backends.mkldnn.enabled}"
    )


def save_checkpoint(path: Path, model, optimizer, epoch, best_acc, class_to_idx):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_acc": best_acc,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "class_to_idx": class_to_idx,
        },
        str(path),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset_prepared", help="Folder with train/ and val/")
    ap.add_argument("--out", type=str, default="runs_good_bad", help="Output folder")
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)      # keep realistic for CPU
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)      # Windows safe
    ap.add_argument("--cpu-pct", type=int, default=90)
    ap.add_argument("--interop", type=int, default=2)
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint .pth to resume")
    args = ap.parse_args()

    data_root = Path(args.data)
    train_root = data_root / "train"
    val_root = data_root / "val"
    if not train_root.exists() or not val_root.exists():
        raise SystemExit(f"[ERROR] Expected: {train_root} and {val_root}")

    out_dir = Path(args.out)
    logger = setup_logging(out_dir)

    device = torch.device("cpu")
    configure_cpu(logger, cpu_pct=args.cpu_pct, interop_threads=args.interop)

    tfm_train = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(root=str(train_root), transform=tfm_train)
    val_ds = datasets.ImageFolder(root=str(val_root), transform=tfm_val)

    class_names = train_ds.classes
    class_to_idx = train_ds.class_to_idx

    # quick sanity checks
    if train_ds.class_to_idx != val_ds.class_to_idx:
        logger.info(f"[WARN] train class_to_idx={train_ds.class_to_idx}")
        logger.info(f"[WARN] val   class_to_idx={val_ds.class_to_idx}")
        raise SystemExit("[ERROR] Class mapping mismatch between train and val folders.")

    logger.info(f"class_to_idx={class_to_idx} | classes={class_names}")
    logger.info(f"[DATA] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = build_model(args.model, num_classes=2).to(device)
    model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_acc = 0.0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_acc = float(ckpt.get("best_acc", 0.0))
        logger.info(f"[RESUME] {args.resume} start_epoch={start_epoch} best_acc={best_acc:.4f}")

    (out_dir / "meta.json").write_text(json.dumps({
        "classes": class_names,
        "class_to_idx": class_to_idx,
        "model": args.model,
        "img": args.img,
        "batch": args.batch,
        "lr": args.lr,
        "epochs": args.epochs,
        "device": "cpu",
        "workers": args.workers,
        "cpu_pct": args.cpu_pct,
        "interop": args.interop,
        "normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    }, indent=2), encoding="utf-8")

    metrics_path = out_dir / "metrics.csv"
    if not metrics_path.exists() or start_epoch == 0:
        metrics_path.write_text("epoch,train_loss,val_loss,val_acc,epoch_time_sec,eta_sec\n", encoding="utf-8")

    epoch_times = []
    total_epochs = args.epochs
    global_start = time.time()

    for epoch in range(start_epoch, total_epochs):
        epoch_t0 = time.time()
        epoch_idx = epoch + 1

        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train E{epoch_idx}/{total_epochs}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device, memory_format=torch.channels_last)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / max(1, len(train_loader))

        # ---- VAL ----
        model.eval()
        y_true, y_pred = [], []
        val_loss = 0.0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Val   E{epoch_idx}/{total_epochs}", leave=False)
            for xb, yb in vbar:
                xb = xb.to(device, memory_format=torch.channels_last)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.item())
                pred = torch.argmax(logits, dim=1)
                y_true.extend(yb.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())

        avg_val_loss = val_loss / max(1, len(val_loader))
        acc = accuracy_score(y_true, y_pred)

        # timing + ETA
        epoch_time = time.time() - epoch_t0
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = total_epochs - epoch_idx
        eta_sec = avg_epoch_time * remaining_epochs

        logger.info(
            f"[EPOCH {epoch_idx}/{total_epochs}] "
            f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | val_acc={acc:.4f} "
            f"| epoch_time={fmt_hms(epoch_time)} | ETA={fmt_hms(eta_sec)}"
        )

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch_idx},{avg_train_loss:.6f},{avg_val_loss:.6f},{acc:.6f},{epoch_time:.3f},{eta_sec:.3f}\n")

        # save last + best
        save_checkpoint(out_dir / "last.pth", model, optimizer, epoch, best_acc, class_to_idx)
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(out_dir / "best.pth", model, optimizer, epoch, best_acc, class_to_idx)
            logger.info(f"[BEST] best_acc={best_acc:.4f} -> saved best.pth")

        # report every 5 epochs
        if (epoch_idx % 5 == 0) or (epoch_idx == total_epochs):
            cm = confusion_matrix(y_true, y_pred)
            rep = classification_report(y_true, y_pred, target_names=class_names, digits=4)
            logger.info("[CONFUSION_MATRIX]\n" + str(cm))
            logger.info("[CLASSIFICATION_REPORT]\n" + rep)

    total_time = time.time() - global_start
    logger.info(f"[DONE] best_acc={best_acc:.4f} | total_time={fmt_hms(total_time)} | out={out_dir}")


if __name__ == "__main__":
    main()