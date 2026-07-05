#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a pure-RGB condition predictor for DreamNav.

Input:  source RGB image + target/generated RGB image, concatenated as 6 channels.
Output: [sin(heading), cos(heading), range / range_scale]

This model is intended to be frozen later and used as a high-level pose/condition
consistency loss for Step2 generation.
"""

import argparse
import json
import math
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Utilities
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def angle_to_sincos(deg: float) -> Tuple[float, float]:
    rad = math.radians(float(deg))
    return math.sin(rad), math.cos(rad)


def sincos_to_angle_deg(s: np.ndarray, c: np.ndarray) -> np.ndarray:
    # Normalize to avoid angle instability if predicted vector is not unit length.
    norm = np.sqrt(s * s + c * c) + 1e-8
    s = s / norm
    c = c / norm
    return np.degrees(np.arctan2(s, c))


def circular_abs_error_deg(pred_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    diff = (pred_deg - true_deg + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def resolve_image_path(root_dir: Path, image_value: str) -> Path:
    """Resolve image path robustly for pairUAV-style JSON fields."""
    p = Path(image_value)
    if p.is_absolute() and p.exists():
        return p

    candidates = [
        root_dir / image_value,
        root_dir / "tours" / image_value,
        root_dir.parent / image_value,
        root_dir.parent / "tours" / image_value,
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    # Return most likely path; Dataset will raise useful error if missing.
    return root_dir / "tours" / image_value


def read_pair_record(json_path: Path, root_dir: Path) -> Optional[Tuple[str, str, float, float, str]]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None

    # Support current pairUAV jsons and possible step1-json-like fields.
    image_a = d.get("image_a", d.get("source", d.get("source_image", None)))
    image_b = d.get("image_b", d.get("target", d.get("target_image", None)))

    if image_a is None or image_b is None:
        return None

    # Prefer true labels for predictor training.
    if "heading_num" in d:
        heading = d["heading_num"]
    elif "true_deg_num" in d:
        heading = d["true_deg_num"]
    else:
        return None

    if "range_num" in d:
        range_num = d["range_num"]
    elif "true_rag_num" in d:
        range_num = d["true_rag_num"]
    else:
        return None

    src = resolve_image_path(root_dir, str(image_a))
    tgt = resolve_image_path(root_dir, str(image_b))
    return str(src), str(tgt), float(heading), float(range_num), str(json_path)


def build_or_load_records(root_dir: Path, dataset_type: str, cache_path: Path, refresh_cache: bool, max_scan_records: int = -1) -> List[Tuple[str, str, float, float, str]]:
    if cache_path.exists() and not refresh_cache:
        print(f"[Cache] Loading records from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    data_dir = root_dir / dataset_type
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    json_files = sorted(data_dir.rglob("*.json"))
    print(f"[Scan] Found {len(json_files)} json files under {data_dir}")

    records = []
    t0 = time.time()
    for i, jp in enumerate(json_files):
        rec = read_pair_record(jp, root_dir)
        if rec is not None:
            records.append(rec)
        if (i + 1) % 100000 == 0:
            print(f"[Scan] {i+1}/{len(json_files)} jsons, valid={len(records)}, elapsed={time.time()-t0:.1f}s")
        if max_scan_records > 0 and len(records) >= max_scan_records:
            print(f"[Scan] Early stop: collected {len(records)} valid records, max_scan_records={max_scan_records}")
            break

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(records, f)
    print(f"[Cache] Saved {len(records)} records to {cache_path}")
    return records


# -----------------------------
# Dataset
# -----------------------------

class RGBPairConditionDataset(Dataset):
    def __init__(self, records: List[Tuple[str, str, float, float, str]], image_size: int, range_scale: float):
        self.records = records
        self.image_size = int(image_size)
        self.range_scale = float(range_scale)

    def __len__(self) -> int:
        return len(self.records)

    def _load_rgb(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 255.0
        # Use roughly [-1, 1], matching generative model convention.
        arr = arr * 2.0 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src_path, tgt_path, heading_deg, range_num, json_path = self.records[idx]
        src = self._load_rgb(src_path)
        tgt = self._load_rgb(tgt_path)
        x = torch.cat([src, tgt], dim=0)  # [6, H, W]

        s, c = angle_to_sincos(heading_deg)
        y = torch.tensor([s, c, range_num / self.range_scale], dtype=torch.float32)

        return {
            "x": x,
            "y": y,
            "heading_deg": torch.tensor(float(heading_deg), dtype=torch.float32),
            "range_num": torch.tensor(float(range_num), dtype=torch.float32),
        }


# -----------------------------
# Models
# -----------------------------

class TinyPoseCNN(nn.Module):
    def __init__(self, in_ch: int = 6, out_dim: int = 3):
        super().__init__()
        def block(cin, cout, stride=2):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.SiLU(inplace=True),
                nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.SiLU(inplace=True),
            )
        self.net = nn.Sequential(
            block(in_ch, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.head(self.net(x))


def make_model(name: str) -> nn.Module:
    name = name.lower()
    if name == "tiny":
        return TinyPoseCNN(6, 3)

    if name == "resnet18":
        try:
            from torchvision import models
            try:
                m = models.resnet18(weights=None)
            except TypeError:
                m = models.resnet18(pretrained=False)
            old_conv = m.conv1
            m.conv1 = nn.Conv2d(6, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                stride=old_conv.stride, padding=old_conv.padding, bias=False)
            m.fc = nn.Linear(m.fc.in_features, 3)
            return m
        except Exception as e:
            print(f"[Warn] torchvision resnet18 unavailable: {e}")
            print("[Warn] Falling back to TinyPoseCNN")
            return TinyPoseCNN(6, 3)

    raise ValueError(f"Unknown model: {name}")


# -----------------------------
# Training / evaluation
# -----------------------------

def compute_loss(pred: torch.Tensor, y: torch.Tensor, range_weight: float, norm_weight: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    angle_loss = F.mse_loss(pred[:, :2], y[:, :2])
    range_loss = F.smooth_l1_loss(pred[:, 2], y[:, 2])
    pred_norm = torch.sqrt(torch.sum(pred[:, :2] ** 2, dim=1) + 1e-8)
    norm_loss = F.mse_loss(pred_norm, torch.ones_like(pred_norm))
    loss = angle_loss + range_weight * range_loss + norm_weight * norm_loss
    return loss, {
        "angle_loss": float(angle_loss.detach().cpu()),
        "range_loss": float(range_loss.detach().cpu()),
        "norm_loss": float(norm_loss.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, range_scale: float, max_batches: int = -1) -> Dict[str, float]:
    model.eval()
    all_pred = []
    all_y = []
    all_h = []
    all_r = []

    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        pred = model(x).detach().cpu().numpy()
        all_pred.append(pred)
        all_y.append(batch["y"].numpy())
        all_h.append(batch["heading_deg"].numpy())
        all_r.append(batch["range_num"].numpy())

    pred = np.concatenate(all_pred, axis=0)
    y = np.concatenate(all_y, axis=0)
    true_h = np.concatenate(all_h, axis=0)
    true_r = np.concatenate(all_r, axis=0)

    pred_h = sincos_to_angle_deg(pred[:, 0], pred[:, 1])
    h_err = circular_abs_error_deg(pred_h, true_h)

    pred_r = pred[:, 2] * range_scale
    r_err = np.abs(pred_r - true_r)

    pred_norm = np.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2)

    return {
        "n": int(len(h_err)),
        "heading_mae": float(h_err.mean()),
        "heading_median": float(np.median(h_err)),
        "heading_p90": float(np.percentile(h_err, 90)),
        "heading_max": float(h_err.max()),
        "range_mae": float(r_err.mean()),
        "range_median": float(np.median(r_err)),
        "range_p90": float(np.percentile(r_err, 90)),
        "range_max": float(r_err.max()),
        "pred_sincos_norm_mean": float(pred_norm.mean()),
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer, epoch: int, global_step: int, args, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "metrics": metrics,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="../pairUAV/")
    parser.add_argument("--dataset_type", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="./outputs_rgb_condition_predictor_R1")
    parser.add_argument("--cache_path", type=str, default="")
    parser.add_argument("--refresh_cache", action="store_true")

    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "tiny"])
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--range_scale", type=float, default=100.0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--range_weight", type=float, default=1.0)
    parser.add_argument("--norm_weight", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_val_samples", type=int, default=20000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every_epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    seed_everything(args.seed)
    root_dir = Path(args.root_dir).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_path:
        cache_path = Path(args.cache_path)
    else:
        cache_path = out_dir / f"cache_{args.dataset_type}.pkl"

    max_scan_records = -1
    if args.max_train_samples > 0:
        max_scan_records = args.max_train_samples + max(args.max_val_samples, 1000)
    records = build_or_load_records(root_dir, args.dataset_type, cache_path, args.refresh_cache, max_scan_records=max_scan_records)
    if len(records) == 0:
        raise RuntimeError("No valid records found.")

    rng = random.Random(args.seed)
    rng.shuffle(records)

    if args.max_train_samples > 0:
        # Keep a little extra for validation.
        max_total = min(len(records), args.max_train_samples + max(args.max_val_samples, 1000))
        records = records[:max_total]

    n_val = max(1, int(len(records) * args.val_ratio))
    if args.max_val_samples > 0:
        n_val = min(n_val, args.max_val_samples)
    val_records = records[:n_val]
    train_records = records[n_val:]
    if args.max_train_samples > 0:
        train_records = train_records[:args.max_train_samples]

    print(f"[Data] train={len(train_records)}, val={len(val_records)}")
    print(f"[Data] root={root_dir}, dataset_type={args.dataset_type}, image_size={args.image_size}")

    train_ds = RGBPairConditionDataset(train_records, args.image_size, args.range_scale)
    val_ds = RGBPairConditionDataset(val_records, args.image_size, args.range_scale)

    pin = torch.cuda.is_available() and args.device.startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              persistent_workers=args.num_workers > 0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(1, min(args.num_workers, 8)), pin_memory=pin,
                            persistent_workers=args.num_workers > 0, drop_last=False)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = make_model(args.model).to(device)
    print(f"[Model] {args.model}, params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_score = float("inf")
    global_step = 0

    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = []
        for bi, batch in enumerate(train_loader):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                pred = model(x)
                loss, parts = compute_loss(pred, y, args.range_weight, args.norm_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            running.append(float(loss.detach().cpu()))

            if global_step % args.log_every == 0:
                print(
                    f"[Train] ep={epoch+1}/{args.epochs} step={global_step} "
                    f"batch={bi+1}/{len(train_loader)} loss={np.mean(running[-args.log_every:]):.5f} "
                    f"angle={parts['angle_loss']:.5f} range={parts['range_loss']:.5f} norm={parts['norm_loss']:.5f} "
                    f"time={time.time()-t0:.1f}s",
                    flush=True,
                )

        metrics = {}
        if (epoch + 1) % args.eval_every_epoch == 0:
            metrics = evaluate(model, val_loader, device, args.range_scale)
            print(
                f"[Val] epoch={epoch+1} n={metrics['n']} "
                f"heading_mae={metrics['heading_mae']:.3f} median={metrics['heading_median']:.3f} "
                f"p90={metrics['heading_p90']:.3f} max={metrics['heading_max']:.3f} | "
                f"range_mae={metrics['range_mae']:.3f} median={metrics['range_median']:.3f} "
                f"p90={metrics['range_p90']:.3f} max={metrics['range_max']:.3f} | "
                f"norm={metrics['pred_sincos_norm_mean']:.3f}",
                flush=True,
            )

            # Use a combined score; heading is primary, range is secondary.
            score = metrics["heading_mae"] + 0.2 * metrics["range_mae"]
            if score < best_score:
                best_score = score
                save_checkpoint(out_dir / "best.pt", model, optimizer, epoch + 1, global_step, args, metrics)
                print(f"[Save] best.pt score={score:.3f}")

        save_checkpoint(out_dir / f"epoch_{epoch+1:03d}.pt", model, optimizer, epoch + 1, global_step, args, metrics)
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch + 1, global_step, args, metrics)
        print(f"[Save] epoch_{epoch+1:03d}.pt and last.pt")

    print("[Done]")


if __name__ == "__main__":
    main()
