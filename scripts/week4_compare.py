"""
week4_compare.py
Evaluates all 5 models on the test set and saves comparison results.

Experiments:
  A — RGB Baseline
  B — Pose Only
  C — Late Fusion
  D — Feature Concat Fusion
  E — Adaptive Fusion (Week 4)

Saves:
  results/week4_comparison.csv
  results/week4_comparison.png
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import csv
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from models.rgb_branch    import RGBBaselineModel
from models.pose_branch   import PoseBranchModel
from models.fusion        import LateFusionModel, FeatureConcatFusionModel
from models.adaptive_model import AdaptiveMultiModalModel

from scripts.train        import VideoDataset
from scripts.train_pose   import PoseDataset
from scripts.train_fusion import FusionDataset

# ── Config ─────────────────────────────────────────────────────────────────────
with open(Path(ROOT) / "configs" / "baseline.yaml") as f:
    cfg = yaml.safe_load(f)

NUM_FRAMES  = cfg["data"]["num_frames"]
NUM_CLASSES = cfg["data"]["num_classes"]
BATCH_SIZE  = cfg["training"]["batch_size"]
CKPT_DIR    = Path(ROOT) / cfg["paths"]["checkpoint_dir"]
RESULTS_DIR = Path(ROOT) / cfg["paths"]["results_dir"]

TRAIN_CSV = Path(ROOT) / cfg["data"]["train_csv"]
VAL_CSV   = Path(ROOT) / cfg["data"]["val_csv"]
TEST_CSV  = Path(ROOT) / cfg["data"]["test_csv"]


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval_rgb(model, csv_path, device) -> float:
    loader = DataLoader(VideoDataset(csv_path, NUM_FRAMES), batch_size=BATCH_SIZE, num_workers=0)
    model.eval(); correct = total = 0
    with torch.no_grad():
        for frames, labels in loader:
            correct += (model(frames.to(device)).argmax(1) == labels.to(device)).sum().item()
            total   += len(labels)
    return correct / total


def eval_pose(model, csv_path, device) -> float:
    loader = DataLoader(PoseDataset(csv_path), batch_size=BATCH_SIZE, num_workers=0)
    model.eval(); correct = total = 0
    with torch.no_grad():
        for pose, labels in loader:
            correct += (model(pose.to(device)).argmax(1) == labels.to(device)).sum().item()
            total   += len(labels)
    return correct / total


def eval_fusion(model, csv_path, device) -> float:
    loader = DataLoader(FusionDataset(csv_path), batch_size=BATCH_SIZE, num_workers=0)
    model.eval(); correct = total = 0
    with torch.no_grad():
        for frames, pose, labels in loader:
            out = model(frames.to(device), pose.to(device))
            # LateFusion / ConcatFusion return tensor; Adaptive returns tuple
            logits = out[0] if isinstance(out, tuple) else out
            correct += (logits.argmax(1) == labels.to(device)).sum().item()
            total   += len(labels)
    return correct / total


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}\n")

    rows = []

    # ── A: RGB Baseline ────────────────────────────────────────────────────────
    print("[A] RGB Baseline ...")
    m = RGBBaselineModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_model.pth", map_location=device))
    rows.append({
        "Model":              "RGB Baseline",
        "Train Accuracy":     eval_rgb(m, TRAIN_CSV, device),
        "Validation Accuracy":eval_rgb(m, VAL_CSV,   device),
        "Test Accuracy":      eval_rgb(m, TEST_CSV,  device),
        "Parameters":         count_params(m),
    })

    # ── B: Pose Only ───────────────────────────────────────────────────────────
    print("[B] Pose Only ...")
    m = PoseBranchModel(input_dim=258, num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_pose_model.pth", map_location=device))
    rows.append({
        "Model":              "Pose Only",
        "Train Accuracy":     eval_pose(m, TRAIN_CSV, device),
        "Validation Accuracy":eval_pose(m, VAL_CSV,   device),
        "Test Accuracy":      eval_pose(m, TEST_CSV,  device),
        "Parameters":         count_params(m),
    })

    # ── C: Late Fusion ─────────────────────────────────────────────────────────
    print("[C] Late Fusion ...")
    rgb_m  = RGBBaselineModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES)
    pose_m = PoseBranchModel(input_dim=258, num_classes=NUM_CLASSES, num_frames=NUM_FRAMES)
    rgb_m.load_state_dict(torch.load(CKPT_DIR / "best_model.pth",       map_location=device))
    pose_m.load_state_dict(torch.load(CKPT_DIR / "best_pose_model.pth", map_location=device))
    m = LateFusionModel(rgb_m, pose_m).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_late_fusion.pth", map_location=device))
    rows.append({
        "Model":              "Late Fusion",
        "Train Accuracy":     eval_fusion(m, TRAIN_CSV, device),
        "Validation Accuracy":eval_fusion(m, VAL_CSV,   device),
        "Test Accuracy":      eval_fusion(m, TEST_CSV,  device),
        "Parameters":         count_params(m),
    })

    # ── D: Feature Concat Fusion ───────────────────────────────────────────────
    print("[D] Feature Concat Fusion ...")
    m = FeatureConcatFusionModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_concat_fusion.pth", map_location=device))
    rows.append({
        "Model":              "Feature Concat Fusion",
        "Train Accuracy":     eval_fusion(m, TRAIN_CSV, device),
        "Validation Accuracy":eval_fusion(m, VAL_CSV,   device),
        "Test Accuracy":      eval_fusion(m, TEST_CSV,  device),
        "Parameters":         count_params(m),
    })

    # ── E: Adaptive Fusion ─────────────────────────────────────────────────────
    print("[E] Adaptive Fusion ...")
    m = AdaptiveMultiModalModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_adaptive_model.pth", map_location=device))
    rows.append({
        "Model":              "Adaptive Fusion",
        "Train Accuracy":     eval_fusion(m, TRAIN_CSV, device),
        "Validation Accuracy":eval_fusion(m, VAL_CSV,   device),
        "Test Accuracy":      eval_fusion(m, TEST_CSV,  device),
        "Parameters":         count_params(m),
    })

    # ── Print table ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {'Model':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'Params':>12}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}")
    for r in rows:
        print(f"  {r['Model']:<25} {r['Train Accuracy']:>7.2%}  "
              f"{r['Validation Accuracy']:>7.2%}  {r['Test Accuracy']:>7.2%}  "
              f"{r['Parameters']:>12,}")
    print(f"{'='*70}")

    best = max(rows, key=lambda x: x["Test Accuracy"])
    print(f"\n  Best model: {best['Model']} (Test Acc: {best['Test Accuracy']:.2%})\n")

    # ── Analysis ───────────────────────────────────────────────────────────────
    log_path = Path(ROOT) / "results" / "adaptive_training_log.csv"
    if log_path.exists():
        log_df    = pd.read_csv(log_path)
        avg_alpha = log_df["avg_alpha"].mean()
        avg_beta  = log_df["avg_beta"].mean()
        adaptive_test = next(r["Test Accuracy"] for r in rows if r["Model"] == "Adaptive Fusion")
        rgb_test      = next(r["Test Accuracy"] for r in rows if r["Model"] == "RGB Baseline")

        print("── Adaptive Fusion Analysis ──────────────────────────────")
        print(f"  Does adaptive fusion improve accuracy? : "
              f"{'YES' if adaptive_test > rgb_test else 'NO'} "
              f"({adaptive_test:.2%} vs RGB {rgb_test:.2%})")
        print(f"  Model relies more on              : "
              f"{'RGB' if avg_alpha > avg_beta else 'Pose'}")
        print(f"  Average α (RGB weight)            : {avg_alpha:.4f}")
        print(f"  Average β (Pose weight)           : {avg_beta:.4f}")
        print("──────────────────────────────────────────────────────────\n")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "week4_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV saved  → {csv_path}")

    # ── Bar chart ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    names  = [r["Model"] for r in rows]
    test_accs = [r["Test Accuracy"] * 100 for r in rows]
    colors = ["steelblue", "darkorange", "seagreen", "crimson", "purple"]
    bars   = ax.bar(names, test_accs, color=colors, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=10)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Week 4 — Model Comparison (Test Set)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    chart_path = RESULTS_DIR / "week4_comparison.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Chart saved → {chart_path}")
