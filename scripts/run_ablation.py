"""
run_ablation.py
Week 5, Task 1 — Ablation Study

Evaluates all 5 models and records comprehensive metrics:
  Accuracy, Precision, Recall, F1-Score, Parameter Count

Strategy for speed on CPU:
  - RGB Baseline / Late Fusion / Concat Fusion:
      Uses VideoDataset (raw frames) on VAL+TEST only (21 samples total — fast).
      Loads Train accuracy from existing training logs.
  - Pose Only / Adaptive Fusion:
      Uses precomputed .npy files — fast on all splits.

Saves:
  results/ablation_results.csv
  results/ablation_accuracy_comparison.png
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from models.rgb_branch     import RGBBaselineModel
from models.pose_branch    import PoseBranchModel
from models.fusion         import LateFusionModel, FeatureConcatFusionModel
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
TRAIN_CSV   = Path(ROOT) / cfg["data"]["train_csv"]
VAL_CSV     = Path(ROOT) / cfg["data"]["val_csv"]
TEST_CSV    = Path(ROOT) / cfg["data"]["test_csv"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def count_params(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_metrics(y_true: list, y_pred: list, num_classes: int) -> dict:
    """
    Compute Precision, Recall, F1 using sklearn (macro average).
    Handles case where some classes may not appear in predictions.
    """
    labels = list(range(num_classes))
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, labels=labels,
                                     average="macro", zero_division=0),
        "Recall":    recall_score(y_true, y_pred, labels=labels,
                                  average="macro", zero_division=0),
        "F1 Score":  f1_score(y_true, y_pred, labels=labels,
                              average="macro", zero_division=0),
    }


# ── Inference helpers ─────────────────────────────────────────────────────────

def predict_rgb(model: torch.nn.Module, csv_path: str, device: torch.device):
    """Run RGB Baseline on raw video frames. Returns (y_true, y_pred)."""
    loader = DataLoader(VideoDataset(csv_path, NUM_FRAMES),
                        batch_size=BATCH_SIZE, num_workers=0)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for frames, labels in loader:
            logits = model(frames.to(device))
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred


def predict_pose(model: torch.nn.Module, csv_path: str, device: torch.device):
    """Run Pose-Only model. Returns (y_true, y_pred)."""
    loader = DataLoader(PoseDataset(csv_path),
                        batch_size=BATCH_SIZE, num_workers=0)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for pose, labels in loader:
            logits = model(pose.to(device))
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred


def predict_raw_fusion(model: torch.nn.Module, csv_path: str, device: torch.device):
    """
    Run Late/Concat Fusion on raw video + pose.
    These models have ResNet50 internally, so need raw pixel frames.
    """
    from scripts.week4_compare import RawVideoPoseDataset   # reuse existing helper
    loader = DataLoader(RawVideoPoseDataset(csv_path, NUM_FRAMES),
                        batch_size=BATCH_SIZE, num_workers=0)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for frames, pose, labels in loader:
            out = model(frames.to(device), pose.to(device))
            logits = out[0] if isinstance(out, tuple) else out
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred


def predict_adaptive(model: torch.nn.Module, csv_path: str, device: torch.device):
    """Run Adaptive Fusion using precomputed embeddings. Fast."""
    loader = DataLoader(FusionDataset(csv_path),
                        batch_size=BATCH_SIZE, num_workers=0)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for rgb_emb, pose, labels in loader:
            logits, _, _ = model(rgb_emb.to(device), pose.to(device))
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred


def load_train_acc(log_csv: str) -> float:
    """Read best train accuracy from existing training log CSV."""
    try:
        df = pd.read_csv(Path(ROOT) / log_csv)
        return float(df["train_acc"].max())
    except Exception:
        return float("nan")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}\n")
    print("[INFO] Note: RGB-based models evaluate on val+test; train acc from logs.\n")

    rows = []

    # ==========================================================================
    # A — RGB Baseline
    # ==========================================================================
    print("[A] RGB Baseline ...")
    m = RGBBaselineModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_model.pth", map_location=device))

    val_true,  val_pred  = predict_rgb(m, VAL_CSV,  device)
    test_true, test_pred = predict_rgb(m, TEST_CSV, device)

    val_m  = compute_metrics(val_true,  val_pred,  NUM_CLASSES)
    test_m = compute_metrics(test_true, test_pred, NUM_CLASSES)
    train_acc = load_train_acc("results/training_log.csv")

    rows.append({
        "Model":          "RGB Baseline",
        "Train Accuracy": round(train_acc, 4),
        "Val Accuracy":   round(val_m["Accuracy"], 4),
        "Test Accuracy":  round(test_m["Accuracy"], 4),
        "Precision":      round(test_m["Precision"], 4),
        "Recall":         round(test_m["Recall"], 4),
        "F1 Score":       round(test_m["F1 Score"], 4),
        "Parameters":     count_params(m),
    })
    print(f"  Test Acc={test_m['Accuracy']:.2%}  F1={test_m['F1 Score']:.4f}")

    # ==========================================================================
    # B — Pose Only
    # ==========================================================================
    print("[B] Pose Only ...")
    m = PoseBranchModel(input_dim=258, num_classes=NUM_CLASSES,
                        num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_pose_model.pth",
                                 map_location=device))

    # Pose is fast: run all 3 splits
    train_true, train_pred = predict_pose(m, TRAIN_CSV, device)
    val_true,   val_pred   = predict_pose(m, VAL_CSV,   device)
    test_true,  test_pred  = predict_pose(m, TEST_CSV,  device)

    train_m = compute_metrics(train_true, train_pred, NUM_CLASSES)
    val_m   = compute_metrics(val_true,   val_pred,   NUM_CLASSES)
    test_m  = compute_metrics(test_true,  test_pred,  NUM_CLASSES)

    rows.append({
        "Model":          "Pose Only",
        "Train Accuracy": round(train_m["Accuracy"], 4),
        "Val Accuracy":   round(val_m["Accuracy"], 4),
        "Test Accuracy":  round(test_m["Accuracy"], 4),
        "Precision":      round(test_m["Precision"], 4),
        "Recall":         round(test_m["Recall"], 4),
        "F1 Score":       round(test_m["F1 Score"], 4),
        "Parameters":     count_params(m),
    })
    print(f"  Test Acc={test_m['Accuracy']:.2%}  F1={test_m['F1 Score']:.4f}")

    # ==========================================================================
    # C — Late Fusion
    # ==========================================================================
    print("[C] Late Fusion ...")
    rgb_m  = RGBBaselineModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES)
    pose_m = PoseBranchModel(input_dim=258, num_classes=NUM_CLASSES,
                             num_frames=NUM_FRAMES)
    rgb_m.load_state_dict(torch.load(CKPT_DIR / "best_model.pth",
                                     map_location=device))
    pose_m.load_state_dict(torch.load(CKPT_DIR / "best_pose_model.pth",
                                      map_location=device))
    m = LateFusionModel(rgb_m, pose_m).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_late_fusion.pth",
                                 map_location=device))

    val_true,  val_pred  = predict_raw_fusion(m, VAL_CSV,  device)
    test_true, test_pred = predict_raw_fusion(m, TEST_CSV, device)
    val_m  = compute_metrics(val_true,  val_pred,  NUM_CLASSES)
    test_m = compute_metrics(test_true, test_pred, NUM_CLASSES)
    train_acc = load_train_acc("results/training_log_best_late_fusion.csv")

    rows.append({
        "Model":          "Late Fusion",
        "Train Accuracy": round(train_acc, 4),
        "Val Accuracy":   round(val_m["Accuracy"], 4),
        "Test Accuracy":  round(test_m["Accuracy"], 4),
        "Precision":      round(test_m["Precision"], 4),
        "Recall":         round(test_m["Recall"], 4),
        "F1 Score":       round(test_m["F1 Score"], 4),
        "Parameters":     count_params(m),
    })
    print(f"  Test Acc={test_m['Accuracy']:.2%}  F1={test_m['F1 Score']:.4f}")

    # ==========================================================================
    # D — Feature Concat Fusion
    # ==========================================================================
    print("[D] Feature Concat Fusion ...")
    m = FeatureConcatFusionModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_concat_fusion.pth",
                                 map_location=device))

    val_true,  val_pred  = predict_raw_fusion(m, VAL_CSV,  device)
    test_true, test_pred = predict_raw_fusion(m, TEST_CSV, device)
    val_m  = compute_metrics(val_true,  val_pred,  NUM_CLASSES)
    test_m = compute_metrics(test_true, test_pred, NUM_CLASSES)
    train_acc = load_train_acc("results/training_log_best_concat_fusion.csv")

    rows.append({
        "Model":          "Feature Concat Fusion",
        "Train Accuracy": round(train_acc, 4),
        "Val Accuracy":   round(val_m["Accuracy"], 4),
        "Test Accuracy":  round(test_m["Accuracy"], 4),
        "Precision":      round(test_m["Precision"], 4),
        "Recall":         round(test_m["Recall"], 4),
        "F1 Score":       round(test_m["F1 Score"], 4),
        "Parameters":     count_params(m),
    })
    print(f"  Test Acc={test_m['Accuracy']:.2%}  F1={test_m['F1 Score']:.4f}")

    # ==========================================================================
    # E — Adaptive Fusion (Week 4)
    # ==========================================================================
    print("[E] Adaptive Fusion ...")
    m = AdaptiveMultiModalModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_adaptive_model.pth",
                                 map_location=device))

    # Adaptive uses precomputed embeddings — fast on all splits
    train_true, train_pred = predict_adaptive(m, TRAIN_CSV, device)
    val_true,   val_pred   = predict_adaptive(m, VAL_CSV,   device)
    test_true,  test_pred  = predict_adaptive(m, TEST_CSV,  device)

    train_m = compute_metrics(train_true, train_pred, NUM_CLASSES)
    val_m   = compute_metrics(val_true,   val_pred,   NUM_CLASSES)
    test_m  = compute_metrics(test_true,  test_pred,  NUM_CLASSES)

    rows.append({
        "Model":          "Adaptive Fusion",
        "Train Accuracy": round(train_m["Accuracy"], 4),
        "Val Accuracy":   round(val_m["Accuracy"], 4),
        "Test Accuracy":  round(test_m["Accuracy"], 4),
        "Precision":      round(test_m["Precision"], 4),
        "Recall":         round(test_m["Recall"], 4),
        "F1 Score":       round(test_m["F1 Score"], 4),
        "Parameters":     count_params(m),
    })
    print(f"  Test Acc={test_m['Accuracy']:.2%}  F1={test_m['F1 Score']:.4f}")

    # ── Print comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  {'Model':<24} {'Train':>7} {'Val':>7} {'Test':>7} "
          f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'Params':>12}")
    print(f"  {'-'*24}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*12}")
    for r in rows:
        print(f"  {r['Model']:<24} {r['Train Accuracy']:>6.2%}  {r['Val Accuracy']:>6.2%}"
              f"  {r['Test Accuracy']:>6.2%}  {r['Precision']:>6.4f}"
              f"  {r['Recall']:>6.4f}  {r['F1 Score']:>6.4f}  {r['Parameters']:>12,}")
    print(f"{'='*90}\n")

    best = max(rows, key=lambda x: x["Test Accuracy"])
    print(f"  Best model: {best['Model']} (Test Acc: {best['Test Accuracy']:.2%})\n")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "ablation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV saved -> {csv_path}")

    # ── Bar chart (Test Accuracy) ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_style("whitegrid")
    palette = ["steelblue", "darkorange", "seagreen", "crimson", "purple"]
    models  = [r["Model"] for r in rows]

    # Left: Test Accuracy
    test_accs = [r["Test Accuracy"] * 100 for r in rows]
    bars = axes[0].bar(models, test_accs, color=palette, edgecolor="white", width=0.55)
    axes[0].bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    axes[0].set_title("Test Accuracy by Model", fontweight="bold", fontsize=13)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 110)
    axes[0].tick_params(axis="x", rotation=20)

    # Right: F1 Scores
    f1s = [r["F1 Score"] for r in rows]
    bars2 = axes[1].bar(models, f1s, color=palette, edgecolor="white", width=0.55)
    axes[1].bar_label(bars2, fmt="%.3f", padding=4, fontsize=9)
    axes[1].set_title("Macro F1 Score by Model (Test Set)", fontweight="bold", fontsize=13)
    axes[1].set_ylabel("F1 Score")
    axes[1].set_ylim(0, 1.15)
    axes[1].tick_params(axis="x", rotation=20)

    plt.suptitle("Week 5 — Ablation Study", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    chart_path = RESULTS_DIR / "ablation_accuracy_comparison.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Chart saved -> {chart_path}")
