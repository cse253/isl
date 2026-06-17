"""
analyze_fusion_weights.py
Week 5, Task 4 — Adaptive Fusion Weight Analysis

Runs the Adaptive Fusion model on every test sample and records:
  - video_path       : which video
  - true_label       : ground truth sign class
  - predicted_label  : model's prediction
  - alpha_rgb        : RGB weight (how much RGB influenced this prediction)
  - beta_pose        : Pose weight (how much Pose influenced this prediction)

Key insight:
  alpha + beta = 1 (guaranteed by Softmax in AdaptiveFusion)

  alpha > 0.5 -> model prefers RGB for this sample
  beta  > 0.5 -> model prefers Pose for this sample

Saves:
  results/fusion_weight_analysis.csv
  results/fusion_weight_distribution.png

Prints:
  Average alpha (RGB weight)
  Average beta  (Pose weight)
  Per-class preference analysis
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

from models.adaptive_model import AdaptiveMultiModalModel
from scripts.train_fusion   import FusionDataset

# ── Config ─────────────────────────────────────────────────────────────────────
with open(Path(ROOT) / "configs" / "baseline.yaml") as f:
    cfg = yaml.safe_load(f)

NUM_FRAMES  = cfg["data"]["num_frames"]
NUM_CLASSES = cfg["data"]["num_classes"]
BATCH_SIZE  = cfg["training"]["batch_size"]
CKPT_DIR    = Path(ROOT) / cfg["paths"]["checkpoint_dir"]
RESULTS_DIR = Path(ROOT) / cfg["paths"]["results_dir"]
TEST_CSV    = Path(ROOT) / cfg["data"]["test_csv"]
TRAIN_CSV   = Path(ROOT) / cfg["data"]["train_csv"]
VAL_CSV     = Path(ROOT) / cfg["data"]["val_csv"]


# ── Collect per-sample fusion weights ─────────────────────────────────────────

def collect_fusion_weights(model: torch.nn.Module,
                           csv_path: str,
                           device: torch.device) -> list:
    """
    Run Adaptive Fusion on all samples in csv_path.
    Returns list of dicts with video_path, labels, alpha, beta, correct.
    """
    df     = pd.read_csv(csv_path)
    loader = DataLoader(FusionDataset(csv_path),
                        batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    records = []

    with torch.no_grad():
        for i, (rgb_emb, pose, label_id) in enumerate(loader):
            logits, alpha, beta = model(rgb_emb.to(device), pose.to(device))
            pred_id = logits.argmax(1).item()

            row = df.iloc[i]
            records.append({
                "video_path":      row["video_path"],
                "true_label":      row["label"],
                "true_label_id":   int(label_id.item()),
                "predicted_label": row["label"] if pred_id == int(row["label_id"]) else
                                   df[df["label_id"] == pred_id]["label"].iloc[0]
                                   if len(df[df["label_id"] == pred_id]) > 0
                                   else str(pred_id),
                "predicted_label_id": pred_id,
                "alpha_rgb":       float(alpha[0].item()),
                "beta_pose":       float(beta[0].item()),
                "correct":         pred_id == int(row["label_id"]),
            })

    return records


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load Adaptive Fusion model
    model = AdaptiveMultiModalModel(num_classes=NUM_CLASSES,
                                    num_frames=NUM_FRAMES).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / "best_adaptive_model.pth",
                                     map_location=device))
    model.eval()

    # ── Collect weights from ALL splits for a comprehensive view ───────────────
    print("[INFO] Collecting fusion weights from test set ...")
    test_records  = collect_fusion_weights(model, TEST_CSV,  device)
    print("[INFO] Collecting fusion weights from train set ...")
    train_records = collect_fusion_weights(model, TRAIN_CSV, device)
    print("[INFO] Collecting fusion weights from val set ...")
    val_records   = collect_fusion_weights(model, VAL_CSV,   device)

    all_records = test_records + train_records + val_records
    print(f"[INFO] Total records collected: {len(all_records)}\n")

    # ── Analysis: compute statistics ──────────────────────────────────────────
    alphas = [r["alpha_rgb"]  for r in all_records]
    betas  = [r["beta_pose"]  for r in all_records]

    avg_alpha = np.mean(alphas)
    avg_beta  = np.mean(betas)
    std_alpha = np.std(alphas)
    std_beta  = np.std(betas)

    print("=" * 60)
    print("  ADAPTIVE FUSION WEIGHT ANALYSIS")
    print("=" * 60)
    print(f"  Total samples      : {len(all_records)}")
    print(f"  Average alpha (RGB): {avg_alpha:.4f}  (std={std_alpha:.4f})")
    print(f"  Average beta (Pose): {avg_beta:.4f}  (std={std_beta:.4f})")
    print(f"  Dominant modality  : {'RGB' if avg_alpha > avg_beta else 'Pose'}")
    print(f"  Samples prefer RGB : "
          f"{sum(1 for a in alphas if a > 0.5)} / {len(alphas)}")
    print(f"  Samples prefer Pose: "
          f"{sum(1 for b in betas if b > 0.5)} / {len(betas)}")
    print("=" * 60)

    # Per-class analysis
    df_all = pd.DataFrame(all_records)
    print("\n  Per-class average alpha (RGB weight):")
    per_class = df_all.groupby("true_label")["alpha_rgb"].agg(["mean", "std"])
    for label, row_data in per_class.iterrows():
        dominant = "RGB" if row_data["mean"] > 0.5 else "Pose"
        print(f"    {label:12s}: alpha={row_data['mean']:.4f}  (std={row_data['std']:.4f})"
              f"  -> prefers {dominant}")
    print()

    # ── Save CSV (test set only for the required output) ───────────────────────
    # Keep only the columns required by the task spec
    out_records = [{
        "video_path":       r["video_path"],
        "true_label":       r["true_label"],
        "predicted_label":  r["predicted_label"],
        "alpha_rgb":        round(r["alpha_rgb"], 6),
        "beta_pose":        round(r["beta_pose"], 6),
    } for r in test_records]

    csv_path = RESULTS_DIR / "fusion_weight_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_records[0].keys())
        writer.writeheader()
        writer.writerows(out_records)
    print(f"[INFO] CSV saved -> {csv_path}")

    # ── Distribution plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.set_style("whitegrid")

    # 1. Alpha distribution histogram
    axes[0, 0].hist(alphas, bins=20, color="steelblue", edgecolor="white",
                    alpha=0.85)
    axes[0, 0].axvline(avg_alpha, color="crimson", linestyle="--",
                        linewidth=2, label=f"Mean={avg_alpha:.3f}")
    axes[0, 0].axvline(0.5, color="gray", linestyle=":", linewidth=1.5,
                        label="50% threshold")
    axes[0, 0].set_title("Distribution of Alpha (RGB weight)", fontweight="bold")
    axes[0, 0].set_xlabel("Alpha value")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()

    # 2. Beta distribution histogram
    axes[0, 1].hist(betas, bins=20, color="darkorange", edgecolor="white",
                    alpha=0.85)
    axes[0, 1].axvline(avg_beta, color="crimson", linestyle="--",
                        linewidth=2, label=f"Mean={avg_beta:.3f}")
    axes[0, 1].axvline(0.5, color="gray", linestyle=":", linewidth=1.5,
                        label="50% threshold")
    axes[0, 1].set_title("Distribution of Beta (Pose weight)", fontweight="bold")
    axes[0, 1].set_xlabel("Beta value")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()

    # 3. Alpha vs Beta scatter (should sum to 1)
    correct_mask = np.array([r["correct"] for r in all_records])
    axes[1, 0].scatter(
        np.array(alphas)[correct_mask],
        np.array(betas)[correct_mask],
        c="seagreen", alpha=0.7, label="Correct", s=60, edgecolors="white"
    )
    axes[1, 0].scatter(
        np.array(alphas)[~correct_mask],
        np.array(betas)[~correct_mask],
        c="crimson", alpha=0.7, label="Wrong", s=60, marker="x", linewidths=2
    )
    axes[1, 0].plot([0, 1], [1, 0], "k--", alpha=0.3, label="alpha+beta=1")
    axes[1, 0].set_xlabel("Alpha (RGB weight)")
    axes[1, 0].set_ylabel("Beta (Pose weight)")
    axes[1, 0].set_title("Alpha vs Beta (per sample)", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].set_xlim(-0.05, 1.05)
    axes[1, 0].set_ylim(-0.05, 1.05)

    # 4. Per-class alpha box plot
    df_all.boxplot(column="alpha_rgb", by="true_label", ax=axes[1, 1],
                   patch_artist=True)
    axes[1, 1].set_title("Alpha (RGB weight) per class", fontweight="bold")
    axes[1, 1].set_xlabel("Class")
    axes[1, 1].set_ylabel("Alpha value")
    axes[1, 1].axhline(0.5, color="crimson", linestyle="--",
                        linewidth=1.5, label="0.5 threshold")
    plt.sca(axes[1, 1])
    plt.xticks(rotation=20, ha="right")

    fig.suptitle("Week 5 — Adaptive Fusion Weight Analysis", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    dist_path = RESULTS_DIR / "fusion_weight_distribution.png"
    fig.savefig(str(dist_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Distribution plot saved -> {dist_path}")
