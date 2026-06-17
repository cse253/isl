"""
visualize_fusion_weights.py
Reads adaptive_training_log.csv and plots alpha/beta per epoch.
Saves: results/fusion_weights.png
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

LOG_PATH    = Path(ROOT) / "results" / "adaptive_training_log.csv"
OUTPUT_PATH = Path(ROOT) / "results" / "fusion_weights.png"


if __name__ == "__main__":
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Log not found: {LOG_PATH}\nRun train_adaptive.py first.")

    df = pd.read_csv(LOG_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left: Alpha and Beta on same plot ─────────────────────────────────────
    axes[0].plot(df["epoch"], df["avg_alpha"], marker="o", color="steelblue",  label="α (RGB)")
    axes[0].plot(df["epoch"], df["avg_beta"],  marker="s", color="darkorange", label="β (Pose)")
    axes[0].axhline(0.5, linestyle="--", color="gray", linewidth=0.8, label="Equal weight (0.5)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Average Weight")
    axes[0].set_title("Adaptive Fusion Weights per Epoch")
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # ── Right: Val Accuracy ────────────────────────────────────────────────────
    axes[1].plot(df["epoch"], df["val_acc"], marker="o", color="seagreen", label="Val Accuracy")
    axes[1].plot(df["epoch"], df["train_acc"], marker="s", color="crimson",  label="Train Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Adaptive Multi-Modal Fusion — Training Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)

    # ── Print summary ──────────────────────────────────────────────────────────
    avg_alpha = df["avg_alpha"].mean()
    avg_beta  = df["avg_beta"].mean()
    print(f"\n[INFO] Fusion weight analysis:")
    print(f"  Overall avg alpha (RGB)  : {avg_alpha:.4f}")
    print(f"  Overall avg beta (Pose) : {avg_beta:.4f}")
    if avg_alpha > avg_beta:
        print(f"  -> Model relies MORE on RGB  ({avg_alpha:.2%} vs {avg_beta:.2%})")
    else:
        print(f"  -> Model relies MORE on Pose ({avg_beta:.2%} vs {avg_alpha:.2%})")
    print(f"\n[INFO] Plot saved -> {OUTPUT_PATH}")
