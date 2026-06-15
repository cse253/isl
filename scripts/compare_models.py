"""
compare_models.py
Evaluates all 4 models on the test set and prints a comparison table.

Models compared:
  1. RGB Baseline            (checkpoints/best_model.pth)
  2. Pose Only               (checkpoints/best_pose_model.pth)
  3. Late Fusion             (checkpoints/best_late_fusion.pth)
  4. Feature Concat Fusion   (checkpoints/best_concat_fusion.pth)

Saves:
  results/model_comparison.png
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from models.rgb_branch  import RGBBaselineModel
from models.pose_branch import PoseBranchModel
from models.fusion      import LateFusionModel, FeatureConcatFusionModel

# Reuse datasets from existing scripts
from scripts.train      import VideoDataset
from scripts.train_pose import PoseDataset
from scripts.train_fusion import FusionDataset

# ── Config ─────────────────────────────────────────────────────────────────────
with open(Path(ROOT) / "configs" / "baseline.yaml") as f:
    cfg = yaml.safe_load(f)

NUM_FRAMES  = cfg["data"]["num_frames"]
NUM_CLASSES = cfg["data"]["num_classes"]
BATCH_SIZE  = cfg["training"]["batch_size"]
TEST_CSV    = Path(ROOT) / cfg["data"]["test_csv"]
CKPT_DIR    = Path(ROOT) / cfg["paths"]["checkpoint_dir"]
RESULTS_DIR = Path(ROOT) / cfg["paths"]["results_dir"]


# ── Inference helpers ──────────────────────────────────────────────────────────
def eval_single(model, loader, device) -> float:
    """Evaluate a single-input model (RGB or Pose). Returns test accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            if isinstance(inputs, (list, tuple)):
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)
            preds  = model(inputs).argmax(1)
            correct += (preds == labels).sum().item()
            total   += len(labels)
    return correct / total


def eval_fusion(model, loader, device) -> float:
    """Evaluate a dual-input fusion model. Returns test accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for frames, pose, labels in loader:
            frames = frames.to(device)
            pose   = pose.to(device)
            labels = labels.to(device)
            preds  = model(frames, pose).argmax(1)
            correct += (preds == labels).sum().item()
            total   += len(labels)
    return correct / total


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}\n")

    results = {}

    # ── 1. RGB Baseline ────────────────────────────────────────────────────────
    print("[INFO] Evaluating RGB Baseline ...")
    rgb_model = RGBBaselineModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    rgb_model.load_state_dict(torch.load(CKPT_DIR / "best_model.pth", map_location=device))
    rgb_loader = DataLoader(VideoDataset(TEST_CSV, NUM_FRAMES), batch_size=BATCH_SIZE, num_workers=0)

    rgb_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for frames, labels in rgb_loader:
            frames, labels = frames.to(device), labels.to(device)
            correct += (rgb_model(frames).argmax(1) == labels).sum().item()
            total   += len(labels)
    results["RGB Baseline"] = correct / total
    print(f"  RGB Baseline     : {results['RGB Baseline']:.4f}")

    # ── 2. Pose Only ───────────────────────────────────────────────────────────
    print("[INFO] Evaluating Pose Only ...")
    pose_model = PoseBranchModel(input_dim=258, num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    pose_model.load_state_dict(torch.load(CKPT_DIR / "best_pose_model.pth", map_location=device))
    pose_loader = DataLoader(PoseDataset(TEST_CSV), batch_size=BATCH_SIZE, num_workers=0)

    pose_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for pose, labels in pose_loader:
            pose, labels = pose.to(device), labels.to(device)
            correct += (pose_model(pose).argmax(1) == labels).sum().item()
            total   += len(labels)
    results["Pose Only"] = correct / total
    print(f"  Pose Only        : {results['Pose Only']:.4f}")

    # ── 3. Late Fusion ─────────────────────────────────────────────────────────
    print("[INFO] Evaluating Late Fusion ...")
    rgb_m  = RGBBaselineModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES)
    pose_m = PoseBranchModel(input_dim=258, num_classes=NUM_CLASSES, num_frames=NUM_FRAMES)
    rgb_m.load_state_dict(torch.load(CKPT_DIR / "best_model.pth",      map_location=device))
    pose_m.load_state_dict(torch.load(CKPT_DIR / "best_pose_model.pth", map_location=device))
    late_model = LateFusionModel(rgb_m, pose_m).to(device)
    late_model.load_state_dict(torch.load(CKPT_DIR / "best_late_fusion.pth", map_location=device))
    fusion_loader = DataLoader(FusionDataset(TEST_CSV), batch_size=BATCH_SIZE, num_workers=0)
    results["Late Fusion"] = eval_fusion(late_model, fusion_loader, device)
    print(f"  Late Fusion      : {results['Late Fusion']:.4f}")

    # ── 4. Feature Concat Fusion ───────────────────────────────────────────────
    print("[INFO] Evaluating Feature Concat Fusion ...")
    concat_model = FeatureConcatFusionModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    concat_model.load_state_dict(torch.load(CKPT_DIR / "best_concat_fusion.pth", map_location=device))
    fusion_loader2 = DataLoader(FusionDataset(TEST_CSV), batch_size=BATCH_SIZE, num_workers=0)
    results["Feature Concat Fusion"] = eval_fusion(concat_model, fusion_loader2, device)
    print(f"  Concat Fusion    : {results['Feature Concat Fusion']:.4f}")

    # ── Print comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*45}")
    print(f"  {'Model':<25} {'Test Acc':>10}")
    print(f"  {'-'*25}  {'-'*10}")
    for name, acc in results.items():
        print(f"  {name:<25} {acc*100:>9.2f}%")
    print(f"{'='*45}")

    best_model = max(results, key=results.get)
    print(f"\n  Best model: {best_model} ({results[best_model]*100:.2f}%)\n")

    # ── Bar chart ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    bars   = ax.bar(results.keys(), [v * 100 for v in results.values()],
                    color=colors, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=10)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Model Comparison — ISL Adjectives (Test Set)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    out_path = RESULTS_DIR / "model_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Comparison chart saved → {out_path}")
