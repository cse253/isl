"""
week4_compare.py
Evaluates all 5 models on train/val/test sets and saves comparison results.

Speed optimisation
------------------
RGB Baseline, Late Fusion, and Feature Concat Fusion all contain a ResNet50
backbone that is VERY slow to run on raw video frames on CPU.
Instead we use the precomputed ResNet50 embeddings already stored in
  datasets/rgb_embeddings/<label>/<stem>.npy  (shape 16 x 2048)
and route each model accordingly:

  Model                  | frames input     | Dataset used
  -----------------------|------------------|----------------------
  RGBBaselineModel       | (B,T,2048)       | RGBEmbDataset
  PoseBranchModel        | (B,T,258)        | PoseDataset
  LateFusionModel        | (B,T,3,224,224)  | RawVideoPoseDataset
  FeatureConcatFusion    | (B,T,3,224,224)  | RawVideoPoseDataset
  AdaptiveMultiModalModel| (B,T,2048)       | FusionDataset

Wait -- LateFusion / ConcatFusion contain the CNN backbone, so they actually
CAN accept precomputed embeddings only if we bypass the CNN.  Since those
models store the full ResNet50 internally and were trained on raw frames, we
must evaluate them on raw frames.  However, to keep evaluation fast we avoid
re-loading the CNN for the RGB-only baseline by using precomputed embeddings
+ a lightweight rgb_proj layer (as AdaptiveMultiModalModel does).

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
from torch.utils.data import Dataset, DataLoader

from models.rgb_branch    import RGBBaselineModel
from models.pose_branch   import PoseBranchModel
from models.fusion        import LateFusionModel, FeatureConcatFusionModel
from models.adaptive_model import AdaptiveMultiModalModel

from scripts.train        import VideoDataset
from scripts.train_pose   import PoseDataset
from scripts.train_fusion import FusionDataset   # returns (rgb_emb 16x2048, pose 16x258, label)

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

RGB_EMB_DIR = Path(ROOT) / "datasets" / "rgb_embeddings"
POSE_DIR    = Path(ROOT) / "datasets" / "pose_data"


# ── Helper datasets ────────────────────────────────────────────────────────────

class RGBEmbDataset(Dataset):
    """
    Loads precomputed ResNet50 embeddings (16, 2048).
    Used to evaluate RGBBaselineModel fast — no video decoding needed.
    """
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        stem = Path(row["video_path"]).stem
        npy  = RGB_EMB_DIR / row["label"] / (stem + ".npy")
        if npy.exists():
            emb = torch.tensor(np.load(str(npy)).astype(np.float32))  # (16, 2048)
        else:
            print(f"  [WARN] Missing RGB emb: {stem}.npy  -> using zeros")
            emb = torch.zeros(NUM_FRAMES, 2048)
        return emb, int(row["label_id"])


class RawVideoPoseDataset(Dataset):
    """
    Returns (raw_frames (16,3,224,224), pose (16,258), label).
    Used for LateFusionModel and FeatureConcatFusionModel which contain
    the ResNet50 CNN internally and need raw pixel input.
    """
    def __init__(self, csv_path: str, num_frames: int = 16):
        self.video_ds = VideoDataset(csv_path, num_frames)
        self.pose_ds  = PoseDataset(csv_path)

    def __len__(self):
        return len(self.video_ds)

    def __getitem__(self, idx):
        frames, label = self.video_ds[idx]
        pose,   _     = self.pose_ds[idx]
        return frames, pose, label


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval_rgb_fast(model: RGBBaselineModel, csv_path, device) -> float:
    """
    Evaluate RGB Baseline using precomputed embeddings.
    AdaptiveMultiModalModel also accepts (B,T,2048), and RGBBaselineModel
    was trained end-to-end with its own CNN.  We replicate its forward pass
    using only the transformer + classifier layers (skip the CNN) by passing
    embeddings through a temporary projection.

    NOTE: Since RGBBaselineModel.input_proj expects 2048-dim input (it projects
    2048 -> d_model), we can feed the embeddings directly after reshaping —
    this is exactly what the model does internally after the CNN.
    """
    loader = DataLoader(RGBEmbDataset(csv_path), batch_size=BATCH_SIZE, num_workers=0)
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for emb, labels in loader:
            # emb shape: (B, T, 2048) — same as what CNN produces per frame
            B, T, D = emb.shape
            emb = emb.to(device)

            # Manually run the transformer portion of RGBBaselineModel:
            # 1. Project 2048 -> d_model
            x = model.input_proj(emb.view(B * T, D)).view(B, T, -1)  # (B, T, d_model)
            # 2. Add positional embedding
            x = x + model.pos_embedding[:, :T, :]
            # 3. Transformer
            x = model.transformer(x)   # (B, T, d_model)
            # 4. Mean pool
            x = x.mean(dim=1)          # (B, d_model)
            # 5. Classify
            logits = model.classifier(x)  # (B, num_classes)

            correct += (logits.argmax(1) == labels.to(device)).sum().item()
            total   += len(labels)

    return correct / total


def eval_pose(model: PoseBranchModel, csv_path, device) -> float:
    """Evaluate Pose-Only model using precomputed .npy pose files."""
    loader = DataLoader(PoseDataset(csv_path), batch_size=BATCH_SIZE, num_workers=0)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for pose, labels in loader:
            correct += (model(pose.to(device)).argmax(1) == labels.to(device)).sum().item()
            total   += len(labels)
    return correct / total


def eval_fusion(model, csv_path, device) -> float:
    """
    Evaluate a dual-input fusion model.
    - AdaptiveMultiModalModel  -> FusionDataset  (precomputed RGB emb + pose)
    - LateFusion / ConcatFusion -> RawVideoPoseDataset (raw frames + pose)
    """
    if isinstance(model, AdaptiveMultiModalModel):
        loader = DataLoader(FusionDataset(csv_path), batch_size=BATCH_SIZE, num_workers=0)
    else:
        loader = DataLoader(RawVideoPoseDataset(csv_path, NUM_FRAMES),
                            batch_size=BATCH_SIZE, num_workers=0)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for frames, pose, labels in loader:
            if total == 0:
                print(f"  [DEBUG] {model.__class__.__name__} | "
                      f"frames={tuple(frames.shape)}  pose={tuple(pose.shape)}")
            out    = model(frames.to(device), pose.to(device))
            logits = out[0] if isinstance(out, tuple) else out   # Adaptive returns tuple
            correct += (logits.argmax(1) == labels.to(device)).sum().item()
            total   += len(labels)
    return correct / total


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}\n")

    rows = []

    # -- A: RGB Baseline --------------------------------------------------------
    print("[A] RGB Baseline (fast — using precomputed embeddings) ...")
    m = RGBBaselineModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_model.pth", map_location=device))
    rows.append({
        "Model":               "RGB Baseline",
        "Train Accuracy":      eval_rgb_fast(m, TRAIN_CSV, device),
        "Validation Accuracy": eval_rgb_fast(m, VAL_CSV,   device),
        "Test Accuracy":       eval_rgb_fast(m, TEST_CSV,  device),
        "Parameters":          count_params(m),
    })
    print(f"  Train {rows[-1]['Train Accuracy']:.2%}  Val {rows[-1]['Validation Accuracy']:.2%}"
          f"  Test {rows[-1]['Test Accuracy']:.2%}")

    # -- B: Pose Only -----------------------------------------------------------
    print("[B] Pose Only ...")
    m = PoseBranchModel(input_dim=258, num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_pose_model.pth", map_location=device))
    rows.append({
        "Model":               "Pose Only",
        "Train Accuracy":      eval_pose(m, TRAIN_CSV, device),
        "Validation Accuracy": eval_pose(m, VAL_CSV,   device),
        "Test Accuracy":       eval_pose(m, TEST_CSV,  device),
        "Parameters":          count_params(m),
    })
    print(f"  Train {rows[-1]['Train Accuracy']:.2%}  Val {rows[-1]['Validation Accuracy']:.2%}"
          f"  Test {rows[-1]['Test Accuracy']:.2%}")

    # -- C: Late Fusion ---------------------------------------------------------
    print("[C] Late Fusion (raw video frames) ...")
    rgb_m  = RGBBaselineModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES)
    pose_m = PoseBranchModel(input_dim=258, num_classes=NUM_CLASSES, num_frames=NUM_FRAMES)
    rgb_m.load_state_dict(torch.load(CKPT_DIR / "best_model.pth",       map_location=device))
    pose_m.load_state_dict(torch.load(CKPT_DIR / "best_pose_model.pth", map_location=device))
    m = LateFusionModel(rgb_m, pose_m).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_late_fusion.pth", map_location=device))
    rows.append({
        "Model":               "Late Fusion",
        "Train Accuracy":      eval_fusion(m, TRAIN_CSV, device),
        "Validation Accuracy": eval_fusion(m, VAL_CSV,   device),
        "Test Accuracy":       eval_fusion(m, TEST_CSV,  device),
        "Parameters":          count_params(m),
    })
    print(f"  Train {rows[-1]['Train Accuracy']:.2%}  Val {rows[-1]['Validation Accuracy']:.2%}"
          f"  Test {rows[-1]['Test Accuracy']:.2%}")

    # -- D: Feature Concat Fusion -----------------------------------------------
    print("[D] Feature Concat Fusion (raw video frames) ...")
    m = FeatureConcatFusionModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_concat_fusion.pth", map_location=device))
    rows.append({
        "Model":               "Feature Concat Fusion",
        "Train Accuracy":      eval_fusion(m, TRAIN_CSV, device),
        "Validation Accuracy": eval_fusion(m, VAL_CSV,   device),
        "Test Accuracy":       eval_fusion(m, TEST_CSV,  device),
        "Parameters":          count_params(m),
    })
    print(f"  Train {rows[-1]['Train Accuracy']:.2%}  Val {rows[-1]['Validation Accuracy']:.2%}"
          f"  Test {rows[-1]['Test Accuracy']:.2%}")

    # -- E: Adaptive Fusion (Week 4) --------------------------------------------
    print("[E] Adaptive Fusion (precomputed embeddings) ...")
    m = AdaptiveMultiModalModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_adaptive_model.pth", map_location=device))
    rows.append({
        "Model":               "Adaptive Fusion",
        "Train Accuracy":      eval_fusion(m, TRAIN_CSV, device),
        "Validation Accuracy": eval_fusion(m, VAL_CSV,   device),
        "Test Accuracy":       eval_fusion(m, TEST_CSV,  device),
        "Parameters":          count_params(m),
    })
    print(f"  Train {rows[-1]['Train Accuracy']:.2%}  Val {rows[-1]['Validation Accuracy']:.2%}"
          f"  Test {rows[-1]['Test Accuracy']:.2%}")

    # -- Print comparison table -------------------------------------------------
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

    # -- Adaptive Fusion Analysis -----------------------------------------------
    log_path = Path(ROOT) / "results" / "adaptive_training_log.csv"
    if log_path.exists():
        log_df    = pd.read_csv(log_path)
        avg_alpha = log_df["avg_alpha"].mean()
        avg_beta  = log_df["avg_beta"].mean()
        adaptive_test = next(r["Test Accuracy"] for r in rows if r["Model"] == "Adaptive Fusion")
        rgb_test      = next(r["Test Accuracy"] for r in rows if r["Model"] == "RGB Baseline")

        print("-- Adaptive Fusion Analysis ------------------------------------------")
        print(f"  Does adaptive fusion improve accuracy? : "
              f"{'YES' if adaptive_test > rgb_test else 'NO'} "
              f"({adaptive_test:.2%} vs RGB {rgb_test:.2%})")
        print(f"  Model relies more on        : "
              f"{'RGB' if avg_alpha > avg_beta else 'Pose'}")
        print(f"  Average alpha (RGB weight)  : {avg_alpha:.4f}")
        print(f"  Average beta  (Pose weight) : {avg_beta:.4f}")
        print("----------------------------------------------------------------------\n")

    # -- Save CSV ---------------------------------------------------------------
    csv_path = RESULTS_DIR / "week4_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV saved  -> {csv_path}")

    # -- Bar chart --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    names     = [r["Model"] for r in rows]
    test_accs = [r["Test Accuracy"] * 100 for r in rows]
    colors    = ["steelblue", "darkorange", "seagreen", "crimson", "purple"]
    bars      = ax.bar(names, test_accs, color=colors, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=10)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Week 4 - Model Comparison (Test Set)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    chart_path = RESULTS_DIR / "week4_comparison.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Chart saved -> {chart_path}")
