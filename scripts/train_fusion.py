"""
train_fusion.py
Trains both fusion models sequentially:
  1. LateFusionModel          → checkpoints/best_late_fusion.pth
  2. FeatureConcatFusionModel → checkpoints/best_concat_fusion.pth

EmbeddingDataset loads precomputed RGB (16,2048) + Pose (16,258) .npy files.
Much faster than loading raw videos every batch.
"""

import os
import sys
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.rgb_branch import RGBBaselineModel
from models.pose_branch import PoseBranchModel
from models.fusion import LateFusionModel, FeatureConcatFusionModel

# ── Config ─────────────────────────────────────────────────────────────────────
with open(Path(ROOT) / "configs" / "baseline.yaml") as f:
    cfg = yaml.safe_load(f)

RGB_EMB_DIR  = Path(ROOT) / "datasets" / "rgb_embeddings"
POSE_DIR     = Path(ROOT) / "datasets" / "pose_data"
NUM_FRAMES   = cfg["data"]["num_frames"]


# ── EmbeddingDataset ───────────────────────────────────────────────────────────
class FusionDataset(Dataset):
    """
    Loads precomputed RGB embeddings (16, 2048) and Pose landmarks (16, 258)
    directly from .npy files. No video decoding or ResNet50 at runtime.
    Returns: rgb_emb (16,2048), pose (16,258), label_id
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        stem = Path(row["video_path"]).stem

        # Load precomputed RGB embedding (16, 2048)
        rgb_path = RGB_EMB_DIR / row["label"] / (stem + ".npy")
        if rgb_path.exists():
            rgb_emb = torch.tensor(np.load(str(rgb_path)).astype(np.float32))
        else:
            print(f"  [WARN] Missing RGB emb: {stem}.npy")
            rgb_emb = torch.zeros(NUM_FRAMES, 2048)

        # Load precomputed Pose landmarks (16, 258)
        pose_path = POSE_DIR / row["label"] / (stem + ".npy")
        if pose_path.exists():
            pose = torch.tensor(np.load(str(pose_path)).astype(np.float32))
        else:
            print(f"  [WARN] Missing Pose: {stem}.npy")
            pose = torch.zeros(NUM_FRAMES, 258)

        return rgb_emb, pose, int(row["label_id"])


# ── Epoch runner ───────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for frames, pose, labels in tqdm(loader, leave=False):
            frames = frames.to(device)
            pose   = pose.to(device)
            labels = labels.to(device)

            logits = model(frames, pose)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

    return total_loss / total, correct / total


# ── Train one model ────────────────────────────────────────────────────────────
def train_model(model, model_name: str, ckpt_name: str,
                train_loader, val_loader, device):

    epochs    = cfg["training"]["epochs"]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler    = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_acc = 0.0
    log_rows     = []
    ckpt_dir     = Path(ROOT) / cfg["paths"]["checkpoint_dir"]
    results_dir  = Path(ROOT) / cfg["paths"]["results_dir"]

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, device, training=True)
        val_loss, val_acc     = run_epoch(model, val_loader,   criterion,
                                          optimizer, device, training=False)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{epochs}  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        log_rows.append({
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6), "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_dir / ckpt_name)
            print(f"  ✔ Best {model_name} saved (val_acc={best_val_acc:.4f})")

    log_path = results_dir / f"training_log_{ckpt_name.replace('.pth','')}.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\n[INFO] Best Val Acc: {best_val_acc:.4f}")
    print(f"[INFO] Log saved → {log_path}")
    return best_val_acc


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Path(ROOT, cfg["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(ROOT, cfg["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_dataset = FusionDataset(Path(ROOT) / cfg["data"]["train_csv"])
    val_dataset   = FusionDataset(Path(ROOT) / cfg["data"]["val_csv"])
    train_loader  = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"],
                               shuffle=True,  num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=cfg["training"]["batch_size"],
                               shuffle=False, num_workers=0)

    print(f"[INFO] Train: {len(train_dataset)}  |  Val: {len(val_dataset)}\n")

    # ── 1. Late Fusion ─────────────────────────────────────────────────────────
    # Load pretrained RGB and Pose weights
    rgb_model  = RGBBaselineModel(num_classes=8, num_frames=NUM_FRAMES)
    pose_model = PoseBranchModel(input_dim=258, num_classes=8, num_frames=NUM_FRAMES)

    rgb_ckpt  = Path(ROOT) / cfg["paths"]["best_model"]
    pose_ckpt = Path(ROOT) / cfg["paths"]["checkpoint_dir"] / "best_pose_model.pth"

    if rgb_ckpt.exists():
        rgb_model.load_state_dict(torch.load(rgb_ckpt, map_location=device))
        print(f"[INFO] Loaded RGB weights  : {rgb_ckpt.name}")
    if pose_ckpt.exists():
        pose_model.load_state_dict(torch.load(pose_ckpt, map_location=device))
        print(f"[INFO] Loaded Pose weights : {pose_ckpt.name}")

    late_model = LateFusionModel(rgb_model, pose_model).to(device)
    train_model(late_model, "LateFusionModel", "best_late_fusion.pth",
                train_loader, val_loader, device)

    # ── 2. Feature Concat Fusion ───────────────────────────────────────────────
    concat_model = FeatureConcatFusionModel(num_classes=8, num_frames=NUM_FRAMES).to(device)
    train_model(concat_model, "FeatureConcatFusionModel", "best_concat_fusion.pth",
                train_loader, val_loader, device)
