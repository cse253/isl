"""
train_pose.py
Trains the PoseBranchModel on extracted .npy landmark sequences.
Saves best checkpoint and training log.
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

from models.pose_branch import PoseBranchModel

# ── Load config ────────────────────────────────────────────────────────────────
with open(Path(ROOT) / "configs" / "baseline.yaml") as f:
    cfg = yaml.safe_load(f)

POSE_DIR = Path(ROOT) / "datasets" / "pose_data"

# ── PoseDataset ────────────────────────────────────────────────────────────────
class PoseDataset(Dataset):
    """
    Reads a CSV with columns: video_path, label, label_id
    Derives the .npy path from video_path:
      datasets/Adjectives_1of8/Adjectives/1. loud/MVI_5177.MOV
      → datasets/pose_data/loud/MVI_5177.npy
    Returns: pose tensor (16, 258), label_id
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def _npy_path(self, video_path: str) -> Path:
        p     = Path(video_path)
        # label folder name is the cleaned class name (already stored in 'label' column)
        return POSE_DIR / p.parent.name.split(". ", 1)[-1] / (p.stem + ".npy")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        npy_path = Path(ROOT) / "datasets" / "pose_data" / row["label"] / (Path(row["video_path"]).stem + ".npy")

        if npy_path.exists():
            pose = np.load(str(npy_path)).astype(np.float32)  # (16, 258)
        else:
            print(f"  [WARN] Missing: {npy_path.name}, using zeros")
            pose = np.zeros((cfg["data"]["num_frames"], 258), dtype=np.float32)

        return torch.tensor(pose), int(row["label_id"])


# ── Epoch runner ───────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for pose, labels in tqdm(loader, leave=False):
            pose   = pose.to(device)
            labels = labels.to(device)

            logits = model(pose)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

    return total_loss / total, correct / total


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check if checkpoint already exists to save time on CPU
    ckpt_path = Path(ROOT) / cfg["paths"]["checkpoint_dir"] / "best_pose_model.pth"
    if ckpt_path.exists() and "--force" not in sys.argv:
        print(f"[INFO] Pre-trained model found at {ckpt_path.relative_to(ROOT)}. Skipping training (use --force to retrain).")
        sys.exit(0)

    ckpt_dir    = Path(ROOT) / cfg["paths"]["checkpoint_dir"]
    results_dir = Path(ROOT) / cfg["paths"]["results_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_dataset = PoseDataset(Path(ROOT) / cfg["data"]["train_csv"])
    val_dataset   = PoseDataset(Path(ROOT) / cfg["data"]["val_csv"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)

    print(f"[INFO] Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")

    model = PoseBranchModel(
        input_dim=258,
        num_classes=cfg["data"]["num_classes"],
        num_frames=cfg["data"]["num_frames"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    epochs       = cfg["training"]["epochs"]
    best_val_acc = 0.0
    log_rows     = []

    print(f"\n[INFO] Starting pose training for {epochs} epochs ...\n")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, training=False)
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
            torch.save(model.state_dict(), ckpt_dir / "best_pose_model.pth")
            print(f"  [OK] Best pose model saved (val_acc={best_val_acc:.4f})")

    log_path = results_dir / "training_log_pose.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\n[INFO] Training complete. Best Val Acc: {best_val_acc:.4f}")
    print(f"[INFO] Log saved -> {log_path}")
