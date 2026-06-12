"""
train.py
Training script for the RGB Baseline Model.
Reads config from configs/baseline.yaml, trains for N epochs,
saves best checkpoint and training log.
"""

import os
import sys
import csv

# Fix import path — add project root to sys.path so 'models' package is found
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import yaml
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.rgb_branch import RGBBaselineModel

# ── Load config ────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(ROOT) / "configs" / "baseline.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# ── VideoDataset ───────────────────────────────────────────────────────────────
class VideoDataset(Dataset):
    """
    Reads a CSV with columns: video_path, label, label_id
    Loads each video with OpenCV, uniformly samples num_frames frames,
    resizes to img_size x img_size, normalizes with ImageNet stats.
    """

    def __init__(self, csv_path: str, num_frames: int = 16, img_size: int = 224):
        self.df         = pd.read_csv(csv_path)
        self.num_frames = num_frames
        self.img_size   = img_size
        self.root       = Path(ROOT)

        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]
            ),
        ])

    def _load_frames(self, video_path: str) -> torch.Tensor:
        """
        Load a video and return a tensor of shape (num_frames, 3, H, W).
        If the video cannot be opened or has too few frames,
        black frames are used as fallback — training never crashes.
        """
        full_path = str(self.root / video_path)
        cap = cv2.VideoCapture(full_path)

        frames = []
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Uniformly sample indices across the video
            indices = np.linspace(0, max(total - 1, 0), self.num_frames, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    # BGR → RGB, then resize
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.img_size, self.img_size))
                    frames.append(frame)
                else:
                    # Use last valid frame or black frame
                    frames.append(frames[-1] if frames else
                                  np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
            cap.release()
        else:
            print(f"  [WARN] Cannot open video: {full_path}")

        # Pad with black frames if we got fewer than num_frames
        while len(frames) < self.num_frames:
            frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))

        # Apply transforms and stack → (num_frames, 3, H, W)
        tensor_frames = torch.stack([self.transform(f) for f in frames])
        return tensor_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        frames   = self._load_frames(row["video_path"])
        label_id = int(row["label_id"])
        return frames, label_id


# ── Training helpers ───────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for frames, labels in tqdm(loader, leave=False):
            frames = frames.to(device)
            labels = labels.to(device)

            logits = model(frames)
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
    # Directories
    ckpt_dir    = Path(ROOT) / cfg["paths"]["checkpoint_dir"]
    results_dir = Path(ROOT) / cfg["paths"]["results_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Datasets & loaders
    num_frames = cfg["data"]["num_frames"]
    img_size   = cfg["data"]["img_size"]
    batch_size = cfg["training"]["batch_size"]

    train_dataset = VideoDataset(Path(ROOT) / cfg["data"]["train_csv"], num_frames, img_size)
    val_dataset   = VideoDataset(Path(ROOT) / cfg["data"]["val_csv"],   num_frames, img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[INFO] Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")

    # Model
    model = RGBBaselineModel(
        num_classes=cfg["data"]["num_classes"],
        num_frames=num_frames,
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_transformer_layers=cfg["model"]["transformer_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )

    # Training loop
    epochs       = cfg["training"]["epochs"]
    best_val_acc = 0.0
    log_rows     = []

    print(f"\n[INFO] Starting training for {epochs} epochs …\n")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, training=False)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{epochs}  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Path(ROOT) / cfg["paths"]["best_model"])
            print(f"  ✔ Best model saved (val_acc={best_val_acc:.4f})")

    # Save training log CSV
    log_path = Path(ROOT) / cfg["paths"]["training_log"]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\n[INFO] Training complete. Best Val Acc: {best_val_acc:.4f}")
    print(f"[INFO] Training log saved → {log_path}")
