"""
train_adaptive.py
Trains AdaptiveMultiModalModel using precomputed .npy embeddings.
Much faster than loading raw videos — no ResNet50 at training time.

Saves:
  checkpoints/best_adaptive_model.pth
  results/adaptive_training_log.csv
"""

import os
import sys
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.adaptive_model import AdaptiveMultiModalModel
# EmbeddingDataset — loads precomputed RGB+Pose .npy files
from scripts.train_fusion import FusionDataset

# ── Config ─────────────────────────────────────────────────────────────────────
with open(Path(ROOT) / "configs" / "baseline.yaml") as f:
    cfg = yaml.safe_load(f)

NUM_FRAMES  = cfg["data"]["num_frames"]
NUM_CLASSES = cfg["data"]["num_classes"]
EPOCHS      = cfg["training"]["epochs"]
BATCH_SIZE  = cfg["training"]["batch_size"]
LR          = cfg["training"]["lr"]
WD          = cfg["training"]["weight_decay"]
CKPT_DIR    = Path(ROOT) / cfg["paths"]["checkpoint_dir"]
RESULTS_DIR = Path(ROOT) / cfg["paths"]["results_dir"]


# ── Epoch runner ───────────────────────────────────────────────────────────────
def run_epoch(
    model: AdaptiveMultiModalModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    training: bool,
) -> tuple[float, float, float, float]:
    """
    Runs one full epoch.
    Returns: loss, accuracy, avg_alpha, avg_beta
    """
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    alpha_sum, beta_sum, batches = 0.0, 0.0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for frames, pose, labels in tqdm(loader, leave=False):
            frames = frames.to(device)
            pose   = pose.to(device)
            labels = labels.to(device)

            logits, alpha, beta = model(frames, pose)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

            # Track average gate weights across batch
            alpha_sum += alpha.mean().item()
            beta_sum  += beta.mean().item()
            batches   += 1

    avg_loss  = total_loss / total
    avg_acc   = correct / total
    avg_alpha = alpha_sum / batches
    avg_beta  = beta_sum  / batches
    return avg_loss, avg_acc, avg_alpha, avg_beta


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check if checkpoint already exists to save time on CPU
    ckpt_path = CKPT_DIR / "best_adaptive_model.pth"
    if ckpt_path.exists() and "--force" not in sys.argv:
        print(f"[INFO] Pre-trained model found at {ckpt_path.relative_to(ROOT)}. Skipping training (use --force to retrain).")
        sys.exit(0)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device : {device}")

    # Datasets
    train_loader = DataLoader(
        FusionDataset(Path(ROOT) / cfg["data"]["train_csv"]),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        FusionDataset(Path(ROOT) / cfg["data"]["val_csv"]),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    print(f"[INFO] Train: {len(train_loader.dataset)}  |  Val: {len(val_loader.dataset)}\n")

    # Model
    model     = AdaptiveMultiModalModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    log_rows     = []

    print(f"[INFO] Starting adaptive fusion training for {EPOCHS} epochs ...\n")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_alpha, tr_beta = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        vl_loss, vl_acc, vl_alpha, vl_beta = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  "
            f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.4f}  "
            f"α(RGB): {vl_alpha:.4f}  β(Pose): {vl_beta:.4f}"
        )

        log_rows.append({
            "epoch":      epoch,
            "train_loss": round(tr_loss,  6),
            "val_loss":   round(vl_loss,  6),
            "train_acc":  round(tr_acc,   6),
            "val_acc":    round(vl_acc,   6),
            "avg_alpha":  round(vl_alpha, 6),
            "avg_beta":   round(vl_beta,  6),
        })

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), CKPT_DIR / "best_adaptive_model.pth")
            print(f"  [OK] Best model saved (val_acc={best_val_acc:.4f})")

    # Save training log
    log_path = RESULTS_DIR / "adaptive_training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\n[INFO] Training complete. Best Val Acc : {best_val_acc:.4f}")
    print(f"[INFO] Log saved -> {log_path}")
