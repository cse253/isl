"""
evaluate.py
Loads the best trained model and evaluates it on the test set.
Saves confusion matrix and classification report.
"""

import os
import sys

# Fix import path — add project root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from models.rgb_branch import RGBBaselineModel

# Reuse VideoDataset from train.py
from scripts.train import VideoDataset

# ── Load config ────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(ROOT) / "configs" / "baseline.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results_dir = Path(ROOT) / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Test dataset
    test_dataset = VideoDataset(
        Path(ROOT) / cfg["data"]["test_csv"],
        num_frames=cfg["data"]["num_frames"],
        img_size=cfg["data"]["img_size"],
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"],
                             shuffle=False, num_workers=0)
    print(f"[INFO] Test samples: {len(test_dataset)}")

    # Load model
    model = RGBBaselineModel(
        num_classes=cfg["data"]["num_classes"],
        num_frames=cfg["data"]["num_frames"],
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_transformer_layers=cfg["model"]["transformer_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    ckpt_path = Path(ROOT) / cfg["paths"]["best_model"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nRun train.py first.")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")

    # Inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(device)
            preds  = model(frames).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Top-1 accuracy
    accuracy = (all_preds == all_labels).mean()
    print(f"\n[RESULT] Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Build full class name list ordered by label_id from train CSV
    # (test set may not contain all 8 classes, so we must not derive names from test CSV alone)
    import pandas as pd
    train_df    = pd.read_csv(Path(ROOT) / cfg["data"]["train_csv"])
    label_map   = (train_df[["label", "label_id"]]
                   .drop_duplicates()
                   .sort_values("label_id"))
    class_names = label_map["label"].tolist()          # all 8 classes, correctly ordered

    # actual label IDs present in predictions / ground truth
    present_ids = sorted(set(all_labels) | set(all_preds))
    present_names = [class_names[i] for i in present_ids]

    # Classification report
    report      = classification_report(all_labels, all_preds,
                                        labels=present_ids,
                                        target_names=present_names,
                                        zero_division=0)
    report_path = Path(ROOT) / cfg["paths"]["classification_report"]
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    print(f"[INFO] Report saved  → {report_path}")

    # Confusion matrix — only rows/cols for classes present in test set
    cm = confusion_matrix(all_labels, all_preds, labels=present_ids)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=present_names, yticklabels=present_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()

    cm_path = Path(ROOT) / cfg["paths"]["confusion_matrix"]
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved → {cm_path}")
