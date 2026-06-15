"""
extract_rgb_embeddings.py
Runs ResNet50 once on all videos and saves per-frame CNN features as .npy files.

Instead of running ResNet50 every training batch (slow),
we extract features once and reuse them (fast).

Output shape per video: (16, 2048)
  - 16 uniformly sampled frames
  - 2048 ResNet50 feature vector per frame (before FC layer)

Saved to: datasets/rgb_embeddings/<class_name>/<video_stem>.npy
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import re
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR = Path(ROOT) / "datasets" / "Adjectives_1of8" / "Adjectives"
OUTPUT_DIR  = Path(ROOT) / "datasets" / "rgb_embeddings"
NUM_FRAMES  = 16
IMG_SIZE    = 224
VALID_EXT   = {".mov", ".mp4", ".avi", ".MOV"}

# ── Build ResNet50 feature extractor (no FC layer) ────────────────────────────
def build_extractor(device: torch.device) -> nn.Module:
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove final FC — output is (batch, 2048, 1, 1) after avgpool
    extractor = nn.Sequential(*list(backbone.children())[:-1])
    extractor.eval()
    return extractor.to(device)

# ── Image preprocessing ────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])

# ── Helpers ────────────────────────────────────────────────────────────────────
def clean_label(folder_name: str) -> str:
    """'1. loud' → 'loud'"""
    return re.sub(r"^\d+\.\s*", "", folder_name).strip()


def extract_video(video_path: Path, extractor: nn.Module, device: torch.device) -> np.ndarray:
    """
    Uniformly sample NUM_FRAMES from video.
    Pass each frame through ResNet50 CNN.
    Returns array of shape (NUM_FRAMES, 2048).
    """
    cap      = cv2.VideoCapture(str(video_path))
    features = np.zeros((NUM_FRAMES, 2048), dtype=np.float32)

    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path.name}")
        return features

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), NUM_FRAMES, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(transform(frame))
        else:
            # Duplicate last valid frame or use zeros
            frames.append(frames[-1] if frames else torch.zeros(3, IMG_SIZE, IMG_SIZE))

    cap.release()

    # Pad if needed
    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1] if frames else torch.zeros(3, IMG_SIZE, IMG_SIZE))

    # Stack → (NUM_FRAMES, 3, 224, 224) and run through CNN
    batch = torch.stack(frames).to(device)          # (16, 3, 224, 224)
    with torch.no_grad():
        out = extractor(batch)                       # (16, 2048, 1, 1)
    features = out.view(NUM_FRAMES, -1).cpu().numpy()  # (16, 2048)
    return features


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device     : {device}")
    print(f"[INFO] Scanning   : {DATASET_DIR}")
    print(f"[INFO] Output     : {OUTPUT_DIR}")
    print(f"[INFO] Frames     : {NUM_FRAMES}  |  Feature dim: 2048\n")

    extractor = build_extractor(device)
    total_processed = 0

    for class_dir in sorted(DATASET_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        label     = clean_label(class_dir.name)
        out_class = OUTPUT_DIR / label
        out_class.mkdir(parents=True, exist_ok=True)

        videos = [f for f in class_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in {e.lower() for e in VALID_EXT}]

        print(f"[CLASS] {class_dir.name}  ({len(videos)} videos)")

        for video_path in sorted(videos):
            out_path = out_class / (video_path.stem + ".npy")

            if out_path.exists():
                print(f"  [SKIP] {video_path.name} already extracted")
                total_processed += 1
                continue

            features = extract_video(video_path, extractor, device)
            np.save(str(out_path), features)
            total_processed += 1
            print(f"  [DONE] {video_path.name} → {out_path.name}  shape={features.shape}")

    print(f"\n{'='*55}")
    print(f"  RGB embedding extraction complete")
    print(f"  Total processed : {total_processed}")
    print(f"  Saved to        : {OUTPUT_DIR}")
    print(f"{'='*55}")
