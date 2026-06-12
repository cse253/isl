"""
split_dataset.py
Scans the dataset, builds a full CSV manifest, then splits into
train (80%) / val (10%) / test (10%) with stratification.
"""

import os
import re
import random
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Paths (relative to project root) ──────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "Adjectives_1of8" / "Adjectives"
SPLITS_DIR  = ROOT / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTENSIONS = {".mov", ".mp4", ".avi"}

# ── 1. Scan and build records ──────────────────────────────────────────────────
def clean_label(folder_name: str) -> str:
    """Remove leading number+dot prefix: '1. loud' → 'loud'."""
    return re.sub(r"^\d+\.\s*", "", folder_name).strip()

def scan(dataset_dir: Path) -> pd.DataFrame:
    records = []
    label_id_map = {}
    current_id = 0

    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = clean_label(class_dir.name)

        # Assign a numeric ID to each unique label (ordered by folder name)
        if label not in label_id_map:
            label_id_map[label] = current_id
            current_id += 1

        # Only direct children — skip subfolders like Extra/
        for f in sorted(class_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
                # Store path relative to project root for portability
                records.append({
                    "video_path": str(f.relative_to(ROOT)),
                    "label":      label,
                    "label_id":   label_id_map[label],
                })

    if not records:
        raise FileNotFoundError(f"No videos found under: {dataset_dir}")

    return pd.DataFrame(records), label_id_map

# ── 2. Split ───────────────────────────────────────────────────────────────────
def split(df: pd.DataFrame, seed: int = 42):
    """
    80 / 10 / 10 split with stratification.
    Falls back to a plain random split if any class has fewer than 3 samples
    (stratification requires at least 1 sample in every subset).
    """
    try:
        train_df, temp_df = train_test_split(
            df, test_size=0.2, random_state=seed, stratify=df["label_id"]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=seed, stratify=temp_df["label_id"]
        )
        print("[INFO] Stratified split succeeded.")
    except ValueError as e:
        print(f"[WARN] Stratified split failed ({e}). Using random split.")
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed)
        val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=seed)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

# ── 3. Print distribution ──────────────────────────────────────────────────────
def print_distribution(name: str, df: pd.DataFrame):
    print(f"\n  {name} ({len(df)} samples):")
    counts = df.groupby("label")["video_path"].count()
    for label, count in counts.items():
        print(f"    {label:<15} {count}")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[INFO] Scanning: {DATASET_DIR}")
    df, label_id_map = scan(DATASET_DIR)

    print(f"\n[INFO] Total videos : {len(df)}")
    print(f"[INFO] Classes      : {len(label_id_map)}")
    print(f"[INFO] Label map    : {label_id_map}")

    train_df, val_df, test_df = split(df)

    # Save CSVs
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR  / "val.csv",   index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv",  index=False)

    print(f"\n[INFO] Splits saved to: {SPLITS_DIR}")
    print(f"  train.csv → {len(train_df)} samples")
    print(f"  val.csv   → {len(val_df)} samples")
    print(f"  test.csv  → {len(test_df)} samples")

    print_distribution("train.csv", train_df)
    print_distribution("val.csv",   val_df)
    print_distribution("test.csv",  test_df)
