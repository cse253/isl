"""
Dataset exploration script for Indian Sign Language Recognition.
Scans dataset, computes statistics, saves visualizations and CSV report.
"""

import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "Adjectives_1of8" / "Adjectives"
RESULTS_DIR  = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Scan dataset ────────────────────────────────────────────────────────────
def scan_dataset(root: Path) -> dict:
    """
    Recursively find every .MOV file.
    Groups files by their immediate parent folder (= class folder).
    Subfolders like 'Extra/' are intentionally skipped — only direct children
    of a class folder are counted as valid samples.
    """
    classes = {}
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        # Only .MOV files directly inside the class folder (not sub-subdirs)
        videos = sorted([f for f in class_dir.iterdir()
                         if f.is_file() and f.suffix.upper() == ".MOV"])
        if videos:
            classes[class_dir.name] = videos
    return classes

# ── 2. Video duration via OpenCV ───────────────────────────────────────────────
def get_duration(video_path: Path) -> float:
    """Return duration in seconds; -1 if the file cannot be opened."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return -1.0
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps > 0 else -1.0

# ── 3. Sample frame extraction ─────────────────────────────────────────────────
def get_middle_frame(video_path: Path):
    """Return the middle frame of a video as an RGB numpy array, or None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(total // 2, 0))
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None

# ── 4. Statistics ──────────────────────────────────────────────────────────────
def compute_stats(classes: dict) -> list:
    """
    For each class: count videos, compute min/max/avg duration.
    Returns a list of dicts (one per class).
    """
    rows = []
    for cls_name, videos in classes.items():
        durations = []
        for v in videos:
            d = get_duration(v)
            if d >= 0:
                durations.append(d)
            else:
                print(f"  [WARN] Could not read: {v.name}")

        rows.append({
            "class":        cls_name,
            "video_count":  len(videos),
            "min_duration": round(min(durations), 3) if durations else 0,
            "max_duration": round(max(durations), 3) if durations else 0,
            "avg_duration": round(np.mean(durations), 3) if durations else 0,
        })
    return rows

# ── 5. Print summary ───────────────────────────────────────────────────────────
def print_summary(rows: list):
    total = sum(r["video_count"] for r in rows)
    print(f"\n{'='*55}")
    print(f"  ISL Dataset — Adjectives (8 classes)")
    print(f"{'='*55}")
    print(f"  {'Class':<20} {'Videos':>6}  {'Min':>6}  {'Max':>6}  {'Avg':>6}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for r in rows:
        print(f"  {r['class']:<20} {r['video_count']:>6}  "
              f"{r['min_duration']:>6.2f}s  {r['max_duration']:>6.2f}s  {r['avg_duration']:>6.2f}s")
    print(f"  {'─'*49}")
    print(f"  {'TOTAL':<20} {total:>6}")
    print(f"{'='*55}\n")

# ── 6. Save CSV ────────────────────────────────────────────────────────────────
def save_csv(rows: list, path: Path):
    fieldnames = ["class", "video_count", "min_duration", "max_duration", "avg_duration"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV saved  → {path}")

# ── 7. Save class distribution bar chart ──────────────────────────────────────
def save_distribution_plot(rows: list, path: Path):
    labels = [r["class"] for r in rows]
    counts = [r["video_count"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts, color="steelblue", edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title("ISL Adjectives — Video Count per Class", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Videos")
    ax.set_ylim(0, max(counts) + 5)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Plot saved → {path}")

# ── 8. Display sample frames ───────────────────────────────────────────────────
def show_sample_frames(classes: dict):
    """Display one middle frame per class in a grid (non-blocking)."""
    n = len(classes)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes = axes.flatten()

    for idx, (cls_name, videos) in enumerate(classes.items()):
        frame = get_middle_frame(videos[0])
        if frame is not None:
            axes[idx].imshow(frame)
        else:
            axes[idx].text(0.5, 0.5, "No frame", ha="center", va="center")
        axes[idx].set_title(cls_name, fontsize=9)
        axes[idx].axis("off")

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Sample Frame per Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[INFO] Scanning: {DATASET_ROOT}")
    classes = scan_dataset(DATASET_ROOT)

    if not classes:
        print("[ERROR] No class folders with .MOV files found. Check DATASET_ROOT.")
        raise SystemExit(1)

    print(f"[INFO] Found {len(classes)} classes, computing durations …")
    rows = compute_stats(classes)

    print_summary(rows)
    save_csv(rows,  RESULTS_DIR / "dataset_statistics.csv")
    save_distribution_plot(rows, RESULTS_DIR / "class_distribution.png")
    show_sample_frames(classes)
