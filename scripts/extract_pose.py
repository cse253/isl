"""
extract_pose.py
Extracts pose, left hand, and right hand landmarks from every video
using MediaPipe Holistic. Saves one .npy file per video.

Landmark breakdown per frame:
  pose:       33 × 4 (x, y, z, visibility) = 132
  left hand:  21 × 3 (x, y, z)             =  63
  right hand: 21 × 3 (x, y, z)             =  63
  Total per frame                           = 258

Output shape per file: (16, 258)
Saved to: datasets/pose_data/<class_name>/<video_stem>.npy
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import re
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR  = Path(ROOT) / "datasets" / "Adjectives_1of8" / "Adjectives"
OUTPUT_DIR   = Path(ROOT) / "datasets" / "pose_data"
NUM_FRAMES   = 16
LANDMARK_DIM = 258   # 33*4 + 21*3 + 21*3

VALID_EXT = {".mov", ".mp4", ".avi", ".MOV"}

# ── Helpers ────────────────────────────────────────────────────────────────────
def clean_label(folder_name: str) -> str:
    """'1. loud' → 'loud'"""
    return re.sub(r"^\d+\.\s*", "", folder_name).strip()


def extract_landmarks(results) -> np.ndarray:
    """
    Pull landmark arrays from a MediaPipe Holistic result object.
    Returns a flat 1-D array of length 258.
    If a landmark group is missing (e.g. hand not visible), returns zeros.
    """
    # Pose: 33 landmarks × (x, y, z, visibility)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in results.pose_landmarks.landmark], dtype=np.float32)
    else:
        pose = np.zeros((33, 4), dtype=np.float32)

    # Left hand: 21 landmarks × (x, y, z)
    if results.left_hand_landmarks:
        left = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
    else:
        left = np.zeros((21, 3), dtype=np.float32)

    # Right hand: 21 landmarks × (x, y, z)
    if results.right_hand_landmarks:
        right = np.array([[lm.x, lm.y, lm.z]
                          for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
    else:
        right = np.zeros((21, 3), dtype=np.float32)

    return np.concatenate([pose.flatten(), left.flatten(), right.flatten()])  # (258,)


def process_video(video_path: Path, holistic) -> np.ndarray:
    """
    Uniformly sample NUM_FRAMES from video, run MediaPipe on each frame.
    Returns array of shape (NUM_FRAMES, 258).
    Falls back to zero vector for unreadable frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    sequence = np.zeros((NUM_FRAMES, LANDMARK_DIM), dtype=np.float32)

    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path.name}")
        return sequence

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), NUM_FRAMES, dtype=int)

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # Keep zero row for this frame
            continue

        # MediaPipe requires RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False          # slight perf boost
        results = holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True

        sequence[i] = extract_landmarks(results)

    cap.release()
    return sequence


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp_holistic = mp.solutions.holistic

    total_videos   = 0
    skipped_videos = 0

    print(f"[INFO] Scanning : {DATASET_DIR}")
    print(f"[INFO] Output   : {OUTPUT_DIR}")
    print(f"[INFO] Frames   : {NUM_FRAMES}  |  Landmark dim: {LANDMARK_DIM}\n")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as holistic:

        for class_dir in sorted(DATASET_DIR.iterdir()):
            if not class_dir.is_dir():
                continue

            label      = clean_label(class_dir.name)
            out_class  = OUTPUT_DIR / label
            out_class.mkdir(parents=True, exist_ok=True)

            videos = [f for f in class_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in VALID_EXT]

            print(f"[CLASS] {class_dir.name}  ({len(videos)} videos)")

            for video_path in sorted(videos):
                out_path = out_class / (video_path.stem + ".npy")

                # Skip if already extracted
                if out_path.exists():
                    print(f"  [SKIP] {video_path.name} already extracted")
                    total_videos += 1
                    continue

                sequence = process_video(video_path, holistic)

                # Warn if all-zero (MediaPipe detected nothing in any frame)
                if sequence.sum() == 0:
                    print(f"  [WARN] No landmarks detected: {video_path.name}")
                    skipped_videos += 1

                np.save(str(out_path), sequence)
                total_videos += 1
                print(f"  [DONE] {video_path.name} → {out_path.name}  shape={sequence.shape}")

    print(f"\n{'='*55}")
    print(f"  Extraction complete")
    print(f"  Total processed : {total_videos}")
    print(f"  Zero-landmark   : {skipped_videos}  (saved as zeros, check these videos)")
    print(f"  Saved to        : {OUTPUT_DIR}")
    print(f"{'='*55}")
