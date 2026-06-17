"""
attention_visualization.py
Week 5, Task 3 — Transformer Attention Visualization

The Transformer Encoder in RGBBaselineModel learns which video frames
are most important for the sign language prediction.

How temporal attention works:
  - Input: sequence of 16 frame embeddings (each 512-dim)
  - Self-attention: every frame attends to every other frame
  - Attention weight[i][j] = how much frame i looks at frame j
  - High weight = important relationship between frames i and j

This script:
  1. Loads the RGB Baseline model.
  2. For each test sample, extracts frame embeddings via the CNN.
  3. Runs them through the Transformer manually, capturing attention weights.
  4. Averages attention over all heads and layers.
  5. Visualizes per-frame importance as a bar chart.

Saves one plot per sample to results/attention_maps/.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader

from models.rgb_branch import RGBBaselineModel
from scripts.train     import VideoDataset

# ── Config ─────────────────────────────────────────────────────────────────────
with open(Path(ROOT) / "configs" / "baseline.yaml") as f:
    cfg = yaml.safe_load(f)

NUM_FRAMES  = cfg["data"]["num_frames"]
NUM_CLASSES = cfg["data"]["num_classes"]
CKPT_DIR    = Path(ROOT) / cfg["paths"]["checkpoint_dir"]
RESULTS_DIR = Path(ROOT) / cfg["paths"]["results_dir"] / "attention_maps"
TEST_CSV    = Path(ROOT) / cfg["data"]["test_csv"]
BATCH_SIZE  = 1   # process one video at a time for visualization

NUM_SAMPLES = 5   # how many test samples to visualize


# ── Attention extraction ───────────────────────────────────────────────────────

def extract_frame_embeddings(model: RGBBaselineModel,
                             frames: torch.Tensor,
                             device: torch.device) -> torch.Tensor:
    """
    Run the CNN on each frame to get per-frame embeddings.

    Args:
        frames : (1, T, C, H, W)
    Returns:
        emb    : (1, T, d_model) — projected and positional-encoded embeddings
    """
    B, T, C, H, W = frames.shape
    x = frames.view(B * T, C, H, W).to(device)

    with torch.no_grad():
        x = model.cnn(x)             # (B*T, 2048, 1, 1)
        x = x.view(B * T, -1)       # (B*T, 2048)
        x = model.input_proj(x)     # (B*T, d_model)
        x = x.view(B, T, -1)        # (B, T, d_model)
        x = x + model.pos_embedding[:, :T, :]   # add positional info

    return x   # (1, T, d_model)


def extract_attention_weights(model: RGBBaselineModel,
                              emb: torch.Tensor) -> list:
    """
    Run through Transformer Encoder layer-by-layer and capture
    attention weights (T x T matrix) from each layer.

    This calls self_attn with need_weights=True on the current hidden state,
    then runs the full layer for the next iteration.

    Args:
        emb : (1, T, d_model)  — output of CNN + projection + pos_emb
    Returns:
        attn_weights : list of (1, T, T) tensors, one per transformer layer
    """
    all_attn = []
    x = emb.clone()

    with torch.no_grad():
        for layer in model.transformer.layers:
            # ── Capture attention weights for this layer ───────────────────────
            # Call self_attn directly with need_weights=True
            # batch_first=True means (B, T, D) input
            _, attn_w = layer.self_attn(
                x, x, x,
                need_weights=True,
                average_attn_weights=True   # average over all heads → (B, T, T)
            )
            all_attn.append(attn_w.squeeze(0).cpu().numpy())   # (T, T)

            # ── Run the full layer for next iteration ──────────────────────────
            x = layer(x)

    return all_attn   # list of (T, T) arrays


def get_frame_importance(attn_weights: list) -> np.ndarray:
    """
    Summarize per-frame importance from all layers.

    Strategy: average all (T, T) attention matrices across layers,
    then take the column-sum (how much each frame is attended TO by others).

    Returns:
        importance : (T,) array, values normalized to [0, 1]
    """
    # Average attention maps across all layers: (T, T)
    avg_attn = np.mean(attn_weights, axis=0)

    # Column-sum: how much frame j is attended to = its "importance"
    importance = avg_attn.sum(axis=0)

    # Normalize to [0, 1]
    if importance.max() > 0:
        importance = (importance - importance.min()) / (importance.max() - importance.min())

    return importance   # (T,)


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_attention(frame_importance: np.ndarray,
                   attn_matrices: list,
                   sample_name: str,
                   true_label: str,
                   pred_label: str,
                   out_path: Path):
    """
    Create a 2-panel figure:
      Left : Bar chart — Frame Number vs Importance Score
      Right : Heatmap — Layer-averaged attention matrix (T x T)
    """
    T = len(frame_importance)
    avg_attn = np.mean(attn_matrices, axis=0)   # (T, T)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_style("whitegrid")

    # ── Left: Frame importance bar chart ──────────────────────────────────────
    frame_nums = list(range(1, T + 1))
    colors_bar = ["crimson" if imp == frame_importance.max()
                  else "steelblue" for imp in frame_importance]

    axes[0].bar(frame_nums, frame_importance, color=colors_bar, edgecolor="white")
    axes[0].set_xlabel("Frame Number", fontsize=12)
    axes[0].set_ylabel("Importance Score (normalized)", fontsize=12)
    axes[0].set_title("Frame-level Temporal Attention\n(red = most attended frame)",
                      fontsize=12, fontweight="bold")
    axes[0].set_xticks(frame_nums)
    axes[0].set_ylim(0, 1.1)

    # Annotate the peak frame
    peak_frame = int(np.argmax(frame_importance)) + 1
    axes[0].annotate(f"Peak: F{peak_frame}",
                     xy=(peak_frame, frame_importance.max()),
                     xytext=(peak_frame + 0.5, frame_importance.max() + 0.05),
                     fontsize=9, color="crimson",
                     arrowprops=dict(arrowstyle="->", color="crimson"))

    # ── Right: Attention matrix heatmap ────────────────────────────────────────
    sns.heatmap(avg_attn, ax=axes[1], cmap="Blues",
                xticklabels=[str(i) for i in frame_nums],
                yticklabels=[str(i) for i in frame_nums],
                cbar_kws={"shrink": 0.8})
    axes[1].set_xlabel("Key Frame (attended TO)", fontsize=11)
    axes[1].set_ylabel("Query Frame (attends FROM)", fontsize=11)
    axes[1].set_title("Self-Attention Matrix\n(avg across all transformer layers)",
                      fontsize=12, fontweight="bold")

    correct = "CORRECT" if true_label == pred_label else "WRONG"
    fig.suptitle(
        f"{sample_name}   |   True: {true_label}  ->  Pred: {pred_label}  [{correct}]",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")
    print(f"[INFO] Output : {RESULTS_DIR}\n")

    # Load model
    model = RGBBaselineModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / "best_model.pth", map_location=device))
    model.eval()

    # Load test data (1 video at a time)
    dataset = VideoDataset(TEST_CSV, NUM_FRAMES)
    test_df = pd.read_csv(TEST_CSV)
    label_map = dict(zip(test_df["label_id"], test_df["label"]))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_samples = min(NUM_SAMPLES, len(dataset))
    print(f"[INFO] Visualizing attention for {n_samples} samples...\n")

    for idx in range(n_samples):
        frames, label_id = dataset[idx]
        true_label = label_map.get(label_id, str(label_id))

        # Add batch dimension: (1, T, C, H, W)
        frames_batch = frames.unsqueeze(0).to(device)

        # Step 1: CNN → frame embeddings with positional encoding
        emb = extract_frame_embeddings(model, frames_batch, device)

        # Step 2: Transformer attention extraction
        attn_matrices = extract_attention_weights(model, emb)   # list of (T, T)

        # Step 3: Compute per-frame importance
        importance = get_frame_importance(attn_matrices)

        # Step 4: Get model prediction
        with torch.no_grad():
            logits   = model(frames_batch)
            pred_id  = logits.argmax(1).item()
        pred_label = label_map.get(pred_id, str(pred_id))

        # Step 5: Plot and save
        video_stem  = Path(test_df.iloc[idx]["video_path"]).stem
        sample_name = f"sample_{idx+1:02d}_{video_stem}"
        out_path    = RESULTS_DIR / f"{sample_name}.png"

        plot_attention(importance, attn_matrices,
                       sample_name, true_label, pred_label, out_path)

        peak_frame = int(np.argmax(importance)) + 1
        correct    = "OK" if true_label == pred_label else "WRONG"
        print(f"  [{idx+1}/{n_samples}] {video_stem}")
        print(f"    True={true_label:10s}  Pred={pred_label:10s}  [{correct}]  Peak frame: {peak_frame}")
        print(f"    Saved -> {out_path}")

    print(f"\n[INFO] Attention visualization complete. Results in {RESULTS_DIR}")
