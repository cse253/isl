"""
grad_cam.py
Week 5, Task 2 — Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) shows which spatial
regions of an input image the model focuses on when making a prediction.

How it works:
  1. Forward pass: get the model's prediction for a given input.
  2. Backward pass: compute the gradient of the predicted class score
     with respect to the LAST CONVOLUTIONAL LAYER's activations (layer4).
  3. Global average pool the gradients across spatial dimensions.
  4. Multiply pooled gradients by the activation map channel-wise.
  5. Apply ReLU (keep only positive influence regions).
  6. Resize and overlay as a heatmap on the original frame.

Target layer: model.cnn[7]  (ResNet50's layer4 — last conv block)
  Activation map shape: (2048, 7, 7)

Saves per sample in results/gradcam/<video_name>/:
  original_frame.png  — raw video frame
  heatmap.png         — Grad-CAM heatmap (colorized)
  overlay.png         — heatmap blended onto original frame
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torchvision.transforms as T
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
RESULTS_DIR = Path(ROOT) / cfg["paths"]["results_dir"] / "gradcam"
TEST_CSV    = Path(ROOT) / cfg["data"]["test_csv"]
IMG_SIZE    = cfg["data"]["img_size"]

# ImageNet normalization (same as VideoDataset)
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])

# Number of test samples to process
NUM_SAMPLES = 5


# ── Grad-CAM Hook System ───────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM implementation using forward/backward hooks.

    Usage:
        cam = GradCAM(model, target_layer=model.cnn[7])
        heatmap = cam.generate(frame_tensor, class_idx)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model  = model
        self.grads  = None   # Will store gradients w.r.t. target layer
        self.acts   = None   # Will store activations from target layer

        # Register hooks on the target layer
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Hook: save the layer's forward activations."""
        self.acts = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook: save the gradients flowing back through the layer."""
        self.grads = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for one frame.

        Args:
            x          : (1, C, H, W) — single frame tensor, requires_grad
            class_idx  : class index to explain
        Returns:
            heatmap    : (H, W) numpy array, values in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward through just the CNN part to get spatial features
        # (We need gradients, so don't use torch.no_grad())
        x = x.requires_grad_(True)
        acts_before_pool = None

        # Run up to layer4 to get (1, 2048, 7, 7) activations
        feat = x
        for i, layer in enumerate(self.model.cnn.children()):
            feat = layer(feat)
            if i == 7:   # layer4 — stop here to get spatial map
                acts_before_pool = feat  # (1, 2048, 7, 7)
                break

        # Global average pool to (1, 2048)
        pooled = acts_before_pool.mean(dim=[2, 3])

        # Project to d_model and classify (simplified single-frame path)
        proj = self.model.input_proj(pooled)     # (1, d_model)

        # Add positional embedding for frame 0
        proj = proj + self.model.pos_embedding[:, 0, :]

        # Mean pool (single frame, no transformer needed for GradCAM)
        logits = self.model.classifier(proj)     # (1, num_classes)

        # Backward: compute gradient of class score w.r.t. layer4 activations
        score = logits[0, class_idx]
        score.backward()

        # Get gradients and activations at layer4: (1, 2048, 7, 7)
        grads = self.grads          # (1, 2048, 7, 7)
        acts  = self.acts           # (1, 2048, 7, 7)

        # Global average pool gradients: (2048,)
        weights = grads.mean(dim=[2, 3])[0]   # (2048,)

        # Weighted combination of activation channels: (7, 7)
        cam = torch.zeros(acts.shape[2:], device=x.device)
        for k, w in enumerate(weights):
            cam += w * acts[0, k]

        # ReLU: keep only positive activations
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam.detach().cpu().numpy()   # (7, 7)


# ── Frame loading ─────────────────────────────────────────────────────────────

def load_raw_frame(video_path: str, frame_idx: int = 0) -> np.ndarray:
    """
    Load a single raw RGB frame from a video file.
    Returns numpy array (H, W, 3) uint8.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Pick the frame at the given index position
    actual_idx = int(np.linspace(0, max(total - 1, 0), NUM_FRAMES)[frame_idx])
    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    return frame


transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def frame_to_tensor(frame_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert raw numpy frame to model-ready tensor (1, C, H, W)."""
    return transform(frame_np).unsqueeze(0).to(device)  # (1, 3, 224, 224)


# ── Overlay helper ─────────────────────────────────────────────────────────────

def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """
    Blend Grad-CAM heatmap onto the original frame.

    Args:
        frame   : (H, W, 3) uint8 original frame
        heatmap : (H, W) float in [0, 1]
        alpha   : blending weight for heatmap
    Returns:
        blended : (H, W, 3) uint8
    """
    # Resize heatmap to frame size
    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

    # Apply colormap (jet: blue=cold=low, red=hot=high)
    colormap = cm.jet(heatmap_resized)[:, :, :3]   # (H, W, 3) float [0,1]
    colormap = (colormap * 255).astype(np.uint8)

    # Blend
    blended = (frame * (1 - alpha) + colormap * alpha).astype(np.uint8)
    return blended, colormap


def save_visualizations(out_dir: Path, frame: np.ndarray, heatmap: np.ndarray,
                        sample_name: str, true_label: str, pred_label: str):
    """Save original frame, colorized heatmap, and overlay side-by-side."""
    out_dir.mkdir(parents=True, exist_ok=True)

    blended, colormap = overlay_heatmap(frame, heatmap, alpha=0.4)

    # Save individual files
    plt.imsave(str(out_dir / "original_frame.png"), frame)
    plt.imsave(str(out_dir / "heatmap.png"),
               cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE)), cmap="jet")
    plt.imsave(str(out_dir / "overlay.png"), blended)

    # Save combined figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(frame);    axes[0].set_title("Original Frame");   axes[0].axis("off")
    axes[1].imshow(colormap); axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
    axes[2].imshow(blended);  axes[2].set_title("Overlay");          axes[2].axis("off")

    correct = "CORRECT" if true_label == pred_label else "WRONG"
    fig.suptitle(f"{sample_name}\nTrue: {true_label}  |  Pred: {pred_label}  |  {correct}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(out_dir / "combined.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")
    print(f"[INFO] Output : {RESULTS_DIR}\n")

    # Load RGB Baseline model
    model = RGBBaselineModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    model.load_state_dict(torch.load(CKPT_DIR / "best_model.pth", map_location=device))
    model.eval()

    # The target layer for Grad-CAM: ResNet50 Layer4 (index 7 in model.cnn children)
    target_layer = list(model.cnn.children())[7]   # layer4
    grad_cam = GradCAM(model, target_layer)

    # Load test CSV for sample info
    test_df = pd.read_csv(TEST_CSV)

    # Label name lookup (label_id -> label name)
    label_map = dict(zip(test_df["label_id"], test_df["label"]))

    # Select samples (up to NUM_SAMPLES)
    samples = test_df.head(NUM_SAMPLES)

    print(f"[INFO] Processing {len(samples)} test samples ...\n")

    for i, (_, row) in enumerate(samples.iterrows()):
        video_path = str(Path(ROOT) / row["video_path"])
        true_label = row["label"]
        true_id    = int(row["label_id"])

        print(f"  [{i+1}/{len(samples)}] {Path(video_path).name}  (label={true_label})")

        # Pick the 8th frame (middle of 16-frame clip) for visualization
        frame_np = load_raw_frame(video_path, frame_idx=8)
        if frame_np is None:
            print(f"    [WARN] Could not load frame, skipping.")
            continue

        # Convert to tensor
        frame_t = frame_to_tensor(frame_np, device)  # (1, 3, 224, 224)

        # Get model prediction (full forward with all frames for context)
        # First get the prediction using the single-frame path
        with torch.no_grad():
            feat_full = model.cnn(frame_t)                 # (1, 2048, 1, 1)
            feat_flat = feat_full.view(1, -1)              # (1, 2048)
            proj      = model.input_proj(feat_flat)        # (1, 512)
            proj      = proj + model.pos_embedding[:, 0, :]
            logits    = model.classifier(proj)             # (1, num_classes)
            pred_id   = logits.argmax(1).item()

        pred_label = label_map.get(pred_id, str(pred_id))

        # Generate Grad-CAM heatmap for the PREDICTED class
        heatmap = grad_cam.generate(frame_t.clone(), class_idx=pred_id)   # (7, 7)

        # Save visualizations
        sample_name = f"sample_{i+1:02d}_{Path(video_path).stem}"
        out_dir     = RESULTS_DIR / sample_name
        save_visualizations(out_dir, frame_np, heatmap,
                            sample_name, true_label, pred_label)

        correct = "OK" if true_label == pred_label else "WRONG"
        print(f"    True={true_label:10s}  Pred={pred_label:10s}  [{correct}]")
        print(f"    Saved -> {out_dir}")

    print(f"\n[INFO] Grad-CAM complete. Results in {RESULTS_DIR}")
