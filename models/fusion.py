"""
fusion.py
Two multi-modal fusion strategies combining RGB and Pose branches.

LateFusionModel:
  - Runs RGB and Pose branches independently
  - Averages their softmax outputs at inference time
  - No joint training needed — uses pretrained branch weights

FeatureConcatFusionModel:
  - Extracts feature vectors from both branches (before their classifiers)
  - Concatenates: 512-dim RGB + 256-dim Pose = 768-dim
  - Passes through a shared MLP → 8 classes
  - Trained end-to-end jointly
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from models.rgb_branch import RGBBaselineModel
from models.pose_branch import PoseBranchModel


# ── Late Fusion ────────────────────────────────────────────────────────────────
class LateFusionModel(nn.Module):
    """
    Averages softmax probabilities from RGB and Pose branches.
    Both branches are loaded with pretrained weights and frozen by default.
    No additional training required.
    """

    def __init__(self, rgb_model: RGBBaselineModel, pose_model: PoseBranchModel):
        super().__init__()
        self.rgb_model  = rgb_model
        self.pose_model = pose_model
        self.softmax    = nn.Softmax(dim=1)

    def forward(self, frames: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames : (batch, T, C, H, W)  — RGB frames
            pose   : (batch, T, 258)       — landmark sequences
        Returns:
            logits : (batch, num_classes)  — averaged probabilities as logits
        """
        rgb_probs  = self.softmax(self.rgb_model(frames))   # (B, 8)
        pose_probs = self.softmax(self.pose_model(pose))    # (B, 8)
        return (rgb_probs + pose_probs) / 2                 # (B, 8)


# ── Feature Concatenation Fusion ───────────────────────────────────────────────
class FeatureConcatFusionModel(nn.Module):
    """
    Extracts penultimate features from both branches, concatenates,
    and learns a joint classifier MLP. Trained end-to-end.

    RGB branch  → 512-dim feature
    Pose branch → 256-dim feature
    Concat      → 768-dim → MLP → 8 classes
    """

    def __init__(
        self,
        num_classes: int = 8,
        num_frames: int = 16,
        rgb_d_model: int = 512,
        pose_d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── RGB encoder (ResNet50 + Transformer, no classifier head) ──────────
        _rgb = RGBBaselineModel(
            num_classes=num_classes,
            num_frames=num_frames,
            d_model=rgb_d_model,
        )
        self.rgb_cnn         = _rgb.cnn
        self.rgb_input_proj  = _rgb.input_proj
        self.rgb_pos_emb     = _rgb.pos_embedding
        self.rgb_transformer = _rgb.transformer

        # ── Pose encoder (Transformer, no classifier head) ────────────────────
        _pose = PoseBranchModel(
            num_classes=num_classes,
            num_frames=num_frames,
            d_model=pose_d_model,
        )
        self.pose_input_proj  = _pose.input_proj
        self.pose_pos_emb     = _pose.pos_embedding
        self.pose_transformer = _pose.transformer

        # ── Fusion MLP ─────────────────────────────────────────────────────────
        fused_dim = rgb_d_model + pose_d_model   # 512 + 256 = 768
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _encode_rgb(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim feature vector from RGB frames."""
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        x = self.rgb_cnn(x).view(B * T, -1)        # (B*T, 2048)
        x = self.rgb_input_proj(x).view(B, T, -1)  # (B, T, 512)
        x = x + self.rgb_pos_emb[:, :T, :]
        x = self.rgb_transformer(x)                 # (B, T, 512)
        return x.mean(dim=1)                        # (B, 512)

    def _encode_pose(self, pose: torch.Tensor) -> torch.Tensor:
        """Extract 256-dim feature vector from pose sequence."""
        x = self.pose_input_proj(pose)              # (B, T, 256)
        x = x + self.pose_pos_emb[:, :x.size(1), :]
        x = self.pose_transformer(x)                # (B, T, 256)
        return x.mean(dim=1)                        # (B, 256)

    def forward(self, frames: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames : (batch, T, C, H, W)
            pose   : (batch, T, 258)
        Returns:
            logits : (batch, num_classes)
        """
        rgb_feat  = self._encode_rgb(frames)        # (B, 512)
        pose_feat = self._encode_pose(pose)         # (B, 256)
        fused     = torch.cat([rgb_feat, pose_feat], dim=1)  # (B, 768)
        return self.fusion_mlp(fused)               # (B, 8)


# ── Sanity test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T = 2, 16
    dummy_frames = torch.randn(B, T, 3, 224, 224)
    dummy_pose   = torch.randn(B, T, 258)

    # Test Late Fusion
    print("[TEST] LateFusionModel ...")
    rgb_model  = RGBBaselineModel(num_classes=8, num_frames=T)
    pose_model = PoseBranchModel(input_dim=258, num_classes=8, num_frames=T)
    late_model = LateFusionModel(rgb_model, pose_model)
    late_model.eval()
    with torch.no_grad():
        out = late_model(dummy_frames, dummy_pose)
    print(f"  Output shape : {out.shape}")
    assert out.shape == torch.Size([B, 8])
    print("  PASSED\n")

    # Test Feature Concat Fusion
    print("[TEST] FeatureConcatFusionModel ...")
    concat_model = FeatureConcatFusionModel(num_classes=8, num_frames=T)
    concat_model.eval()
    with torch.no_grad():
        out = concat_model(dummy_frames, dummy_pose)
    print(f"  Output shape : {out.shape}")
    assert out.shape == torch.Size([B, 8])
    print("  PASSED")
