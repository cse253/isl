"""
adaptive_model.py
Adaptive Multi-Modal Fusion Model — uses precomputed embeddings for speed.

Pipeline:
  RGB emb  (B,T,2048) → Linear proj + Transformer → rgb_feat  (B,512)
  Pose seq (B,T,258)  → Linear proj + Transformer → pose_feat (B,512)
  rgb_feat + pose_feat → AdaptiveFusion → fused (B,512) + alpha, beta
  fused → Linear → logits (B, num_classes)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

from models.adaptive_fusion import AdaptiveFusion


class AdaptiveMultiModalModel(nn.Module):
    """
    Accepts precomputed RGB embeddings (B,T,2048) and Pose sequences (B,T,258).
    No ResNet50 at training time — embeddings are loaded from .npy files.
    """

    def __init__(
        self,
        num_classes: int            = 8,
        num_frames: int             = 16,
        rgb_input_dim: int          = 2048,
        rgb_d_model: int            = 512,
        pose_input_dim: int         = 258,
        pose_d_model: int           = 256,
        nhead_rgb: int              = 8,
        nhead_pose: int             = 4,
        num_transformer_layers: int = 4,
        dropout: float              = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames

        # ── RGB Branch (no CNN — accepts precomputed features) ─────────────────
        self.rgb_proj    = nn.Linear(rgb_input_dim, rgb_d_model)   # 2048 → 512
        self.rgb_pos_emb = nn.Parameter(torch.randn(1, num_frames, rgb_d_model))
        self.rgb_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=rgb_d_model, nhead=nhead_rgb,
                dropout=dropout, batch_first=True,
            ),
            num_layers=num_transformer_layers,
        )

        # ── Pose Branch ────────────────────────────────────────────────────────
        self.pose_proj    = nn.Linear(pose_input_dim, pose_d_model)  # 258 → 256
        self.pose_pos_emb = nn.Parameter(torch.randn(1, num_frames, pose_d_model))
        self.pose_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=pose_d_model, nhead=nhead_pose,
                dropout=dropout, batch_first=True,
            ),
            num_layers=num_transformer_layers,
        )
        self.pose_to_rgb = nn.Linear(pose_d_model, rgb_d_model)    # 256 → 512

        # ── Adaptive Fusion + Classifier ───────────────────────────────────────
        self.adaptive_fusion = AdaptiveFusion(emb_dim=rgb_d_model)
        self.classifier      = nn.Linear(rgb_d_model, num_classes)

    def encode_rgb(self, rgb_emb: torch.Tensor) -> torch.Tensor:
        """rgb_emb (B,T,2048) → (B,512)"""
        x = self.rgb_proj(rgb_emb)                    # (B, T, 512)
        x = x + self.rgb_pos_emb[:, :x.size(1), :]
        x = self.rgb_transformer(x)                   # (B, T, 512)
        return x.mean(dim=1)                          # (B, 512)

    def encode_pose(self, pose: torch.Tensor) -> torch.Tensor:
        """pose (B,T,258) → (B,512)"""
        x = self.pose_proj(pose)                      # (B, T, 256)
        x = x + self.pose_pos_emb[:, :x.size(1), :]
        x = self.pose_transformer(x)                  # (B, T, 256)
        x = x.mean(dim=1)                             # (B, 256)
        return self.pose_to_rgb(x)                    # (B, 512)

    def forward(
        self,
        rgb_emb: torch.Tensor,   # (B, T, 2048)
        pose: torch.Tensor,      # (B, T, 258)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits : (B, num_classes)
            alpha  : (B, 1)  — RGB weight
            beta   : (B, 1)  — Pose weight
        """
        rgb_feat  = self.encode_rgb(rgb_emb)
        pose_feat = self.encode_pose(pose)
        fused, alpha, beta = self.adaptive_fusion(rgb_feat, pose_feat)
        return self.classifier(fused), alpha, beta


# ── Sanity test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[TEST] Building AdaptiveMultiModalModel (embedding mode) ...")
    model = AdaptiveMultiModalModel(num_classes=8, num_frames=16)
    model.eval()

    dummy_rgb  = torch.randn(2, 16, 2048)
    dummy_pose = torch.randn(2, 16, 258)

    with torch.no_grad():
        logits, alpha, beta = model(dummy_rgb, dummy_pose)

    print(f"  logits       : {logits.shape}")
    print(f"  alpha        : {alpha[0].item():.4f}  beta : {beta[0].item():.4f}")
    print(f"  alpha + beta : {(alpha[0] + beta[0]).item():.4f}  (must be 1.0000)")
    assert logits.shape == torch.Size([2, 8])
    assert abs((alpha[0] + beta[0]).item() - 1.0) < 1e-5
    print("[TEST] PASSED")
