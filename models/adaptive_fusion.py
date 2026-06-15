"""
adaptive_fusion.py
Learnable gate that dynamically weights RGB and Pose embeddings per sample.

alpha + beta = 1  (guaranteed by Softmax)
fused = alpha * rgb_emb + beta * pose_emb
"""

import torch
import torch.nn as nn


class AdaptiveFusion(nn.Module):
    """
    Learns per-sample importance weights for RGB and Pose embeddings.

    Args:
        emb_dim : size of both input embeddings (must be equal)
    """

    def __init__(self, emb_dim: int = 512):
        super().__init__()
        # Small gate network: sees both embeddings, outputs 2 weights
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),  # 1024 → 128
            nn.ReLU(),
            nn.Linear(128, 2),            # 128 → 2
            nn.Softmax(dim=1),            # alpha + beta = 1
        )

    def forward(
        self,
        rgb_emb: torch.Tensor,   # (B, emb_dim)
        pose_emb: torch.Tensor,  # (B, emb_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused : (B, emb_dim)  weighted combination
            alpha : (B, 1)        RGB weight
            beta  : (B, 1)        Pose weight
        """
        weights = self.gate(torch.cat([rgb_emb, pose_emb], dim=1))  # (B, 2)
        alpha   = weights[:, 0:1]   # (B, 1)
        beta    = weights[:, 1:2]   # (B, 1)
        fused   = alpha * rgb_emb + beta * pose_emb
        return fused, alpha, beta
