"""
pose_branch.py
Pose-only baseline model for ISL recognition.

Input:  (batch, 16, 258)  — 16 frames, 258 landmark values per frame
Output: (batch, 8)        — class logits

Architecture:
  Linear projection 258 → 256
  Learnable positional embedding
  Transformer Encoder (4 layers, 4 heads)
  Mean pool over time
  Linear classifier → 8 classes
"""

import torch
import torch.nn as nn


class PoseBranchModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 258,
        num_classes: int = 8,
        num_frames: int = 16,
        d_model: int = 256,
        nhead: int = 4,
        num_transformer_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames

        # Project raw landmark vector to transformer dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, d_model))

        # Transformer encoder — models temporal relationships between frames
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, 258)
        Returns:
            logits: (batch, num_classes)
        """
        x = self.input_proj(x)               # (B, T, d_model)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)              # (B, T, d_model)
        x = x.mean(dim=1)                    # (B, d_model)
        return self.classifier(x)            # (B, num_classes)


# ── Sanity test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[TEST] Building PoseBranchModel ...")
    model = PoseBranchModel(input_dim=258, num_classes=8, num_frames=16)
    model.eval()

    dummy = torch.randn(2, 16, 258)   # batch=2, 16 frames, 258 landmarks
    with torch.no_grad():
        out = model(dummy)

    print(f"[TEST] Input  shape : {dummy.shape}")
    print(f"[TEST] Output shape : {out.shape}")
    assert out.shape == torch.Size([2, 8]), "Unexpected output shape!"
    print("[TEST] PASSED — expected torch.Size([2, 8])")
