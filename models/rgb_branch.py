"""
rgb_branch.py
RGB Spatial-Temporal Baseline Model.

Architecture:
  ResNet50 (per-frame CNN encoder)
  → Linear projection 2048 → d_model
  → Learnable positional embedding
  → Transformer Encoder (captures temporal relationships)
  → Mean pool over time
  → Linear classifier
"""

import torch
import torch.nn as nn
import torchvision.models as models


class RGBBaselineModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        num_frames: int = 16,
        d_model: int = 512,
        nhead: int = 8,
        num_transformer_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames

        # ── CNN backbone (ResNet50 without final FC) ───────────────────────────
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove avgpool and fc — keep only feature extractor layers
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        # ResNet50 outputs (batch, 2048, 1, 1) after avgpool

        # ── Project 2048 → d_model ─────────────────────────────────────────────
        self.input_proj = nn.Linear(2048, d_model)

        # ── Learnable positional embedding (1, T, d_model) ────────────────────
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, d_model))

        # ── Transformer Encoder ────────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,   # expects (batch, seq, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # ── Final classifier ───────────────────────────────────────────────────
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, C, H, W)  — T frames per video
        Returns:
            logits: (batch, num_classes)
        """
        B, T, C, H, W = x.shape

        # Merge batch and time so CNN processes each frame independently
        x = x.view(B * T, C, H, W)          # (B*T, C, H, W)
        x = self.cnn(x)                      # (B*T, 2048, 1, 1)
        x = x.view(B * T, -1)               # (B*T, 2048)

        # Project to transformer dimension
        x = self.input_proj(x)              # (B*T, d_model)

        # Restore temporal dimension
        x = x.view(B, T, -1)               # (B, T, d_model)

        # Add positional embedding
        x = x + self.pos_embedding[:, :T, :]

        # Transformer: model temporal relationships between frames
        x = self.transformer(x)             # (B, T, d_model)

        # Average pool across time dimension
        x = x.mean(dim=1)                   # (B, d_model)

        # Classify
        logits = self.classifier(x)         # (B, num_classes)
        return logits


# ── Quick sanity test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[TEST] Building RGBBaselineModel …")
    model = RGBBaselineModel(num_classes=8, num_frames=16)
    model.eval()

    dummy = torch.randn(2, 16, 3, 224, 224)   # batch=2, 16 frames, RGB 224×224
    with torch.no_grad():
        out = model(dummy)

    print(f"[TEST] Input  shape : {dummy.shape}")
    print(f"[TEST] Output shape : {out.shape}")
    assert out.shape == torch.Size([2, 8]), "Unexpected output shape!"
    print("[TEST] PASSED — expected torch.Size([2, 8])")
