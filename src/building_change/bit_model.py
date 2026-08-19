"""Standalone BIT (Bitemporal Image Transformer) for change detection.

This implements the core BIT architecture from:
  Chen et al., "Remote Sensing Image Change Detection with Transformers" (2021)

The model uses a ResNet backbone to extract features from both dates,
tokenises the spatial features, refines them through a Transformer
encoder, and produces a binary change mask.

This implementation is self-contained — it depends only on PyTorch, timm,
and einops — avoiding the heavy mmcv/mmengine/mmsegmentation stack that
Open-CD requires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import einops
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tokeniser: Convert spatial feature maps to compact semantic tokens
# ---------------------------------------------------------------------------

class SpatialTokeniser(nn.Module):
    """Learn *token_count* compact semantic tokens from a spatial feature map."""

    def __init__(self, channels: int, token_count: int = 16, token_dim: int = 64):
        super().__init__()
        self.token_count = token_count
        self.token_dim = token_dim
        self.project = nn.Conv2d(channels, token_dim, kernel_size=1)
        self.attention = nn.Linear(token_dim, token_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → tokens: (B, token_count, token_dim)."""
        projected = self.project(x)                             # (B, D, H, W)
        flat = einops.rearrange(projected, "b d h w -> b (h w) d")  # (B, HW, D)
        weights = self.attention(flat)                           # (B, HW, K)
        weights = F.softmax(weights, dim=1)                     # spatial softmax
        tokens = torch.bmm(weights.transpose(1, 2), flat)      # (B, K, D)
        return tokens


# ---------------------------------------------------------------------------
# Transformer encoder for token refinement
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# BIT change detection model
# ---------------------------------------------------------------------------

class BITChangeDetector(nn.Module):
    """Bitemporal Image Transformer for binary change detection.

    Architecture:
        1. Shared ResNet-18 backbone → feature maps from both dates
        2. SpatialTokeniser → compact semantic tokens per date
        3. Concatenated tokens refined through Transformer encoder
        4. Token difference projected back to spatial domain
        5. Upsampled to full resolution as a binary change mask
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained_backbone: bool = True,
        token_count: int = 16,
        token_dim: int = 64,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
    ):
        super().__init__()
        # Shared backbone (feature extractor from both dates)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            features_only=True,
            out_indices=(2, 3),  # stage3 (stride 8) and stage4 (stride 16)
        )
        feature_info = self.backbone.feature_info.channels()
        backbone_channels = feature_info[-1]  # Use the deepest features

        # Reduce backbone channels to token dim
        self.reduce = nn.Conv2d(backbone_channels, token_dim, kernel_size=1)

        # Tokeniser and Transformer
        self.tokeniser = SpatialTokeniser(token_dim, token_count, token_dim)
        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(token_dim, transformer_heads) for _ in range(transformer_layers)]
        )

        # Token-to-spatial projection: project token differences back to feature space
        self.token_to_spatial = nn.Linear(token_dim, token_dim)

        # Decode change mask from token-attended feature differences
        self.decoder = nn.Sequential(
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(token_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),  # 2 classes: no-change, change
        )

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial features through the shared backbone."""
        features = self.backbone(x)
        return self.reduce(features[-1])  # Use deepest feature map

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Predict change logits from two co-registered image tensors.

        Args:
            x1: Before image, (B, 3, H, W), normalised to [0, 1].
            x2: After image, (B, 3, H, W), normalised to [0, 1].

        Returns:
            Logits (B, 2, H, W) — class 0 = no-change, class 1 = change.
        """
        input_size = x1.shape[2:]

        # Shared backbone
        f1 = self._extract(x1)  # (B, D, h, w)
        f2 = self._extract(x2)

        # Tokenise both dates
        t1 = self.tokeniser(f1)  # (B, K, D)
        t2 = self.tokeniser(f2)

        # Concatenate and refine through Transformer
        tokens = torch.cat([t1, t2], dim=1)     # (B, 2K, D)
        tokens = self.transformer(tokens)
        t1_refined = tokens[:, :t1.size(1), :]
        t2_refined = tokens[:, t1.size(1):, :]

        # Token difference
        token_diff = self.token_to_spatial(t2_refined - t1_refined)  # (B, K, D)

        # Attend token differences back to spatial feature map
        B, D, h, w = f2.shape
        spatial_flat = einops.rearrange(f2, "b d h w -> b (h w) d")
        # Attention: spatial queries, token keys/values
        attention_weights = torch.bmm(spatial_flat, token_diff.transpose(1, 2))  # (B, HW, K)
        attention_weights = F.softmax(attention_weights / math.sqrt(D), dim=2)
        change_features = torch.bmm(attention_weights, token_diff)               # (B, HW, D)
        change_features = einops.rearrange(change_features, "b (h w) d -> b d h w", h=h, w=w)

        # Decode and upsample
        logits = self.decoder(change_features)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
