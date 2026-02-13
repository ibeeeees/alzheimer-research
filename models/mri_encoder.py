"""3D ResNet-18 encoder for structural MRI volumes."""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class MRIEncoder(nn.Module):
    """3D ResNet-18 adapted for single-channel brain MRI.

    Input : (B, 1, D, H, W) — e.g. (B, 1, 128, 128, 128)
    Output: (B, embed_dim)   — default 256-D embedding
    """

    def __init__(self, embed_dim: int = 256, pretrained: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        backbone = r3d_18(weights=weights)

        # Adapt first conv: 3-channel RGB → 1-channel grayscale
        old_conv = backbone.stem[0]
        new_conv = nn.Conv3d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.stem[0] = new_conv

        # Remove the original FC head
        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool3d(1)

        self.projection = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W) MRI volume, intensity-normalised.
        Returns:
            (B, embed_dim) embedding.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)  # (B, 512)
        return self.projection(x)
