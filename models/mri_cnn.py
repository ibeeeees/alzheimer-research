"""MRI feature extractors: 3D ResNet-18 for volumes, 2D ResNet-18 for slices."""

import torch
import torch.nn as nn
import torchvision.models as models


class MRIResNet3D(nn.Module):
    """3D ResNet-18 for volumetric MRI data (e.g., NACC NIfTI volumes).

    Adapts the video ResNet r3d_18 architecture for single-channel
    grayscale MRI volumes.

    Input:  (B, 1, D, H, W) — e.g., (B, 1, 64, 128, 128)
    Output: (B, embed_dim)
    """

    def __init__(self, embed_dim=256, pretrained=True):
        super().__init__()
        # Use video ResNet-18 as 3D backbone
        backbone = models.video.r3d_18(
            weights=models.video.R3D_18_Weights.DEFAULT if pretrained else None
        )

        # Adapt first conv: 3 channels → 1 channel (grayscale MRI)
        old_conv = backbone.stem[0]
        self.stem_conv = nn.Conv3d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            # Average the pretrained RGB weights across channel dim
            with torch.no_grad():
                self.stem_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        # Replace stem conv and keep the rest of the stem
        stem_layers = list(backbone.stem.children())
        stem_layers[0] = self.stem_conv
        self.stem = nn.Sequential(*stem_layers)

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # Projection to embedding space
        backbone_out_dim = 512
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Extract embedding from 3D MRI volume.

        Args:
            x: (B, 1, D, H, W) grayscale MRI volume

        Returns:
            embedding: (B, embed_dim)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.projection(x)
        return x

    def extract_embedding(self, x):
        """Alias for forward — explicit embedding extraction interface."""
        return self.forward(x)


class MRIResNet2D(nn.Module):
    """2D ResNet-18 for single MRI slices (e.g., Kaggle/Mendeley JPG images).

    Uses ImageNet-pretrained ResNet-18, adapted for single-channel input.

    Input:  (B, 1, H, W) — e.g., (B, 1, 224, 224)
    Output: (B, embed_dim)
    """

    def __init__(self, embed_dim=256, pretrained=True, depth="resnet18"):
        super().__init__()
        # Select backbone depth
        if depth == "resnet18":
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            backbone_out_dim = 512
        elif depth == "resnet34":
            backbone = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT if pretrained else None
            )
            backbone_out_dim = 512
        else:
            raise ValueError(f"Unsupported depth: {depth}. Use 'resnet18' or 'resnet34'.")

        # Adapt first conv: 3 channels → 1 channel (grayscale)
        old_conv = backbone.conv1
        self.conv1 = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # Projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Extract embedding from 2D MRI slice.

        Args:
            x: (B, 1, H, W) grayscale MRI slice

        Returns:
            embedding: (B, embed_dim)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.projection(x)
        return x

    def extract_embedding(self, x):
        """Alias for forward — explicit embedding extraction interface."""
        return self.forward(x)
