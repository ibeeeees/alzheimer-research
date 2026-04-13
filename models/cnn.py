"""
cnn.py — Convolutional Neural Network for spectrogram classification.

The CNN treats spectrograms as grayscale images:
    Input shape: (batch, 1, frequency, time)
    Output: probability of Alzheimer's (0 to 1)

Architecture overview:
    [Conv2D → ReLU → MaxPool] × N  →  Flatten  →  Dense → Dropout → Output

Three sizes are available (configured in config.py):
    - "small":  2 conv layers, 64-unit FC  — fast, good for prototyping
    - "medium": 3 conv layers, 128-unit FC — balanced
    - "large":  4 conv layers, 256-unit FC — highest capacity
"""

import torch
import torch.nn as nn

import config


class SpectrogramCNN(nn.Module):
    """
    CNN that classifies spectrograms as healthy (0) or Alzheimer's (1).
    """

    def __init__(self, input_height, input_width, conv_channels, kernel_size=3,
                 pool_size=2, fc_size=128, dropout=0.4):
        """
        Args:
            input_height: number of frequency bins (e.g., 128 for Mel, 40 for MFCC)
            input_width: number of time steps in the spectrogram
            conv_channels: list of output channels for each conv layer
                           e.g., [32, 64, 128] means 3 conv layers
            kernel_size: size of the convolution kernel (default 3×3)
            pool_size: size of the max pooling kernel (default 2×2)
            fc_size: number of neurons in the fully connected layer
            dropout: dropout probability (0 = no dropout, 1 = drop everything)
        """
        super().__init__()

        # ---- Build the convolutional layers dynamically ----
        # We create a list of layers and then wrap them in nn.Sequential
        conv_layers = []
        in_channels = 1  # spectrograms have 1 channel (grayscale)

        for out_channels in conv_channels:
            conv_layers.extend([
                # Conv2D: slides a small kernel over the spectrogram to detect patterns
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),

                # BatchNorm: normalizes activations, helps training stability
                nn.BatchNorm2d(out_channels),

                # ReLU: activation function, replaces negative values with 0
                nn.ReLU(inplace=True),

                # MaxPool: shrinks the spatial dimensions by 2x,
                # keeping only the strongest activations
                nn.MaxPool2d(kernel_size=pool_size),
            ])
            in_channels = out_channels

        self.conv_block = nn.Sequential(*conv_layers)

        # ---- Calculate the size after all conv+pool layers ----
        # Each MaxPool2d(2) halves both height and width
        n_pools = len(conv_channels)
        final_height = input_height // (pool_size ** n_pools)
        final_width = input_width // (pool_size ** n_pools)
        flat_size = conv_channels[-1] * max(1, final_height) * max(1, final_width)

        # ---- Fully connected classification head ----
        self.classifier = nn.Sequential(
            # Dropout: randomly zeroes some neurons during training
            # This prevents overfitting (model memorizing the training data)
            nn.Dropout(dropout),

            # Dense layer: learns high-level combinations of features
            nn.Linear(flat_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Output layer: single neuron for binary classification
            nn.Linear(fc_size, 1),
        )

    def forward(self, x):
        """
        Forward pass: spectrogram in → probability out.

        Args:
            x: tensor of shape (batch, 1, freq, time)

        Returns:
            tensor of shape (batch, 1) — raw logits (before sigmoid)
        """
        # Pass through conv layers
        x = self.conv_block(x)

        # Flatten: (batch, channels, height, width) → (batch, channels*height*width)
        x = x.view(x.size(0), -1)

        # Pass through classifier
        x = self.classifier(x)

        return x


def build_model(input_height, input_width, architecture=None):
    """
    Build a CNN using a named architecture from config.

    Args:
        input_height: spectrogram frequency dimension
        input_width: spectrogram time dimension
        architecture: "small", "medium", or "large" (default: from config)

    Returns:
        SpectrogramCNN model
    """
    if architecture is None:
        architecture = config.CNN_ARCHITECTURE

    if architecture not in config.CNN_CONFIGS:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from: {list(config.CNN_CONFIGS.keys())}"
        )

    cfg = config.CNN_CONFIGS[architecture]

    model = SpectrogramCNN(
        input_height=input_height,
        input_width=input_width,
        conv_channels=cfg["conv_channels"],
        kernel_size=cfg["kernel_size"],
        pool_size=cfg["pool_size"],
        fc_size=cfg["fc_size"],
        dropout=cfg["dropout"],
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {architecture} CNN")
    print(f"  Input: ({input_height}, {input_width})")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}")

    return model
