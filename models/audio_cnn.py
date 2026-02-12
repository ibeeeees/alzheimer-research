"""Audio feature extractor: log-mel spectrogram + 2D ResNet backbone."""

import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchvision.models as models


class AudioCNN(nn.Module):
    """Audio feature extractor using log-mel spectrograms and a 2D CNN.

    Converts raw waveforms to log-mel spectrograms, then extracts
    embeddings using a pretrained ResNet backbone.

    Input:  (B, 1, T) raw waveform at sample_rate Hz
            OR (B, 1, n_mels, time_frames) pre-computed spectrogram
    Output: (B, embed_dim)
    """

    def __init__(
        self,
        embed_dim=256,
        pretrained=True,
        sample_rate=16000,
        n_mels=128,
        n_fft=1024,
        hop_length=512,
        depth="resnet18",
        from_spectrogram=False,
    ):
        super().__init__()
        self.from_spectrogram = from_spectrogram

        # Mel spectrogram transform (used when input is raw waveform)
        if not from_spectrogram:
            self.mel_spec = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            self.amplitude_to_db = T.AmplitudeToDB(stype="power")

        # 2D CNN backbone
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

        # Adapt first conv: 3 channels → 1 channel (single-channel spectrogram)
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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def _waveform_to_spectrogram(self, waveform):
        """Convert raw waveform to log-mel spectrogram.

        Args:
            waveform: (B, 1, T) raw audio

        Returns:
            spec: (B, 1, n_mels, time_frames)
        """
        # Remove channel dim for mel_spec, then add back
        x = waveform.squeeze(1)  # (B, T)
        x = self.mel_spec(x)     # (B, n_mels, time_frames)
        x = self.amplitude_to_db(x)  # Convert to dB scale
        x = x.unsqueeze(1)       # (B, 1, n_mels, time_frames)
        return x

    def forward(self, x):
        """Extract embedding from audio input.

        Args:
            x: (B, 1, T) raw waveform or (B, 1, n_mels, time) spectrogram

        Returns:
            embedding: (B, embed_dim)
        """
        if not self.from_spectrogram:
            x = self._waveform_to_spectrogram(x)

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
