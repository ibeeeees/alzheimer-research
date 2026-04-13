"""
rnn.py — Recurrent Neural Network for spectrogram classification.

The RNN treats the spectrogram as a sequence over time:
    Input shape: (batch, 1, frequency, time)
    → reshaped to (batch, time, frequency) for the GRU
    Output: probability of Alzheimer's (0 to 1)

Architecture:
    Spectrogram → Bidirectional GRU → Dense → Dropout → Output

Uses GRU (Gated Recurrent Unit), which captures temporal patterns
like pauses, rhythm changes, and prosody shifts in speech.
"""

import torch
import torch.nn as nn

import config


class SpectrogramRNN(nn.Module):
    """
    RNN that classifies spectrograms as healthy (0) or Alzheimer's (1).

    The spectrogram's time axis is treated as the sequence dimension,
    and each time step's frequency bins are the input features.
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.3, fc_size=64, bidirectional=True):
        """
        Args:
            input_size: number of frequency bins (features per time step)
            hidden_size: GRU hidden state size
            num_layers: number of stacked GRU layers
            dropout: dropout between GRU layers
            fc_size: fully connected layer size
            bidirectional: use bidirectional GRU
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        gru_output_size = hidden_size * self.num_directions

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_output_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_size, 1),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: tensor of shape (batch, 1, freq, time)

        Returns:
            tensor of shape (batch, 1) — raw logits
        """
        # Remove channel dim and transpose: (batch, 1, freq, time) -> (batch, time, freq)
        x = x.squeeze(1).permute(0, 2, 1)

        # GRU forward pass
        output, hidden = self.gru(x)

        # Use the last time step's output for classification
        # For bidirectional, concatenate the last forward and first backward outputs
        if self.bidirectional:
            # hidden shape: (num_layers * 2, batch, hidden_size)
            forward_last = hidden[-2]   # last forward layer
            backward_last = hidden[-1]  # last backward layer
            last_output = torch.cat([forward_last, backward_last], dim=1)
        else:
            last_output = hidden[-1]

        x = self.classifier(last_output)
        return x


def build_rnn(input_height):
    """
    Build an RNN using settings from config.

    Args:
        input_height: number of frequency bins (n_mels or n_mfcc)

    Returns:
        SpectrogramRNN model
    """
    model = SpectrogramRNN(
        input_size=input_height,
        hidden_size=config.RNN_HIDDEN_SIZE,
        num_layers=config.RNN_NUM_LAYERS,
        dropout=config.RNN_DROPOUT,
        fc_size=config.RNN_FC_SIZE,
        bidirectional=config.RNN_BIDIRECTIONAL,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Bidirectional GRU" if config.RNN_BIDIRECTIONAL else "Model: GRU")
    print(f"  Input features: {input_height}")
    print(f"  Hidden size: {config.RNN_HIDDEN_SIZE}")
    print(f"  Layers: {config.RNN_NUM_LAYERS}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}")

    return model
