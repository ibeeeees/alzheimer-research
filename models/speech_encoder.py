"""Speech encoder: MLP projection of pre-extracted feature vectors.

Expects concatenated features from:
  - Handcrafted acoustic (eGeMAPS summary stats): 216-D
  - wav2vec 2.0 embedding:                        768-D
  - Handcrafted linguistic features:                14-D
  - Sentence-BERT embedding:                       384-D
  Total:                                          1382-D  →  256-D output

The heavy-duty feature extraction (wav2vec2, ASR, Sentence-BERT) happens
in the preprocessing / notebook stage, not here.
"""

import torch
import torch.nn as nn


class SpeechEncoder(nn.Module):
    """Two-layer MLP that projects concatenated speech features into the
    shared latent space.

    Input:  (B, input_dim)  — default 1382
    Output: (B, embed_dim)  — default 256
    """

    def __init__(
        self,
        input_dim: int = 1382,
        hidden_dim: int = 512,
        embed_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.67),  # lighter dropout on last layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
