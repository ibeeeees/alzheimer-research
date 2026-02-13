"""Temporal module for longitudinal visit sequences.

Uses sinusoidal time encoding (irregular intervals) + GRU aggregation.
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEncoding(nn.Module):
    """Continuous positional encoding for irregular time intervals.

    Maps a scalar Δt (months since baseline) to a d-dimensional vector
    using sinusoidal functions with log-spaced frequencies.
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        # Precompute frequency bands (not learnable)
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)  # (dim//2,)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dt: (B,) or (B, T) — time delta in months.
        Returns:
            (B, dim) or (B, T, dim) — sinusoidal encoding.
        """
        # Expand dims for broadcasting
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)        # (B, 1)
        elif dt.dim() == 2:
            pass                          # (B, T) → will broadcast
        else:
            raise ValueError(f"Expected 1D or 2D dt, got {dt.dim()}D")

        # dt: (..., 1) or (...,)
        args = dt.unsqueeze(-1) * self.freqs  # (..., dim//2)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (..., dim)


class TemporalGRU(nn.Module):
    """Time-aware GRU for longitudinal visit aggregation.

    For each subject, takes a variable-length sequence of per-visit
    embeddings with their time deltas and produces a single
    longitudinal representation.

    Single-visit subjects pass through a single GRU step (no special-casing).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        time_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_enc = SinusoidalTimeEncoding(time_dim)
        self.input_proj = nn.Linear(embed_dim + time_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        visit_embeddings: torch.Tensor,
        time_deltas: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visit_embeddings: (B, T_max, embed_dim) — padded visit embeddings.
            time_deltas:      (B, T_max) — months since baseline, 0-padded.
            lengths:          (B,) — actual number of visits per subject.
        Returns:
            (B, hidden_dim) — final hidden state per subject.
        """
        te = self.time_enc(time_deltas)                     # (B, T_max, time_dim)
        x = torch.cat([visit_embeddings, te], dim=-1)       # (B, T_max, embed+time)
        x = self.input_proj(x)                              # (B, T_max, hidden)

        # Pack to handle variable-length sequences efficiently
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, h_n = self.gru(packed)    # h_n: (num_layers, B, hidden)
        return h_n[-1]               # (B, hidden)  — last layer, final step
