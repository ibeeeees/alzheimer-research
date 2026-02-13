"""Multi-task prediction heads.

1. OrdinalHead   — CORAL ordinal regression for CDR severity
2. SurvivalHead  — discrete-time hazard model for MCI→AD conversion
3. AmyloidHead   — binary classification for amyloid positivity
"""

import torch
import torch.nn as nn


class OrdinalHead(nn.Module):
    """CORAL ordinal regression head.

    Produces a scalar severity score via a small MLP, then compares
    against K-1 learnable thresholds to yield cumulative logits.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),            # scalar severity
        )
        # Learnable thresholds, initialised linearly spaced
        self.thresholds = nn.Parameter(
            torch.linspace(-1.0, 1.0, self.num_thresholds)
        )

    def forward(self, h: torch.Tensor) -> dict:
        """
        Args:
            h: (B, embed_dim)
        Returns:
            dict with keys:
              severity  : (B, 1) raw severity score
              cum_logits: (B, K-1) cumulative logits (for CORAL loss)
        """
        severity = self.mlp(h)                              # (B, 1)
        cum_logits = severity - self.thresholds.unsqueeze(0)  # (B, K-1)
        return {"severity": severity, "cum_logits": cum_logits}

    @torch.no_grad()
    def predict(self, h: torch.Tensor) -> torch.Tensor:
        """Return ordinal class predictions (B,)."""
        out = self.forward(h)
        return (out["cum_logits"] > 0).sum(dim=1)

    @torch.no_grad()
    def class_probabilities(self, h: torch.Tensor) -> torch.Tensor:
        """Return (B, K) class probability distribution."""
        out = self.forward(h)
        cum_probs = torch.sigmoid(out["cum_logits"])  # P(Y > k)
        probs = torch.zeros(
            h.size(0), self.num_classes, device=h.device, dtype=h.dtype
        )
        probs[:, 0] = 1.0 - cum_probs[:, 0]
        for k in range(1, self.num_thresholds):
            probs[:, k] = cum_probs[:, k - 1] - cum_probs[:, k]
        probs[:, -1] = cum_probs[:, -1]
        return probs.clamp(min=1e-7)


class SurvivalHead(nn.Module):
    """Discrete-time survival head for MCI→AD conversion.

    Predicts per-interval hazards h_j = P(convert in j | survived to j).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        num_intervals: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_intervals = num_intervals
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_intervals),  # one logit per interval
        )

    def forward(self, h: torch.Tensor) -> dict:
        """
        Args:
            h: (B, embed_dim)
        Returns:
            dict with keys:
              hazard_logits: (B, J) raw logits
              hazards:       (B, J) sigmoid hazards ∈ (0, 1)
              survival:      (B, J) cumulative survival S(t_j)
        """
        logits = self.mlp(h)                            # (B, J)
        hazards = torch.sigmoid(logits)                 # (B, J)
        # S(t_j) = Π_{l=1}^{j} (1 - h_l)
        survival = torch.cumprod(1.0 - hazards, dim=1)  # (B, J)
        return {
            "hazard_logits": logits,
            "hazards": hazards,
            "survival": survival,
        }

    @torch.no_grad()
    def predict_conversion_prob(
        self, h: torch.Tensor, at_interval: int = -1
    ) -> torch.Tensor:
        """P(convert by interval j | x).  Default: last interval (36 mo)."""
        out = self.forward(h)
        return 1.0 - out["survival"][:, at_interval]


class AmyloidHead(nn.Module):
    """Binary classification head for amyloid positivity."""

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> dict:
        """
        Returns:
            dict with keys:
              logit: (B, 1)
              prob:  (B, 1)
        """
        logit = self.mlp(h)
        return {"logit": logit, "prob": torch.sigmoid(logit)}
