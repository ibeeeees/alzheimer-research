"""Unified multi-task model assembling all components.

Wraps MRI encoder, temporal GRU, speech encoder, and all task heads
into a single nn.Module for clean training loops.
"""

import torch
import torch.nn as nn

from .mri_encoder import MRIEncoder
from .temporal_module import TemporalGRU
from .speech_encoder import SpeechEncoder
from .task_heads import OrdinalHead, SurvivalHead, AmyloidHead


class AlzheimerMultiTaskModel(nn.Module):
    """Full cross-cohort multi-task model.

    Depending on the input, runs:
      - MRI branch   → 3D CNN → optional temporal GRU → shared embedding
      - Speech branch → MLP → shared embedding
    Then feeds the embedding to task-specific heads.
    """

    def __init__(
        self,
        # Encoder dims
        mri_embed_dim: int = 256,
        speech_input_dim: int = 1382,
        speech_hidden_dim: int = 512,
        speech_embed_dim: int = 256,
        speech_dropout: float = 0.3,
        # MRI encoder
        mri_pretrained: bool = True,
        mri_dropout: float = 0.1,
        # Temporal
        time_dim: int = 64,
        gru_hidden_dim: int = 256,
        gru_num_layers: int = 1,
        # Task heads
        num_classes: int = 4,
        ordinal_hidden: int = 128,
        ordinal_dropout: float = 0.2,
        survival_hidden: int = 128,
        survival_dropout: float = 0.2,
        survival_intervals: int = 6,
        amyloid_hidden: int = 64,
        amyloid_dropout: float = 0.2,
    ):
        super().__init__()

        # ── Encoders ──────────────────────────────────────────────
        self.mri_encoder = MRIEncoder(
            embed_dim=mri_embed_dim,
            pretrained=mri_pretrained,
            dropout=mri_dropout,
        )
        self.temporal_gru = TemporalGRU(
            embed_dim=mri_embed_dim,
            time_dim=time_dim,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
        )
        self.speech_encoder = SpeechEncoder(
            input_dim=speech_input_dim,
            hidden_dim=speech_hidden_dim,
            embed_dim=speech_embed_dim,
            dropout=speech_dropout,
        )

        # ── Task heads (shared across modalities) ─────────────────
        embed_dim = mri_embed_dim  # both encoders produce same dim
        self.ordinal_head = OrdinalHead(
            embed_dim=embed_dim,
            hidden_dim=ordinal_hidden,
            num_classes=num_classes,
            dropout=ordinal_dropout,
        )
        self.survival_head = SurvivalHead(
            embed_dim=embed_dim,
            hidden_dim=survival_hidden,
            num_intervals=survival_intervals,
            dropout=survival_dropout,
        )
        self.amyloid_head = AmyloidHead(
            embed_dim=embed_dim,
            hidden_dim=amyloid_hidden,
            dropout=amyloid_dropout,
        )

    # ── Forward methods for each branch ───────────────────────────

    def encode_mri_single(self, volume: torch.Tensor) -> torch.Tensor:
        """Encode a single-visit MRI batch.

        Args:
            volume: (B, 1, D, H, W)
        Returns:
            (B, embed_dim) MRI embedding.
        """
        return self.mri_encoder(volume)

    def encode_mri_longitudinal(
        self,
        volumes: torch.Tensor,
        time_deltas: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a longitudinal sequence of MRI volumes.

        Args:
            volumes:     (B, T_max, 1, D, H, W) — padded volume sequences.
            time_deltas: (B, T_max) — months since baseline.
            lengths:     (B,) — actual sequence lengths.
        Returns:
            (B, gru_hidden_dim) longitudinal representation.
        """
        B, T_max = volumes.shape[:2]
        vol_shape = volumes.shape[2:]  # (1, D, H, W)

        # Flatten batch and time → encode all volumes at once
        flat = volumes.reshape(B * T_max, *vol_shape)
        embeddings = self.mri_encoder(flat)  # (B*T_max, embed_dim)
        embeddings = embeddings.reshape(B, T_max, -1)  # (B, T_max, embed_dim)

        return self.temporal_gru(embeddings, time_deltas, lengths)

    def encode_speech(self, features: torch.Tensor) -> torch.Tensor:
        """Encode pre-extracted speech features.

        Args:
            features: (B, speech_input_dim)
        Returns:
            (B, embed_dim)
        """
        return self.speech_encoder(features)

    # ── Task head forward methods ─────────────────────────────────

    def predict_ordinal(self, h: torch.Tensor) -> dict:
        """Ordinal CDR prediction from any encoder's embedding."""
        return self.ordinal_head(h)

    def predict_survival(self, h: torch.Tensor) -> dict:
        """Conversion hazard/survival prediction (MRI only)."""
        return self.survival_head(h)

    def predict_amyloid(self, h: torch.Tensor) -> dict:
        """Amyloid positivity prediction (MRI only)."""
        return self.amyloid_head(h)

    # ── Convenience: full forward for Phase 1 (single-visit MRI) ──

    def forward_phase1(self, volume: torch.Tensor) -> dict:
        """Phase 1: single-visit MRI → ordinal CDR only.

        Args:
            volume: (B, 1, D, H, W)
        Returns:
            dict with ordinal head outputs.
        """
        h = self.encode_mri_single(volume)
        return {"embedding": h, **self.predict_ordinal(h)}

    # ── Convenience: full forward for Phase 2 ─────────────────────

    def forward_mri_multitask(
        self,
        volumes: torch.Tensor,
        time_deltas: torch.Tensor,
        lengths: torch.Tensor,
        run_survival: bool = False,
        run_amyloid: bool = False,
    ) -> dict:
        """Phase 2: longitudinal MRI → all applicable task heads.

        Args:
            volumes:     (B, T_max, 1, D, H, W)
            time_deltas: (B, T_max)
            lengths:     (B,)
            run_survival: whether to compute survival predictions
            run_amyloid:  whether to compute amyloid predictions
        Returns:
            dict of outputs from each active head, plus the embedding.
        """
        h = self.encode_mri_longitudinal(volumes, time_deltas, lengths)
        result = {"embedding": h}

        # Ordinal (always)
        ord_out = self.predict_ordinal(h)
        result.update({f"ord_{k}": v for k, v in ord_out.items()})

        if run_survival:
            surv_out = self.predict_survival(h)
            result.update({f"surv_{k}": v for k, v in surv_out.items()})

        if run_amyloid:
            amy_out = self.predict_amyloid(h)
            result.update({f"amy_{k}": v for k, v in amy_out.items()})

        return result

    def forward_speech(self, features: torch.Tensor) -> dict:
        """Speech branch → ordinal CDR only."""
        h = self.encode_speech(features)
        ord_out = self.predict_ordinal(h)
        return {"embedding": h, **ord_out}

    @classmethod
    def from_config(cls, cfg) -> "AlzheimerMultiTaskModel":
        """Build model from a Config dataclass."""
        return cls(
            mri_embed_dim=cfg.mri_embed_dim,
            speech_input_dim=cfg.speech_input_dim,
            speech_hidden_dim=cfg.speech_hidden_dim,
            speech_embed_dim=cfg.speech_embed_dim,
            speech_dropout=cfg.speech_dropout,
            mri_pretrained=cfg.mri_pretrained,
            mri_dropout=cfg.mri_dropout,
            time_dim=cfg.time_encoding_dim,
            gru_hidden_dim=cfg.gru_hidden_dim,
            gru_num_layers=cfg.gru_num_layers,
            num_classes=cfg.num_classes,
            ordinal_hidden=cfg.ordinal_hidden_dim,
            ordinal_dropout=cfg.ordinal_dropout,
            survival_hidden=cfg.survival_hidden_dim,
            survival_dropout=cfg.survival_dropout,
            survival_intervals=cfg.survival_num_intervals,
            amyloid_hidden=cfg.amyloid_hidden_dim,
            amyloid_dropout=cfg.amyloid_dropout,
        )
