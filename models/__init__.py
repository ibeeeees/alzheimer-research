"""Model components for the longitudinal multi-task Alzheimer's system."""

from .mri_encoder import MRIEncoder
from .temporal_module import SinusoidalTimeEncoding, TemporalGRU
from .speech_encoder import SpeechEncoder
from .task_heads import OrdinalHead, SurvivalHead, AmyloidHead
from .alignment import ClassConditionedMMD
from .losses import (
    coral_ordinal_loss,
    discrete_survival_loss,
    amyloid_bce_loss,
    MultiTaskLoss,
)
from .full_model import AlzheimerMultiTaskModel

__all__ = [
    "MRIEncoder",
    "SinusoidalTimeEncoding",
    "TemporalGRU",
    "SpeechEncoder",
    "OrdinalHead",
    "SurvivalHead",
    "AmyloidHead",
    "ClassConditionedMMD",
    "coral_ordinal_loss",
    "discrete_survival_loss",
    "amyloid_bce_loss",
    "MultiTaskLoss",
    "AlzheimerMultiTaskModel",
]
