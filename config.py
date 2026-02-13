"""Central configuration for the longitudinal multi-task Alzheimer's system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Config:
    """Full system configuration.

    Covers MRI encoder, speech encoder, temporal module, multi-task heads,
    cross-cohort alignment, and training hyperparameters.
    """

    # ── Paths ───────────────────────────────────────────────────────────
    project_root: Path = Path(__file__).parent
    data_dir: Path = Path(__file__).parent / "data_raw"
    embedding_dir: Path = Path(__file__).parent / "data_embeddings"
    checkpoint_dir: Path = Path(__file__).parent / "checkpoints"
    results_dir: Path = Path(__file__).parent / "experiment_results"

    # NACC-specific paths (set before use)
    nacc_mri_dir: Optional[Path] = None       # directory of .nii.gz volumes
    nacc_csv_path: Optional[Path] = None       # UDS longitudinal CSV
    dementiabank_dir: Optional[Path] = None    # Pitt Corpus root

    # ── CDR label mapping ───────────────────────────────────────────────
    class_names: Tuple[str, ...] = (
        "NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"
    )
    cdr_values: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    num_classes: int = 4
    num_thresholds: int = 3  # K - 1

    # ── MRI encoder ─────────────────────────────────────────────────────
    mri_volume_shape: Tuple[int, int, int] = (128, 128, 128)  # D, H, W
    mri_embed_dim: int = 256
    mri_backbone: str = "r3d_18"       # 3D ResNet-18
    mri_pretrained: bool = True
    mri_dropout: float = 0.1

    # ── Temporal module (longitudinal GRU) ──────────────────────────────
    time_encoding_dim: int = 64
    gru_hidden_dim: int = 256
    gru_num_layers: int = 1
    gru_dropout: float = 0.0   # only used when num_layers > 1

    # ── Speech encoder ──────────────────────────────────────────────────
    acoustic_handcrafted_dim: int = 216    # eGeMAPS summary stats
    wav2vec_embed_dim: int = 768           # wav2vec2-base output
    linguistic_handcrafted_dim: int = 14   # lexical/syntactic features
    sentbert_embed_dim: int = 384          # all-MiniLM-L6-v2
    speech_embed_dim: int = 256
    speech_hidden_dim: int = 512
    speech_dropout: float = 0.3

    # ── Task heads ──────────────────────────────────────────────────────
    # Ordinal CDR
    ordinal_hidden_dim: int = 128
    ordinal_dropout: float = 0.2

    # Survival (MCI→AD conversion)
    survival_hidden_dim: int = 128
    survival_dropout: float = 0.2
    survival_num_intervals: int = 6        # 6 × 6-month intervals = 36 months
    survival_interval_months: int = 6

    # Amyloid positivity (optional auxiliary)
    amyloid_hidden_dim: int = 64
    amyloid_dropout: float = 0.2

    # ── Cross-cohort alignment ──────────────────────────────────────────
    alignment_lambda: float = 0.1          # fixed weight for MMD loss
    mmd_kernel_bandwidth: str = "median"   # "median" heuristic or float
    alignment_warmup_epochs: int = 5       # linearly warm λ from 0

    # ── Multi-task loss weighting ───────────────────────────────────────
    # Homoscedastic uncertainty (Kendall et al. 2018)
    # s_t = log(σ_t²), initialized to 0 → σ²=1 → weight=0.5
    init_log_var_ord: float = 0.0
    init_log_var_surv: float = 0.0
    init_log_var_amy: float = 0.0

    # ── Training: Phase 1 (MRI pretrain, ordinal only) ──────────────────
    phase1_epochs: int = 30
    phase1_batch_size: int = 8
    phase1_grad_accum_steps: int = 2       # effective batch = 16
    phase1_lr: float = 3e-4
    phase1_min_lr: float = 1e-6
    phase1_weight_decay: float = 1e-4
    phase1_warmup_frac: float = 0.05       # 5% of steps
    phase1_patience: int = 7

    # ── Training: Phase 2 (multi-task + longitudinal + alignment) ───────
    phase2_epochs: int = 40
    phase2_mri_batch_size: int = 4         # sequences (variable-length)
    phase2_speech_batch_size: int = 16
    phase2_grad_accum_steps: int = 1
    phase2_lr_backbone: float = 1e-4
    phase2_lr_heads: float = 5e-4
    phase2_min_lr: float = 1e-6
    phase2_weight_decay: float = 1e-4
    phase2_warmup_frac: float = 0.05
    phase2_patience: int = 10

    # ── General training ────────────────────────────────────────────────
    seed: int = 42
    num_workers: int = 2
    pin_memory: bool = True
    use_amp: bool = True                   # mixed-precision training

    # ── Data splits ─────────────────────────────────────────────────────
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    n_folds_speech: int = 5                # k-fold for small DementiaBank

    # ── Data augmentation (MRI) ─────────────────────────────────────────
    aug_rotation_degrees: float = 10.0
    aug_scale_range: Tuple[float, float] = (0.95, 1.05)
    aug_translate_voxels: int = 5
    aug_intensity_shift: float = 0.1
    aug_intensity_scale: Tuple[float, float] = (0.9, 1.1)
    aug_noise_std: float = 0.02
    aug_flip_prob: float = 0.5

    # ── Experiment tracking ─────────────────────────────────────────────
    data_fractions: List[float] = field(
        default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 1.0]
    )
    embedding_dtype: str = "float16"

    # ── Conversion label ────────────────────────────────────────────────
    conversion_window_months: int = 36
    mci_cdr_value: float = 0.5
    ad_cdr_threshold: float = 1.0

    @property
    def speech_input_dim(self) -> int:
        """Total dimensionality of concatenated speech features."""
        return (
            self.acoustic_handcrafted_dim
            + self.wav2vec_embed_dim
            + self.linguistic_handcrafted_dim
            + self.sentbert_embed_dim
        )

    @property
    def embed_dim(self) -> int:
        """Unified embedding dim (must match across encoders)."""
        assert self.mri_embed_dim == self.speech_embed_dim
        return self.mri_embed_dim

    def ensure_dirs(self):
        """Create output directories."""
        for d in (
            self.data_dir,
            self.embedding_dir,
            self.checkpoint_dir,
            self.results_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
