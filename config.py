"""Central configuration for the Alzheimer's ordinal regression pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class Config:
    # --- Paths ---
    project_root: Path = Path(__file__).parent
    embedding_dir: Path = Path(__file__).parent / "data_embeddings"
    checkpoint_dir: Path = Path(__file__).parent / "checkpoints"
    results_dir: Path = Path(__file__).parent / "experiment_results"

    # --- CDR label mapping ---
    # 4 ordered classes: CDR 0, 0.5, 1, 2+
    class_names: Tuple[str, ...] = ("NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented")
    cdr_values: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    num_classes: int = 4
    num_thresholds: int = 3  # num_classes - 1

    # --- Model architecture ---
    embed_dim: int = 256
    mri_input_shape_3d: Tuple[int, ...] = (1, 64, 128, 128)  # (C, D, H, W)
    mri_input_shape_2d: Tuple[int, ...] = (1, 224, 224)       # (C, H, W)
    audio_sample_rate: int = 16000
    audio_n_mels: int = 128
    audio_n_fft: int = 1024
    audio_hop_length: int = 512

    # Fusion network
    fusion_hidden_dims: Tuple[int, ...] = (256, 128)
    fusion_dropout: float = 0.3

    # --- Training ---
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 10
    scheduler: str = "cosine"  # "cosine" or "step"

    # --- Experiments ---
    data_fractions: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    n_folds: int = 5
    seed: int = 42

    # --- Embedding storage ---
    embedding_dtype: str = "float16"

    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
