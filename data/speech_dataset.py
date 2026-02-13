"""PyTorch dataset for pre-extracted DementiaBank speech features."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SpeechEmbeddingDataset(Dataset):
    """Dataset for pre-extracted speech feature vectors.

    Expects an .npz file with:
        features:  (N, D) float32 — concatenated speech features
        labels:    (N,)   int     — ordinal class labels {0, 1, 2, 3}

    Optionally also:
        subject_ids: (N,) — for subject-level splitting
    """

    def __init__(
        self,
        npz_path: str | Path,
        subject_ids: Optional[np.ndarray] = None,
    ):
        """
        Args:
            npz_path: Path to .npz file with 'features' and 'labels'.
            subject_ids: If provided, filter to these subjects only.
        """
        data = np.load(npz_path, allow_pickle=True)
        features = data["features"].astype(np.float32)
        labels = data["labels"].astype(np.int64)

        if "subject_ids" in data and subject_ids is not None:
            stored_ids = data["subject_ids"]
            mask = np.isin(stored_ids, subject_ids)
            features = features[mask]
            labels = labels[mask]

        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "label": self.labels[idx],
        }


class SyntheticSpeechDataset(Dataset):
    """Synthetic speech embeddings for testing the pipeline.

    Creates label-correlated random vectors.
    """

    def __init__(
        self,
        n_samples: int = 500,
        input_dim: int = 1382,
        num_classes: int = 4,
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)
        labels = rng.randint(0, num_classes, size=n_samples)
        # Label-correlated features
        features = rng.randn(n_samples, input_dim).astype(np.float32) * 0.3
        for k in range(num_classes):
            mask = labels == k
            features[mask, :64] += k * 0.5  # first 64 dims carry signal

        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "label": self.labels[idx],
        }
