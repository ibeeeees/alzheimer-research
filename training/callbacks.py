"""Training callbacks: early stopping and checkpointing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


class EarlyStopping:
    """Early stopping based on a monitored metric.

    Tracks the best value of a metric and stops training if no
    improvement is seen for `patience` epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 1e-4,
    ):
        """
        Args:
            patience: Number of epochs without improvement before stopping.
            mode: 'max' (e.g. QWK, accuracy) or 'min' (e.g. loss, MAE).
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Update with a new metric value.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        improved = (
            value > self.best_value + self.min_delta
            if self.mode == "max"
            else value < self.best_value - self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        self.best_value = None
        self.counter = 0
        self.should_stop = False


class CheckpointManager:
    """Save and load training checkpoints.

    Saves the best model (by validation metric) and the latest model
    for resume capability.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        prefix: str = "alzheimer",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.best_metric: Optional[float] = None

    def save(
        self,
        state: Dict[str, Any],
        metric: float,
        epoch: int,
        is_best: bool = False,
    ):
        """Save checkpoint.

        Args:
            state: Dict with model/optimizer/scaler state_dicts.
            metric: Current validation metric value.
            epoch: Current epoch number.
            is_best: If True, also save as 'best' checkpoint.
        """
        state["epoch"] = epoch
        state["metric"] = metric

        latest_path = self.checkpoint_dir / f"{self.prefix}_latest.pt"
        torch.save(state, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / f"{self.prefix}_best.pt"
            torch.save(state, best_path)
            self.best_metric = metric

    def load(
        self,
        path: Optional[str | Path] = None,
        which: str = "best",
    ) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            path: Explicit path, or None to use `which`.
            which: 'best' or 'latest'.

        Returns:
            Checkpoint state dict.
        """
        if path is None:
            path = self.checkpoint_dir / f"{self.prefix}_{which}.pt"
        return torch.load(path, map_location="cpu", weights_only=False)

    def exists(self, which: str = "latest") -> bool:
        path = self.checkpoint_dir / f"{self.prefix}_{which}.pt"
        return path.exists()
