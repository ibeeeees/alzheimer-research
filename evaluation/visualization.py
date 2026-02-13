"""Visualization utilities for experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str | Path,
    title: str = "Confusion Matrix",
):
    """Plot and save a confusion matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_learning_curves(
    history: List[Dict[str, float]],
    save_path: str | Path,
    title: str = "Learning Curves",
):
    """Plot training loss and validation QWK over epochs."""
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", h.get("total", 0)) for h in history]
    val_qwk = [h.get("val_qwk", 0) for h in history]
    val_mae = [h.get("val_mae", 0) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_loss, "b-")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_qwk, "g-")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("QWK")
    axes[1].set_title("Validation QWK")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, val_mae, "r-")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MAE")
    axes[2].set_title("Validation MAE")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_survival_curves(
    predicted_survival: np.ndarray,
    event_indicators: np.ndarray,
    interval_months: int = 6,
    save_path: Optional[str | Path] = None,
    title: str = "Predicted Survival by Risk Group",
):
    """Plot Kaplan-Meier-style curves stratified by predicted risk quartile."""
    import matplotlib.pyplot as plt

    n_intervals = predicted_survival.shape[1]
    time_axis = np.arange(1, n_intervals + 1) * interval_months

    # Risk = 1 - final survival
    risk = 1.0 - predicted_survival[:, -1]
    quartiles = np.quantile(risk, [0.25, 0.5, 0.75])
    groups = np.digitize(risk, quartiles)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Low risk", "Medium-low", "Medium-high", "High risk"]
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    for g in range(4):
        mask = groups == g
        if mask.sum() == 0:
            continue
        mean_surv = predicted_survival[mask].mean(axis=0)
        ax.plot(time_axis, mean_surv, color=colors[g], label=labels[g], linewidth=2)
        # Shaded region for ±1 SD
        std_surv = predicted_survival[mask].std(axis=0)
        ax.fill_between(
            time_axis, mean_surv - std_surv, mean_surv + std_surv,
            alpha=0.15, color=colors[g],
        )

    ax.set_xlabel("Months from baseline")
    ax.set_ylabel("Predicted survival probability")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_calibration(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str | Path] = None,
):
    """Plot reliability diagram for ordinal class probabilities."""
    import matplotlib.pyplot as plt

    pred_labels = y_probs.argmax(axis=1)
    confidences = y_probs.max(axis=1)
    accuracies = (pred_labels == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_confs = []

    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_accs.append(accuracies[mask].mean())
        bin_confs.append(confidences[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(bin_centers, bin_accs, width=1 / n_bins * 0.8, alpha=0.5, label="Model")
    ax.scatter(bin_confs, bin_accs, color="red", zorder=3, s=30)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_embedding_tsne(
    mri_embeddings: np.ndarray,
    mri_labels: np.ndarray,
    speech_embeddings: Optional[np.ndarray] = None,
    speech_labels: Optional[np.ndarray] = None,
    save_path: Optional[str | Path] = None,
    title: str = "Latent Space (t-SNE)",
):
    """t-SNE visualization of aligned MRI and speech embeddings."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    all_embeddings = [mri_embeddings]
    all_labels = [mri_labels]
    all_modalities = [np.zeros(len(mri_labels))]  # 0 = MRI

    if speech_embeddings is not None:
        all_embeddings.append(speech_embeddings)
        all_labels.append(speech_labels)
        all_modalities.append(np.ones(len(speech_labels)))  # 1 = Speech

    combined = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    modalities = np.concatenate(all_modalities)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = tsne.fit_transform(combined)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color by severity
    scatter = axes[0].scatter(
        coords[:, 0], coords[:, 1], c=labels, cmap="RdYlGn_r",
        alpha=0.6, s=15, edgecolors="none",
    )
    fig.colorbar(scatter, ax=axes[0], label="CDR class")
    axes[0].set_title("Colored by severity")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Color by modality
    colors = ["#3498db" if m == 0 else "#e74c3c" for m in modalities]
    axes[1].scatter(
        coords[:, 0], coords[:, 1], c=colors, alpha=0.6, s=15, edgecolors="none",
    )
    from matplotlib.patches import Patch
    axes[1].legend(
        handles=[Patch(color="#3498db", label="MRI"), Patch(color="#e74c3c", label="Speech")],
    )
    axes[1].set_title("Colored by modality")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
