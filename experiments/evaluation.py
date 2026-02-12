"""Full evaluation suite: metrics, data fraction experiments, comparison tables."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config
from models.ordinal_utils import (
    OrdinalHead,
    compute_all_metrics,
    optimize_thresholds,
)
from models.fusion_model import FusionModel, GatedFusionModel, AttentionFusionModel
from experiments.train_unimodal import (
    EmbeddingDataset,
    SyntheticEmbeddingDataset,
    train_model as train_unimodal,
)
from experiments.train_multimodal import (
    SyntheticMultimodalDataset,
    train_model as train_multimodal,
    build_fusion_model,
)


# ---------- Plotting ----------


def plot_confusion_matrix(cm, class_names, save_path=None, title="Confusion Matrix"):
    """Plot confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")
    plt.close(fig)
    return fig


def plot_calibration(y_true, y_probs, n_bins=10, save_path=None):
    """Plot reliability diagram for calibration analysis."""
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    pred_classes = y_probs.argmax(axis=1)
    confidences = y_probs.max(axis=1)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)
        else:
            bin_accs.append(float((pred_classes[mask] == y_true[mask]).mean()))
            bin_confs.append(float(confidences[mask].mean()))
            bin_counts.append(int(mask.sum()))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.bar(bin_confs, bin_accs, width=1.0 / n_bins, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Mean Predicted Confidence")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Reliability Diagram")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.bar(bin_confs, bin_counts, width=1.0 / n_bins, alpha=0.7, edgecolor="black", color="orange")
    ax2.set_xlabel("Mean Predicted Confidence")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved calibration plot to {save_path}")
    plt.close(fig)
    return fig


def plot_learning_curves(history, save_path=None, title="Learning Curves"):
    """Plot training/validation loss and QWK over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_qwk"], label="Val QWK", color="green")
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="blue")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Metrics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved learning curves to {save_path}")
    plt.close(fig)
    return fig


def plot_data_fraction_results(results, save_path=None):
    """Plot performance vs training data fraction."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    fractions = sorted(results.keys())
    metrics_to_plot = ["accuracy", "qwk", "mae"]
    titles = ["Accuracy", "QWK", "MAE"]
    colors = ["blue", "green", "red"]

    for ax, metric, title, color in zip(axes, metrics_to_plot, titles, colors):
        means = [results[f][f"{metric}_mean"] for f in fractions]
        stds = [results[f][f"{metric}_std"] for f in fractions]

        ax.errorbar(
            [f * 100 for f in fractions], means, yerr=stds,
            marker="o", capsize=5, color=color, linewidth=2,
        )
        ax.set_xlabel("Training Data (%)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs Data Fraction")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved data fraction plot to {save_path}")
    plt.close(fig)
    return fig


# ---------- Experiments ----------


def run_data_fraction_experiment(config, dataset_class, dataset_kwargs,
                                 embed_dim=256, loss_type="coral",
                                 is_multimodal=False, fusion_type="concat"):
    """Train models at different data fractions and collect metrics."""
    results = {}

    for fraction in config.data_fractions:
        print(f"\n{'='*50}")
        print(f"Data Fraction: {fraction*100:.0f}%")
        print(f"{'='*50}")

        dataset = dataset_class(**dataset_kwargs)

        n_use = max(10, int(len(dataset) * fraction))
        indices = np.random.RandomState(config.seed).permutation(len(dataset))[:n_use]
        subset = Subset(dataset, indices)

        # Get labels for stratification
        if is_multimodal:
            all_labels = np.array([subset[i][2].item() for i in range(len(subset))])
        else:
            all_labels = np.array([subset[i][1].item() for i in range(len(subset))])

        skf = StratifiedKFold(n_splits=min(config.n_folds, n_use // config.num_classes),
                              shuffle=True, random_state=config.seed)

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(subset)), all_labels)):
            train_sub = Subset(subset, train_idx)
            val_sub = Subset(subset, val_idx)

            if is_multimodal:
                _, _, metrics, _ = train_multimodal(
                    train_sub, val_sub, config,
                    embed_dim=embed_dim, fusion_type=fusion_type,
                    loss_type=loss_type, fold=fold + 1, verbose=False,
                )
            else:
                _, _, metrics, _ = train_unimodal(
                    train_sub, val_sub, config,
                    embed_dim=embed_dim, loss_type=loss_type,
                    fold=fold + 1, verbose=False,
                )
            fold_metrics.append(metrics)

        scalar_keys = ["accuracy", "qwk", "mae", "off_by_1", "loss"]
        frac_results = {}
        for key in scalar_keys:
            vals = [m[key] for m in fold_metrics]
            frac_results[f"{key}_mean"] = float(np.mean(vals))
            frac_results[f"{key}_std"] = float(np.std(vals))

        results[fraction] = frac_results

        print(f"  Acc: {frac_results['accuracy_mean']:.4f} ± {frac_results['accuracy_std']:.4f}")
        print(f"  QWK: {frac_results['qwk_mean']:.4f} ± {frac_results['qwk_std']:.4f}")
        print(f"  MAE: {frac_results['mae_mean']:.4f} ± {frac_results['mae_std']:.4f}")

    return results


def run_model_comparison(config, embed_dim=256):
    """Compare unimodal MRI, unimodal Audio, and multimodal fusion on synthetic data."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON EXPERIMENT")
    print("=" * 60)

    comparison_results = {}

    # Unimodal MRI
    print("\n--- Unimodal MRI ---")
    mri_dataset = SyntheticEmbeddingDataset(
        n_samples=1000, embed_dim=embed_dim, seed=config.seed
    )
    all_labels = np.array([mri_dataset[i][1].item() for i in range(len(mri_dataset))])
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(mri_dataset)), all_labels)):
        _, _, metrics, _ = train_unimodal(
            Subset(mri_dataset, train_idx), Subset(mri_dataset, val_idx),
            config, embed_dim=embed_dim, fold=fold + 1, verbose=False,
        )
        fold_metrics.append(metrics)
    comparison_results["MRI Only"] = _aggregate_fold_metrics(fold_metrics)

    # Unimodal Audio
    print("\n--- Unimodal Audio ---")
    audio_dataset = SyntheticEmbeddingDataset(
        n_samples=1000, embed_dim=embed_dim, seed=config.seed + 1
    )
    all_labels = np.array([audio_dataset[i][1].item() for i in range(len(audio_dataset))])
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(audio_dataset)), all_labels)):
        _, _, metrics, _ = train_unimodal(
            Subset(audio_dataset, train_idx), Subset(audio_dataset, val_idx),
            config, embed_dim=embed_dim, fold=fold + 1, verbose=False,
        )
        fold_metrics.append(metrics)
    comparison_results["Audio Only"] = _aggregate_fold_metrics(fold_metrics)

    # Multimodal Fusion (all 3 fusion types)
    multi_dataset = SyntheticMultimodalDataset(
        n_samples=1000, embed_dim=embed_dim, seed=config.seed
    )
    all_labels = np.array([multi_dataset[i][2].item() for i in range(len(multi_dataset))])

    for fusion_type in ["concat", "gated", "attention"]:
        print(f"\n--- Multimodal ({fusion_type}) ---")
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(multi_dataset)), all_labels)):
            _, _, metrics, _ = train_multimodal(
                Subset(multi_dataset, train_idx), Subset(multi_dataset, val_idx),
                config, embed_dim=embed_dim, fusion_type=fusion_type,
                fold=fold + 1, verbose=False,
            )
            fold_metrics.append(metrics)
        comparison_results[f"Multimodal ({fusion_type})"] = _aggregate_fold_metrics(fold_metrics)

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Model':<25} {'Acc':>10} {'QWK':>10} {'MAE':>10} {'Off-by-1':>10}")
    print("-" * 80)
    for model_name, metrics in comparison_results.items():
        print(
            f"{model_name:<25} "
            f"{metrics['accuracy_mean']:.4f}±{metrics['accuracy_std']:.3f} "
            f"{metrics['qwk_mean']:.4f}±{metrics['qwk_std']:.3f} "
            f"{metrics['mae_mean']:.4f}±{metrics['mae_std']:.3f} "
            f"{metrics['off_by_1_mean']:.4f}±{metrics['off_by_1_std']:.3f}"
        )
    print("=" * 80)

    return comparison_results


def run_loss_comparison(config, embed_dim=256):
    """Compare CORAL ordinal loss vs standard cross-entropy."""
    print("\n" + "=" * 60)
    print("LOSS FUNCTION COMPARISON (CORAL vs CE)")
    print("=" * 60)

    dataset = SyntheticEmbeddingDataset(
        n_samples=1000, embed_dim=embed_dim, seed=config.seed
    )
    all_labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)

    results = {}
    for loss_type in ["coral", "ce"]:
        print(f"\n--- Loss: {loss_type.upper()} ---")
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), all_labels)):
            _, _, metrics, _ = train_unimodal(
                Subset(dataset, train_idx), Subset(dataset, val_idx),
                config, embed_dim=embed_dim, loss_type=loss_type,
                fold=fold + 1, verbose=False,
            )
            fold_metrics.append(metrics)
        results[loss_type] = _aggregate_fold_metrics(fold_metrics)

    print(f"\n{'Loss':<10} {'Acc':>10} {'QWK':>10} {'MAE':>10}")
    print("-" * 50)
    for loss_type, metrics in results.items():
        print(
            f"{loss_type.upper():<10} "
            f"{metrics['accuracy_mean']:.4f}±{metrics['accuracy_std']:.3f} "
            f"{metrics['qwk_mean']:.4f}±{metrics['qwk_std']:.3f} "
            f"{metrics['mae_mean']:.4f}±{metrics['mae_std']:.3f}"
        )

    return results


def _aggregate_fold_metrics(fold_metrics):
    """Helper to aggregate metrics across folds."""
    scalar_keys = ["accuracy", "qwk", "mae", "off_by_1", "off_by_2", "loss"]
    agg = {}
    for key in scalar_keys:
        vals = [m[key] for m in fold_metrics]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
    return agg


# ---------- Main ----------


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation suite for Alzheimer's staging")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "comparison", "data_fraction", "loss_comparison"])
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    config.num_epochs = args.epochs
    config.early_stopping_patience = args.patience
    config.n_folds = args.n_folds
    config.batch_size = args.batch_size
    config.seed = args.seed
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.ensure_dirs()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print(f"Device: {config.device}")

    all_results = {}

    if args.experiment in ("all", "comparison"):
        comp = run_model_comparison(config, embed_dim=args.embed_dim)
        all_results["model_comparison"] = comp

    if args.experiment in ("all", "loss_comparison"):
        loss_comp = run_loss_comparison(config, embed_dim=args.embed_dim)
        all_results["loss_comparison"] = loss_comp

    if args.experiment in ("all", "data_fraction"):
        print("\n--- Data Fraction Experiment (Unimodal) ---")
        frac_results = run_data_fraction_experiment(
            config,
            SyntheticEmbeddingDataset,
            {"n_samples": 1000, "embed_dim": args.embed_dim, "seed": config.seed},
            embed_dim=args.embed_dim,
        )
        all_results["data_fraction_unimodal"] = frac_results
        plot_data_fraction_results(
            frac_results,
            save_path=str(config.results_dir / "data_fraction_unimodal.png"),
        )

    # Save all results to CSV
    results_path = config.results_dir / "evaluation_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "model", "metric", "mean", "std"])
        for exp_name, exp_results in all_results.items():
            if isinstance(exp_results, dict):
                for model_name, metrics in exp_results.items():
                    if isinstance(metrics, dict):
                        for key, val in metrics.items():
                            if key.endswith("_mean"):
                                base = key.replace("_mean", "")
                                std_val = metrics.get(f"{base}_std", 0.0)
                                writer.writerow([exp_name, model_name, base, val, std_val])
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
