"""
model_comparison.py — Compare all four models: CNN, RNN, SVM, Logistic Regression.

Trains each model on the same data split and compares performance.
This is the main experiment for the project.

Run:
    python3 experiments/model_comparison.py
    python3 experiments/model_comparison.py --synthetic
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd

import config
from data.dataset import load_dataset, create_data_loaders
from models.cnn import build_model as build_cnn
from models.rnn import build_rnn
from models.sklearn_models import (
    build_svm,
    build_logreg,
    train_sklearn_model,
    evaluate_sklearn_model,
)
from training.trainer import Trainer
from evaluation.metrics import evaluate_model, print_classification_report

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns


def plot_all_visualizations(all_results, all_histories, results_dir):
    """Generate comprehensive comparison visualizations for all models."""
    os.makedirs(results_dir, exist_ok=True)
    model_names = list(all_results.keys())

    # ── 1. Bar chart: Accuracy / Precision / Recall / F1 ──
    metrics = ["accuracy", "precision", "recall", "f1"]
    data = {m: [all_results[name][m] for name in model_names] for m in metrics}
    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, data[metric], width, label=metric.capitalize())
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "model_comparison_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # ── 2. Confusion matrices side by side ──
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        cm = all_results[name]["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Healthy", "Alzheimer's"],
                    yticklabels=["Healthy", "Alzheimer's"], ax=ax)
        ax.set_title(f"{name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices — All Models", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(results_dir, "model_comparison_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── 3. Training curves for neural models (CNN, RNN) ──
    neural_models = [n for n in model_names if n in all_histories and all_histories[n]]
    if neural_models:
        fig, axes = plt.subplots(2, len(neural_models), figsize=(7 * len(neural_models), 8))
        if len(neural_models) == 1:
            axes = axes.reshape(-1, 1)
        for col, name in enumerate(neural_models):
            h = all_histories[name]
            epochs = range(1, len(h["train_loss"]) + 1)
            # Loss
            axes[0, col].plot(epochs, h["train_loss"], "b-", label="Train")
            axes[0, col].plot(epochs, h["val_loss"], "r-", label="Val")
            axes[0, col].set_title(f"{name} — Loss")
            axes[0, col].set_xlabel("Epoch")
            axes[0, col].set_ylabel("Loss")
            axes[0, col].legend()
            axes[0, col].grid(alpha=0.3)
            # Accuracy
            axes[1, col].plot(epochs, h["train_acc"], "b-", label="Train")
            axes[1, col].plot(epochs, h["val_acc"], "r-", label="Val")
            axes[1, col].set_title(f"{name} — Accuracy")
            axes[1, col].set_xlabel("Epoch")
            axes[1, col].set_ylabel("Accuracy")
            axes[1, col].legend()
            axes[1, col].grid(alpha=0.3)
        plt.suptitle("Training History — Neural Models", fontsize=14)
        plt.tight_layout()
        path = os.path.join(results_dir, "model_comparison_training.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

    # ── 4. F1-score comparison (horizontal bar chart) ──
    f1_scores = [all_results[n]["f1"] for n in model_names]
    colors = sns.color_palette("viridis", len(model_names))
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(model_names, f1_scores, color=colors)
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=11)
    ax.set_xlabel("F1-Score")
    ax.set_title("F1-Score Comparison")
    ax.set_xlim(0, max(f1_scores) * 1.2 if max(f1_scores) > 0 else 1.0)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "model_comparison_f1.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def run_model_comparison(synthetic=False):
    """Train and evaluate all four model types on the same data."""

    # Set seed
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Load data once
    features, labels = load_dataset(synthetic=synthetic)
    train_loader, val_loader, test_loader = create_data_loaders(features, labels)

    sample = features[0]
    input_height, input_width = sample.shape

    results_summary = []
    all_results = {}
    all_histories = {}

    # ---- CNN ----
    print(f"\n{'='*70}")
    print("MODEL: CNN")
    print(f"{'='*70}")

    cnn = build_cnn(input_height, input_width)
    trainer = Trainer(cnn, train_loader, val_loader, checkpoint_path="best_model_cnn.pt")
    cnn_history = trainer.train()

    cnn_ckpt = os.path.join(config.CHECKPOINT_DIR, "best_model_cnn.pt")
    if os.path.exists(cnn_ckpt):
        cnn.load_state_dict(torch.load(cnn_ckpt, weights_only=True))

    cnn_results = evaluate_model(cnn, test_loader)
    print_classification_report(cnn_results)
    all_results["CNN"] = cnn_results
    all_histories["CNN"] = cnn_history
    results_summary.append({
        "model": "CNN",
        "accuracy": cnn_results["accuracy"],
        "precision": cnn_results["precision"],
        "recall": cnn_results["recall"],
        "f1": cnn_results["f1"],
    })

    # ---- RNN ----
    print(f"\n{'='*70}")
    print("MODEL: RNN (Bidirectional GRU)")
    print(f"{'='*70}")

    rnn = build_rnn(input_height)
    trainer = Trainer(rnn, train_loader, val_loader, checkpoint_path="best_model_rnn.pt")
    rnn_history = trainer.train()

    rnn_ckpt = os.path.join(config.CHECKPOINT_DIR, "best_model_rnn.pt")
    if os.path.exists(rnn_ckpt):
        rnn.load_state_dict(torch.load(rnn_ckpt, weights_only=True))

    rnn_results = evaluate_model(rnn, test_loader)
    print_classification_report(rnn_results)
    all_results["RNN"] = rnn_results
    all_histories["RNN"] = rnn_history
    results_summary.append({
        "model": "RNN",
        "accuracy": rnn_results["accuracy"],
        "precision": rnn_results["precision"],
        "recall": rnn_results["recall"],
        "f1": rnn_results["f1"],
    })

    # ---- SVM ----
    print(f"\n{'='*70}")
    print("MODEL: SVM")
    print(f"{'='*70}")

    svm = build_svm()
    train_sklearn_model(svm, train_loader, val_loader)
    svm_results = evaluate_sklearn_model(svm, test_loader)
    print_classification_report(svm_results)
    all_results["SVM"] = svm_results
    all_histories["SVM"] = None
    results_summary.append({
        "model": "SVM",
        "accuracy": svm_results["accuracy"],
        "precision": svm_results["precision"],
        "recall": svm_results["recall"],
        "f1": svm_results["f1"],
    })

    # ---- Logistic Regression ----
    print(f"\n{'='*70}")
    print("MODEL: Logistic Regression")
    print(f"{'='*70}")

    logreg = build_logreg()
    train_sklearn_model(logreg, train_loader, val_loader)
    logreg_results = evaluate_sklearn_model(logreg, test_loader)
    print_classification_report(logreg_results)
    all_results["Logistic Regression"] = logreg_results
    all_histories["Logistic Regression"] = None
    results_summary.append({
        "model": "Logistic Regression",
        "accuracy": logreg_results["accuracy"],
        "precision": logreg_results["precision"],
        "recall": logreg_results["recall"],
        "f1": logreg_results["f1"],
    })

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    df = pd.DataFrame(results_summary)
    print(df.to_string(index=False, float_format="%.4f"))

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Generate all visualizations
    print(f"\nGenerating visualizations...")
    plot_all_visualizations(all_results, all_histories, config.RESULTS_DIR)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    run_model_comparison(synthetic=args.synthetic)
