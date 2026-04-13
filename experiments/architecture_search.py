"""
architecture_search.py — Compare different CNN architectures.

Tests three CNN sizes (small, medium, large) and varying dropout values.

Run:
    python3 experiments/architecture_search.py
    python3 experiments/architecture_search.py --synthetic
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd

import config
from data.dataset import load_dataset, create_data_loaders
from models.cnn import SpectrogramCNN, build_model
from training.trainer import Trainer
from evaluation.metrics import evaluate_model, print_classification_report


def run_architecture_search(synthetic=False):
    """Train and evaluate with different CNN architectures and dropout values."""

    features, labels = load_dataset(synthetic=synthetic)
    train_loader, val_loader, test_loader = create_data_loaders(features, labels)

    sample = features[0]
    input_height, input_width = sample.shape

    experiments = [
        ("small", None),
        ("medium", None),
        ("large", None),
        ("medium", 0.2),
        ("medium", 0.5),
    ]

    results_summary = []

    for arch_name, dropout_override in experiments:
        label = f"{arch_name}"
        if dropout_override is not None:
            label += f" (dropout={dropout_override})"

        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {label}")
        print(f"{'='*70}")

        cfg = config.CNN_CONFIGS[arch_name].copy()
        if dropout_override is not None:
            cfg["dropout"] = dropout_override

        model = SpectrogramCNN(
            input_height=input_height,
            input_width=input_width,
            conv_channels=cfg["conv_channels"],
            kernel_size=cfg["kernel_size"],
            pool_size=cfg["pool_size"],
            fc_size=cfg["fc_size"],
            dropout=cfg["dropout"],
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")

        trainer = Trainer(model, train_loader, val_loader)
        history = trainer.train()

        results = evaluate_model(model, test_loader)
        print_classification_report(results)

        results_summary.append({
            "architecture": label,
            "parameters": total_params,
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"],
        })

    print(f"\n{'='*70}")
    print("ARCHITECTURE COMPARISON SUMMARY")
    print(f"{'='*70}")
    df = pd.DataFrame(results_summary)
    print(df.to_string(index=False, float_format="%.4f"))

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR, "architecture_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    run_architecture_search(synthetic=args.synthetic)
