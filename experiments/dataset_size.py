"""
dataset_size.py — Compare model performance with different amounts of training data.

Trains the same model using 25%, 50%, 75%, and 100% of the data.

Run:
    python3 experiments/dataset_size.py
    python3 experiments/dataset_size.py --synthetic
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd

import config
from data.dataset import load_dataset, create_data_loaders
from models.cnn import build_model
from training.trainer import Trainer
from evaluation.metrics import evaluate_model, print_classification_report


def run_dataset_size_experiment(synthetic=False):
    """Train and evaluate with 25%, 50%, 75%, and 100% of the data."""

    features, labels = load_dataset(synthetic=synthetic)

    data_fractions = [0.25, 0.50, 0.75, 1.00]
    results_summary = []

    sample = features[0]
    input_height, input_width = sample.shape

    for fraction in data_fractions:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: Using {fraction*100:.0f}% of data")
        print(f"{'='*70}")

        train_loader, val_loader, test_loader = create_data_loaders(
            features, labels, data_fraction=fraction
        )

        model = build_model(input_height, input_width)

        trainer = Trainer(model, train_loader, val_loader)
        history = trainer.train()

        results = evaluate_model(model, test_loader)
        print_classification_report(results)

        n_samples = int(len(features) * fraction)
        results_summary.append({
            "data_fraction": f"{fraction*100:.0f}%",
            "n_samples": n_samples,
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"],
        })

    print(f"\n{'='*70}")
    print("DATASET SIZE COMPARISON SUMMARY")
    print(f"{'='*70}")
    df = pd.DataFrame(results_summary)
    print(df.to_string(index=False, float_format="%.4f"))

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR, "dataset_size_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    run_dataset_size_experiment(synthetic=args.synthetic)
