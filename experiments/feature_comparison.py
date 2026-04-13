"""
feature_comparison.py — Compare MFCC vs Mel vs log-Mel spectrogram features.

Trains the same CNN architecture three times, each with a different audio
feature representation. Shows which feature type gives the best performance.

Run:
    python3 experiments/feature_comparison.py
    python3 experiments/feature_comparison.py --synthetic   # test without data
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


def run_feature_comparison(synthetic=False):
    """Train and evaluate with MFCC, Mel, and log-Mel features."""

    feature_types = ["mfcc", "mel", "logmel"]
    results_summary = []

    for feat_type in feature_types:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: Feature type = {feat_type}")
        print(f"{'='*70}")

        features, labels = load_dataset(feature_type=feat_type, synthetic=synthetic)
        train_loader, val_loader, test_loader = create_data_loaders(features, labels)

        sample = features[0]
        input_height, input_width = sample.shape

        model = build_model(input_height, input_width)
        trainer = Trainer(model, train_loader, val_loader)
        history = trainer.train()

        results = evaluate_model(model, test_loader)
        print_classification_report(results)

        results_summary.append({
            "feature_type": feat_type,
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"],
        })

    print(f"\n{'='*70}")
    print("FEATURE COMPARISON SUMMARY")
    print(f"{'='*70}")
    df = pd.DataFrame(results_summary)
    print(df.to_string(index=False, float_format="%.4f"))

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR, "feature_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    run_feature_comparison(synthetic=args.synthetic)
