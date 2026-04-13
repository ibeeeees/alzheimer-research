"""
main.py — Entry point for the Alzheimer's Speech Classification pipeline.

Supports four models:
    - CNN:  Convolutional Neural Network on 2D spectrograms
    - RNN:  Bidirectional GRU on sequential spectrogram features
    - SVM:  Support Vector Machine on flattened features
    - LogReg: Logistic Regression baseline on flattened features

Usage:
    python3 main.py                    # Train with default model (CNN)
    python3 main.py --model cnn        # Train CNN
    python3 main.py --model rnn        # Train RNN
    python3 main.py --model svm        # Train SVM
    python3 main.py --model logreg     # Train Logistic Regression
    python3 main.py --synthetic        # Use synthetic data (no download needed)

All settings are controlled from config.py.
"""

import os
import argparse
import torch
import numpy as np

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
from evaluation.metrics import (
    evaluate_model,
    print_classification_report,
    plot_confusion_matrix,
    plot_training_history,
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Alzheimer's Speech Classification")
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["cnn", "rnn", "svm", "logreg"],
        help="Model type (default: from config.py)",
    )
    parser.add_argument(
        "--feature", type=str, default=None,
        choices=["mfcc", "mel", "logmel"],
        help="Feature type (default: from config.py)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data for testing the pipeline",
    )
    return parser.parse_args()


def train_neural_model(model_type, features, labels):
    """Train a CNN or RNN model."""
    train_loader, val_loader, test_loader = create_data_loaders(features, labels)

    sample = features[0]
    input_height, input_width = sample.shape
    print(f"Spectrogram shape: ({input_height}, {input_width})")

    # Build the model
    if model_type == "cnn":
        model = build_cnn(input_height, input_width)
    else:
        model = build_rnn(input_height)

    # Train
    trainer = Trainer(model, train_loader, val_loader)
    history = trainer.train()

    # Load best checkpoint
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        print("Loaded best model from checkpoint.")

    # Evaluate
    results = evaluate_model(model, test_loader)

    # Plot training curves
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plot_training_history(
        history,
        save_path=os.path.join(config.RESULTS_DIR, f"{model_type}_training_history.png"),
    )

    return results, test_loader


def train_sklearn(model_type, features, labels):
    """Train an SVM or Logistic Regression model."""
    train_loader, val_loader, test_loader = create_data_loaders(features, labels)

    if model_type == "svm":
        model = build_svm()
    else:
        model = build_logreg()

    train_sklearn_model(model, train_loader, val_loader)
    results = evaluate_sklearn_model(model, test_loader)

    return results, test_loader


def main():
    args = parse_args()

    model_type = args.model or config.MODEL_TYPE
    feature_type = args.feature or config.FEATURE_TYPE

    set_seed(config.SEED)
    print("Alzheimer's Speech Classification")
    print("=" * 50)
    print(f"Model: {model_type.upper()}")
    print(f"Features: {feature_type}")

    # Step 1: Load and process audio data
    print(f"\n[1/3] Loading and processing audio data...")
    features, labels = load_dataset(
        feature_type=feature_type,
        synthetic=args.synthetic,
    )

    # Step 2: Train the model
    print(f"\n[2/3] Training {model_type.upper()} model...")

    if model_type in ("cnn", "rnn"):
        results, test_loader = train_neural_model(model_type, features, labels)
    else:
        results, test_loader = train_sklearn(model_type, features, labels)

    # Step 3: Evaluate
    print(f"\n[3/3] Evaluation results...")
    print_classification_report(results)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plot_confusion_matrix(
        results,
        save_path=os.path.join(config.RESULTS_DIR, f"{model_type}_confusion_matrix.png"),
    )

    print(f"\nDone! Results saved to: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
