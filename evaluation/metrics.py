"""
metrics.py — Evaluation metrics and visualization.

After training, use these functions to measure how well the model performs:
- Accuracy: what fraction of predictions are correct
- Precision: of all predicted Alzheimer's, how many actually are
- Recall: of all actual Alzheimer's, how many did we catch
- F1-score: harmonic mean of precision and recall
- Confusion matrix: 2×2 table showing correct/incorrect predictions
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import config


@torch.no_grad()
def get_predictions(model, data_loader, device=None):
    """
    Run the model on all data in the loader and collect predictions + true labels.

    Args:
        model: trained CNN
        data_loader: DataLoader to evaluate on (usually test set)
        device: device to run on

    Returns:
        all_preds: numpy array of predicted labels (0 or 1)
        all_labels: numpy array of true labels
        all_probs: numpy array of predicted probabilities
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for batch_features, batch_labels in data_loader:
        batch_features = batch_features.to(device)

        # Get model output (raw logits)
        outputs = model(batch_features)

        # Convert logits to probabilities using sigmoid
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        # Convert probabilities to binary predictions (threshold = 0.5)
        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(batch_labels.numpy().flatten().astype(int))

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def evaluate_model(model, data_loader, device=None):
    """
    Compute all evaluation metrics.

    Args:
        model: trained CNN
        data_loader: DataLoader (usually test set)
        device: device to run on

    Returns:
        dict with accuracy, precision, recall, f1, and confusion_matrix
    """
    preds, labels, probs = get_predictions(model, data_loader, device)

    results = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds),
        "predictions": preds,
        "labels": labels,
        "probabilities": probs,
    }

    return results


def print_classification_report(results):
    """
    Print a formatted summary of all metrics.

    Args:
        results: dict from evaluate_model()
    """
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy:  {results['accuracy']:.4f}  ({results['accuracy']*100:.1f}%)")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-score:  {results['f1']:.4f}")
    print()

    # Also print sklearn's detailed report
    print("Detailed report:")
    print(classification_report(
        results["labels"],
        results["predictions"],
        target_names=["Healthy", "Alzheimer's"],
    ))


def plot_confusion_matrix(results, save_path=None):
    """
    Display the confusion matrix as a heatmap.

    The confusion matrix shows:
                        Predicted
                    Healthy  Alzheimer's
    Actual Healthy  [ TN        FP     ]
           Alz     [ FN        TP     ]

    - TN (True Negative): correctly predicted healthy
    - TP (True Positive): correctly predicted Alzheimer's
    - FP (False Positive): healthy person misclassified as Alzheimer's
    - FN (False Negative): Alzheimer's person misclassified as healthy

    Args:
        results: dict from evaluate_model()
        save_path: if provided, save the plot to this file
    """
    cm = results["confusion_matrix"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,          # show numbers in each cell
        fmt="d",             # integer format
        cmap="Blues",         # color scheme
        xticklabels=["Healthy", "Alzheimer's"],
        yticklabels=["Healthy", "Alzheimer's"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.

    These plots help you diagnose:
    - Overfitting: training loss keeps going down but val loss goes up
    - Underfitting: both losses are high and not improving
    - Good fit: both losses decrease and stabilize

    Args:
        history: dict from trainer.train()
        save_path: if provided, save the plot to this file
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Training history saved to: {save_path}")

    plt.show()
