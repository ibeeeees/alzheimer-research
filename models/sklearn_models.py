"""
sklearn_models.py — SVM and Logistic Regression classifiers.

These models operate on flattened spectrogram features (1D vectors)
rather than 2D images. They serve as baselines to compare against
the deep learning models (CNN, RNN).

SVM: Support Vector Machine with RBF kernel — finds a decision boundary
     in high-dimensional feature space.
Logistic Regression: Linear model that estimates class probabilities —
     establishes a performance floor.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import config


def _flatten_loader(data_loader):
    """
    Extract all features and labels from a DataLoader, flattening spectrograms.

    Converts (batch, 1, freq, time) tensors into (n_samples, freq*time) arrays
    suitable for sklearn models.
    """
    all_features = []
    all_labels = []

    for batch_features, batch_labels in data_loader:
        # batch_features shape: (batch, 1, freq, time)
        batch_flat = batch_features.numpy().reshape(batch_features.size(0), -1)
        all_features.append(batch_flat)
        all_labels.append(batch_labels.numpy().flatten())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0).astype(int)
    return X, y


def build_svm():
    """
    Build an SVM pipeline with feature scaling.

    Returns:
        sklearn Pipeline (StandardScaler + SVC)
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=config.SVM_KERNEL,
            C=config.SVM_C,
            gamma=config.SVM_GAMMA,
            probability=True,
        )),
    ])

    print(f"Model: SVM")
    print(f"  Kernel: {config.SVM_KERNEL}")
    print(f"  C: {config.SVM_C}")
    print(f"  Gamma: {config.SVM_GAMMA}")

    return model


def build_logreg():
    """
    Build a Logistic Regression pipeline with feature scaling.

    Returns:
        sklearn Pipeline (StandardScaler + LogisticRegression)
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            C=config.LOGREG_C,
            max_iter=config.LOGREG_MAX_ITER,
            random_state=config.SEED,
        )),
    ])

    print(f"Model: Logistic Regression")
    print(f"  C: {config.LOGREG_C}")
    print(f"  Max iter: {config.LOGREG_MAX_ITER}")

    return model


def train_sklearn_model(model, train_loader, val_loader):
    """
    Train an sklearn model using data from PyTorch DataLoaders.

    Args:
        model: sklearn Pipeline (SVM or LogReg)
        train_loader: training DataLoader
        val_loader: validation DataLoader

    Returns:
        trained model
    """
    print("\nExtracting and flattening features...")
    X_train, y_train = _flatten_loader(train_loader)
    X_val, y_val = _flatten_loader(val_loader)

    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]} samples")

    print("Training...")
    model.fit(X_train, y_train)

    # Report validation accuracy
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"  Validation accuracy: {val_acc:.4f}")

    return model


def evaluate_sklearn_model(model, data_loader):
    """
    Evaluate an sklearn model and return results in the same format
    as evaluation.metrics.evaluate_model for consistency.

    Args:
        model: trained sklearn Pipeline
        data_loader: test DataLoader

    Returns:
        dict with accuracy, precision, recall, f1, confusion_matrix, etc.
    """
    X, y = _flatten_loader(data_loader)

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    results = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y, preds),
        "predictions": preds,
        "labels": y,
        "probabilities": probs,
    }

    return results
