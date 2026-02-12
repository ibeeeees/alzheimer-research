"""Ordinal regression utilities: CORAL loss, learnable thresholds, and evaluation metrics."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
)


class OrdinalHead(nn.Module):
    """Ordinal regression head using the CORAL framework.

    Maps a scalar severity score to ordered class probabilities
    via K-1 learnable thresholds.
    """

    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        # Learnable bias thresholds (b1, b2, ..., b_{K-1})
        # Initialized linearly spaced for stable training start
        init_vals = torch.linspace(-1.0, 1.0, self.num_thresholds)
        self.thresholds = nn.Parameter(init_vals)

    def forward(self, severity_score):
        """Compute cumulative logits P(Y >= k) for each threshold.

        Args:
            severity_score: (B, 1) continuous severity output from backbone

        Returns:
            cumulative_logits: (B, K-1) logits for P(Y >= k)
        """
        # severity_score: (B, 1), thresholds: (K-1,)
        # cumulative_logits[i, k] = severity_score[i] - threshold[k]
        cumulative_logits = severity_score - self.thresholds.unsqueeze(0)
        return cumulative_logits

    def predict(self, severity_score):
        """Get predicted ordinal class from severity score.

        Args:
            severity_score: (B, 1) continuous severity

        Returns:
            predicted_class: (B,) integer class labels
        """
        logits = self.forward(severity_score)
        # P(Y >= k) > 0.5 means logit > 0
        cumulative_preds = (logits > 0).float()
        # Predicted class = number of thresholds exceeded
        predicted_class = cumulative_preds.sum(dim=1).long()
        return predicted_class

    def class_probabilities(self, severity_score):
        """Compute probability for each class.

        Args:
            severity_score: (B, 1) continuous severity

        Returns:
            probs: (B, K) probability distribution over classes
        """
        logits = self.forward(severity_score)
        cum_probs = torch.sigmoid(logits)  # P(Y >= k) for k=1,...,K-1

        # P(Y = 0) = 1 - P(Y >= 1)
        # P(Y = k) = P(Y >= k) - P(Y >= k+1) for 0 < k < K-1
        # P(Y = K-1) = P(Y >= K-1)
        probs = torch.zeros(severity_score.size(0), self.num_classes,
                            device=severity_score.device)
        probs[:, 0] = 1.0 - cum_probs[:, 0]
        for k in range(1, self.num_thresholds):
            probs[:, k] = cum_probs[:, k - 1] - cum_probs[:, k]
        probs[:, -1] = cum_probs[:, -1]

        # Clamp for numerical stability
        probs = probs.clamp(min=1e-7)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs


def coral_ordinal_loss(cumulative_logits, labels, num_classes=4):
    """CORAL (Consistent Rank Logits) ordinal loss.

    Binary cross-entropy applied independently to each of the K-1
    cumulative threshold comparisons.

    Args:
        cumulative_logits: (B, K-1) raw logits from OrdinalHead
        labels: (B,) integer class labels in {0, 1, ..., K-1}
        num_classes: number of ordered classes

    Returns:
        loss: scalar loss value
    """
    num_thresholds = num_classes - 1
    batch_size = labels.size(0)

    # Create binary targets: target[i, k] = 1 if label[i] >= k+1
    targets = torch.zeros(batch_size, num_thresholds,
                          device=labels.device, dtype=torch.float32)
    for k in range(num_thresholds):
        targets[:, k] = (labels > k).float()

    loss = F.binary_cross_entropy_with_logits(cumulative_logits, targets)
    return loss


def categorical_cross_entropy_loss(severity_score, ordinal_head, labels):
    """Standard cross-entropy baseline for comparison with CORAL.

    Args:
        severity_score: (B, 1) from backbone
        ordinal_head: OrdinalHead module
        labels: (B,) integer class labels

    Returns:
        loss: scalar CE loss
    """
    probs = ordinal_head.class_probabilities(severity_score)
    log_probs = torch.log(probs + 1e-7)
    loss = F.nll_loss(log_probs, labels)
    return loss


def ordinal_predictions(severity_score, ordinal_head):
    """Get class predictions from severity scores.

    Args:
        severity_score: (B, 1) continuous severity
        ordinal_head: OrdinalHead module

    Returns:
        preds: (B,) predicted class indices
    """
    return ordinal_head.predict(severity_score)


# ---------- Evaluation Metrics ----------


def compute_accuracy(y_true, y_pred):
    """Standard classification accuracy."""
    return accuracy_score(y_true, y_pred)


def compute_qwk(y_true, y_pred):
    """Quadratic Weighted Kappa — standard metric for ordinal agreement."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_mae(y_true, y_pred):
    """Mean Absolute Error between ordinal class indices."""
    return mean_absolute_error(y_true, y_pred)


def compute_off_by_k(y_true, y_pred, k=1):
    """Fraction of predictions within k classes of the true label."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) <= k))


def compute_ece(y_true, y_probs, n_bins=10):
    """Expected Calibration Error.

    Args:
        y_true: (N,) true class labels
        y_probs: (N, K) predicted class probabilities
        n_bins: number of calibration bins

    Returns:
        ece: scalar ECE value
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    # Get predicted class and its confidence
    pred_classes = y_probs.argmax(axis=1)
    confidences = y_probs.max(axis=1)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = (pred_classes[mask] == y_true[mask]).mean()
        bin_conf = confidences[mask].mean()
        bin_weight = mask.sum() / total
        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def compute_confusion_matrix(y_true, y_pred, num_classes=4):
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def compute_all_metrics(y_true, y_pred, y_probs=None, num_classes=4):
    """Compute all evaluation metrics.

    Args:
        y_true: (N,) true labels
        y_pred: (N,) predicted labels
        y_probs: (N, K) predicted probabilities (optional, for ECE)
        num_classes: number of classes

    Returns:
        dict of metric_name -> value
    """
    metrics = {
        "accuracy": compute_accuracy(y_true, y_pred),
        "qwk": compute_qwk(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "off_by_1": compute_off_by_k(y_true, y_pred, k=1),
        "off_by_2": compute_off_by_k(y_true, y_pred, k=2),
    }

    if y_probs is not None:
        metrics["ece"] = compute_ece(y_true, y_probs)

    metrics["confusion_matrix"] = compute_confusion_matrix(
        y_true, y_pred, num_classes
    )

    return metrics


# ---------- Threshold Optimization ----------


def optimize_thresholds(severity_scores, true_labels, num_classes=4, n_grid=200):
    """Post-hoc threshold optimization via grid search on validation data.

    Finds thresholds that maximize QWK on the validation set.

    Args:
        severity_scores: (N,) numpy array of continuous severity scores
        true_labels: (N,) numpy array of integer labels
        num_classes: number of ordered classes
        n_grid: number of grid points per threshold

    Returns:
        best_thresholds: list of K-1 optimal thresholds
        best_qwk: QWK achieved with optimal thresholds
    """
    severity_scores = np.asarray(severity_scores)
    true_labels = np.asarray(true_labels)

    s_min, s_max = severity_scores.min(), severity_scores.max()
    margin = (s_max - s_min) * 0.1
    grid = np.linspace(s_min - margin, s_max + margin, n_grid)

    num_thresholds = num_classes - 1
    best_qwk = -1.0
    best_thresholds = [0.0] * num_thresholds

    if num_thresholds == 3:
        # 3-threshold grid search (4 classes)
        for t1 in grid:
            for t2 in grid:
                if t2 <= t1:
                    continue
                for t3 in grid:
                    if t3 <= t2:
                        continue
                    preds = np.zeros_like(true_labels)
                    preds[severity_scores > t1] = 1
                    preds[severity_scores > t2] = 2
                    preds[severity_scores > t3] = 3
                    qwk = cohen_kappa_score(true_labels, preds, weights="quadratic")
                    if qwk > best_qwk:
                        best_qwk = qwk
                        best_thresholds = [t1, t2, t3]
    else:
        # General case: greedy sequential threshold placement
        thresholds = []
        for _ in range(num_thresholds):
            best_t = grid[0]
            best_k_qwk = -1.0
            for t in grid:
                trial = thresholds + [t]
                if len(trial) > 1 and t <= trial[-2]:
                    continue
                preds = np.zeros_like(true_labels)
                for i, thresh in enumerate(trial):
                    preds[severity_scores > thresh] = i + 1
                qwk = cohen_kappa_score(true_labels, preds, weights="quadratic")
                if qwk > best_k_qwk:
                    best_k_qwk = qwk
                    best_t = t
            thresholds.append(best_t)
        best_thresholds = thresholds
        # Compute final QWK
        preds = np.zeros_like(true_labels)
        for i, thresh in enumerate(best_thresholds):
            preds[severity_scores > thresh] = i + 1
        best_qwk = cohen_kappa_score(true_labels, preds, weights="quadratic")

    return best_thresholds, best_qwk
