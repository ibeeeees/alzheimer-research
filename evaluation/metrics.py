"""Evaluation metrics for all three tasks.

Ordinal CDR:      QWK, MAE, accuracy, off-by-k, ECE, per-class F1
Survival:         C-index, time-dependent AUC, Brier score
Amyloid:          AUROC, AUPRC, F1
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ══════════════════════════════════════════════════════════════════════
# Ordinal CDR Metrics
# ══════════════════════════════════════════════════════════════════════

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic Weighted Kappa — primary ordinal metric."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true.astype(float) - y_pred.astype(float)))


def compute_off_by_k(y_true: np.ndarray, y_pred: np.ndarray,
                      k: int = 1) -> float:
    """Fraction of predictions within k ordinal classes of truth."""
    return np.mean(np.abs(y_true - y_pred) <= k)


def compute_ece(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.

    Args:
        y_true:  (N,) integer labels.
        y_probs: (N, K) predicted class probabilities.
    """
    pred_labels = y_probs.argmax(axis=1)
    confidences = y_probs.max(axis=1)
    accuracies = (pred_labels == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += mask.sum() / len(y_true) * abs(avg_acc - avg_conf)
    return ece


def compute_all_ordinal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    num_classes: int = 4,
) -> Dict[str, float]:
    """Compute all ordinal CDR metrics."""
    metrics = {
        "accuracy": compute_accuracy(y_true, y_pred),
        "qwk": compute_qwk(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "off_by_1": compute_off_by_k(y_true, y_pred, 1),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_probs is not None:
        metrics["ece"] = compute_ece(y_true, y_probs)
    return metrics


# ══════════════════════════════════════════════════════════════════════
# Survival Metrics
# ══════════════════════════════════════════════════════════════════════

def compute_c_index(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    """Harrell's concordance index.

    Args:
        event_times:      (N,) observed times.
        event_indicators: (N,) 1=event, 0=censored.
        risk_scores:      (N,) higher = more risk (e.g. 1-survival).
    """
    n = len(event_times)
    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(n):
        if event_indicators[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if event_times[j] > event_times[i]:
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] < risk_scores[j]:
                    discordant += 1
                else:
                    tied_risk += 1

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied_risk) / total


def compute_time_dependent_auc(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    risk_scores: np.ndarray,
    eval_time: float,
) -> float:
    """Time-dependent AUC at a specific evaluation time.

    Uses the incident/dynamic definition:
      Cases:    event in (0, eval_time]
      Controls: event-free at eval_time
    """
    cases = (event_times <= eval_time) & (event_indicators == 1)
    controls = event_times > eval_time  # still at risk at eval_time

    if cases.sum() < 1 or controls.sum() < 1:
        return float("nan")

    y_true = np.concatenate([np.ones(cases.sum()), np.zeros(controls.sum())])
    y_score = np.concatenate([risk_scores[cases], risk_scores[controls]])

    return roc_auc_score(y_true, y_score)


def compute_brier_score(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    predicted_probs: np.ndarray,
    eval_time: float,
) -> float:
    """Brier score at a specific evaluation time.

    BS(t) = (1/N) Σ_i (P_hat(T<=t|x_i) - 𝟙[T_i<=t, δ_i=1])²

    (Uncensored version; for the full IPCW version, use lifelines.)
    """
    actual = ((event_times <= eval_time) & (event_indicators == 1)).astype(float)
    return np.mean((predicted_probs - actual) ** 2)


def compute_all_survival_metrics(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    risk_scores: np.ndarray,
    predicted_survival: Optional[np.ndarray] = None,
    eval_times: Optional[list] = None,
) -> Dict[str, float]:
    """Compute all survival metrics.

    Args:
        event_times:        (N,) observed event/censor times in months.
        event_indicators:   (N,) 1=event, 0=censored.
        risk_scores:        (N,) predicted risk (higher=worse).
        predicted_survival: (N, J) predicted survival probabilities.
        eval_times:         List of timepoints for td-AUC (default [12, 24, 36]).
    """
    if eval_times is None:
        eval_times = [12.0, 24.0, 36.0]

    metrics = {
        "c_index": compute_c_index(event_times, event_indicators, risk_scores),
    }

    for t in eval_times:
        auc = compute_time_dependent_auc(
            event_times, event_indicators, risk_scores, t
        )
        metrics[f"td_auc_{int(t)}mo"] = auc

    if predicted_survival is not None:
        for i, t in enumerate(eval_times):
            if i < predicted_survival.shape[1]:
                conv_prob = 1.0 - predicted_survival[:, i]
                bs = compute_brier_score(
                    event_times, event_indicators, conv_prob, t
                )
                metrics[f"brier_{int(t)}mo"] = bs

    return metrics


# ══════════════════════════════════════════════════════════════════════
# Amyloid Metrics
# ══════════════════════════════════════════════════════════════════════

def compute_amyloid_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Binary classification metrics for amyloid positivity."""
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "auroc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "auprc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }

    # Sensitivity at 90% specificity
    if len(np.unique(y_true)) > 1:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        idx = np.where(fpr <= 0.10)[0]
        metrics["sens_at_90spec"] = tpr[idx[-1]] if len(idx) > 0 else 0.0

    return metrics


# ══════════════════════════════════════════════════════════════════════
# Post-hoc Threshold Optimisation
# ══════════════════════════════════════════════════════════════════════

def optimize_thresholds(
    severity_scores: np.ndarray,
    true_labels: np.ndarray,
    num_classes: int = 4,
    n_grid: int = 200,
) -> Tuple[list, float]:
    """Grid search for optimal CORAL thresholds on validation data.

    Finds thresholds that maximise QWK.

    Args:
        severity_scores: (N,) scalar severity from model.
        true_labels:     (N,) integer ordinal labels.
        num_classes: K.
        n_grid: Grid density per threshold.

    Returns:
        (best_thresholds, best_qwk)
    """
    lo = severity_scores.min() - 0.1 * abs(severity_scores.min())
    hi = severity_scores.max() + 0.1 * abs(severity_scores.max())
    grid = np.linspace(lo, hi, n_grid)

    num_thresholds = num_classes - 1

    if num_thresholds == 3:
        # Exhaustive 3D grid for 4-class case
        best_qwk = -2.0
        best_thresholds = [0.0] * 3

        for i, t1 in enumerate(grid):
            for j, t2 in enumerate(grid[i + 1 :], start=i + 1):
                for t3 in grid[j + 1 :]:
                    preds = np.zeros(len(severity_scores), dtype=int)
                    preds[severity_scores > t1] = 1
                    preds[severity_scores > t2] = 2
                    preds[severity_scores > t3] = 3
                    qwk = compute_qwk(true_labels, preds)
                    if qwk > best_qwk:
                        best_qwk = qwk
                        best_thresholds = [t1, t2, t3]

        return best_thresholds, best_qwk
    else:
        # Greedy sequential for general case
        thresholds = []
        remaining_grid = grid.copy()
        for k in range(num_thresholds):
            best_t = remaining_grid[0]
            best_q = -2.0
            for t in remaining_grid:
                trial = thresholds + [t]
                trial_sorted = sorted(trial)
                preds = np.zeros(len(severity_scores), dtype=int)
                for th in trial_sorted:
                    preds[severity_scores > th] += 1
                preds = np.clip(preds, 0, num_classes - 1)
                q = compute_qwk(true_labels, preds)
                if q > best_q:
                    best_q = q
                    best_t = t
            thresholds.append(best_t)
            remaining_grid = remaining_grid[remaining_grid > best_t]
            if len(remaining_grid) == 0:
                remaining_grid = np.array([best_t + 0.01])

        return sorted(thresholds), best_q
