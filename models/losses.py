"""Loss functions for the multi-task system.

1. coral_ordinal_loss   — CORAL ordinal regression
2. discrete_survival_loss — censoring-aware discrete-time survival
3. amyloid_bce_loss     — binary cross-entropy for Aβ+
4. MultiTaskLoss        — uncertainty-weighted combination (Kendall et al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── CORAL Ordinal Loss ────────────────────────────────────────────────

def coral_ordinal_loss(
    cum_logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 4,
) -> torch.Tensor:
    """Consistent Rank Logits (CORAL) ordinal loss.

    L = -(1/(K-1)) Σ_k [ y*_k log σ(ℓ_k) + (1-y*_k) log(1-σ(ℓ_k)) ]
    where y*_k = 𝟙[y ≥ k].

    Args:
        cum_logits: (B, K-1) cumulative logits from OrdinalHead.
        labels:     (B,)     integer class labels in {0, ..., K-1}.
        num_classes: K.
    Returns:
        Scalar loss.
    """
    num_thresholds = num_classes - 1
    # Cumulative binary targets: 1 if label ≥ k (for k = 1 .. K-1)
    targets = torch.zeros_like(cum_logits)
    for k in range(num_thresholds):
        targets[:, k] = (labels > k).float()
    return F.binary_cross_entropy_with_logits(cum_logits, targets)


# ── Discrete-Time Survival Loss ──────────────────────────────────────

def discrete_survival_loss(
    hazard_logits: torch.Tensor,
    event_indicators: torch.Tensor,
    at_risk_masks: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood for discrete-time survival.

    For each subject, we observe them through intervals where they're
    at risk.  This reduces to binary cross-entropy over at-risk intervals.

    L = -Σ_i Σ_{l: at_risk} [ d_il log σ(h_l) + (1-d_il) log(1-σ(h_l)) ]

    Args:
        hazard_logits:   (B, J) raw logits per interval.
        event_indicators: (B, J) binary — 1 iff event occurred in interval j.
        at_risk_masks:   (B, J) binary — 1 iff subject is at risk in interval j.
    Returns:
        Scalar loss (averaged over total at-risk interval-observations).
    """
    # Mask to only at-risk intervals
    bce = F.binary_cross_entropy_with_logits(
        hazard_logits, event_indicators, reduction="none"
    )
    masked = bce * at_risk_masks
    n_obs = at_risk_masks.sum().clamp(min=1.0)
    return masked.sum() / n_obs


# ── Amyloid BCE Loss ─────────────────────────────────────────────────

def amyloid_bce_loss(
    logit: torch.Tensor,
    label: torch.Tensor,
) -> torch.Tensor:
    """Standard binary cross-entropy for amyloid positivity.

    Args:
        logit: (B, 1) raw logit from AmyloidHead.
        label: (B,) or (B, 1) binary label.
    Returns:
        Scalar loss.
    """
    label = label.view_as(logit).float()
    return F.binary_cross_entropy_with_logits(logit, label)


# ── Multi-Task Weighted Loss ─────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """Homoscedastic uncertainty weighting (Kendall et al. 2018).

    L_total = Σ_t [ 1/(2σ_t²) L_t + (1/2)log(σ_t²) ] + λ L_align

    We parameterise s_t = log(σ_t²) for numerical stability.
    Then:  1/(2σ²) = 1/(2 exp(s))  and  log(σ²)/2 = s/2.
    """

    def __init__(
        self,
        init_log_var_ord: float = 0.0,
        init_log_var_surv: float = 0.0,
        init_log_var_amy: float = 0.0,
        alignment_lambda: float = 0.1,
    ):
        super().__init__()
        self.log_var_ord = nn.Parameter(torch.tensor(init_log_var_ord))
        self.log_var_surv = nn.Parameter(torch.tensor(init_log_var_surv))
        self.log_var_amy = nn.Parameter(torch.tensor(init_log_var_amy))
        self.alignment_lambda = alignment_lambda

    def forward(
        self,
        loss_ord: torch.Tensor,
        loss_surv: torch.Tensor | None = None,
        loss_amy: torch.Tensor | None = None,
        loss_align: torch.Tensor | None = None,
        alignment_warmup_factor: float = 1.0,
    ) -> dict:
        """Combine task losses with learned weights.

        Any task loss can be None (skipped for this batch).

        Returns:
            dict with 'total' and per-task weighted components.
        """
        total = torch.tensor(0.0, device=loss_ord.device)
        details = {}

        # Ordinal (always active)
        w_ord = 0.5 * torch.exp(-self.log_var_ord)
        reg_ord = 0.5 * self.log_var_ord
        l_ord_w = w_ord * loss_ord + reg_ord
        total = total + l_ord_w
        details["ord_weighted"] = l_ord_w.detach()
        details["ord_raw"] = loss_ord.detach()
        details["ord_weight"] = w_ord.detach()

        # Survival (MCI subjects only)
        if loss_surv is not None:
            w_surv = 0.5 * torch.exp(-self.log_var_surv)
            reg_surv = 0.5 * self.log_var_surv
            l_surv_w = w_surv * loss_surv + reg_surv
            total = total + l_surv_w
            details["surv_weighted"] = l_surv_w.detach()
            details["surv_raw"] = loss_surv.detach()
            details["surv_weight"] = w_surv.detach()

        # Amyloid (labelled subjects only)
        if loss_amy is not None:
            w_amy = 0.5 * torch.exp(-self.log_var_amy)
            reg_amy = 0.5 * self.log_var_amy
            l_amy_w = w_amy * loss_amy + reg_amy
            total = total + l_amy_w
            details["amy_weighted"] = l_amy_w.detach()
            details["amy_raw"] = loss_amy.detach()
            details["amy_weight"] = w_amy.detach()

        # Alignment (fixed weight with warmup)
        if loss_align is not None:
            l_align = self.alignment_lambda * alignment_warmup_factor * loss_align
            total = total + l_align
            details["align"] = l_align.detach()

        details["total"] = total
        return details
