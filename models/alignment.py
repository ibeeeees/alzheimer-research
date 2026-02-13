"""Cross-cohort alignment via class-conditioned Maximum Mean Discrepancy."""

import torch
import torch.nn as nn


def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor,
                     bandwidth: float) -> torch.Tensor:
    """Gaussian RBF kernel k(x,y) = exp(-||x-y||² / 2σ²)."""
    dist_sq = torch.cdist(x, y, p=2.0).pow(2)
    return torch.exp(-dist_sq / (2.0 * bandwidth ** 2))


def _mmd_squared(x: torch.Tensor, y: torch.Tensor,
                 bandwidth: float) -> torch.Tensor:
    """Unbiased estimate of MMD² between samples x and y."""
    n = x.size(0)
    m = y.size(0)
    if n == 0 or m == 0:
        return torch.tensor(0.0, device=x.device)

    k_xx = _gaussian_kernel(x, x, bandwidth)
    k_yy = _gaussian_kernel(y, y, bandwidth)
    k_xy = _gaussian_kernel(x, y, bandwidth)

    # Unbiased estimator: exclude diagonal for k_xx and k_yy
    diag_xx = torch.diagonal(k_xx)
    diag_yy = torch.diagonal(k_yy)

    if n > 1:
        term_xx = (k_xx.sum() - diag_xx.sum()) / (n * (n - 1))
    else:
        term_xx = torch.tensor(0.0, device=x.device)
    if m > 1:
        term_yy = (k_yy.sum() - diag_yy.sum()) / (m * (m - 1))
    else:
        term_yy = torch.tensor(0.0, device=x.device)
    term_xy = k_xy.mean()

    return term_xx + term_yy - 2.0 * term_xy


def _median_bandwidth(x: torch.Tensor, y: torch.Tensor) -> float:
    """Median heuristic for Gaussian kernel bandwidth."""
    with torch.no_grad():
        combined = torch.cat([x, y], dim=0)
        dists = torch.cdist(combined, combined, p=2.0)
        # Take upper triangle (exclude diagonal zeros)
        mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
        median_dist = dists[mask].median().item()
        return max(median_dist, 1e-5)


class ClassConditionedMMD(nn.Module):
    """MMD alignment loss conditioned on ordinal class labels.

    L_align = Σ_k  MMD²( {h_m : y_m = k}, {z_s : y_s = k} )

    Samples from both modalities are matched by label, and MMD is
    computed per class then summed.  Uses the median heuristic for
    kernel bandwidth.
    """

    def __init__(self, num_classes: int = 4, bandwidth: str = "median"):
        super().__init__()
        self.num_classes = num_classes
        self.fixed_bandwidth = None
        if bandwidth != "median":
            self.fixed_bandwidth = float(bandwidth)

    def forward(
        self,
        h_mri: torch.Tensor,
        y_mri: torch.Tensor,
        z_speech: torch.Tensor,
        y_speech: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_mri:     (N_m, D) MRI embeddings
            y_mri:     (N_m,)   ordinal labels
            z_speech:  (N_s, D) speech embeddings
            y_speech:  (N_s,)   ordinal labels
        Returns:
            Scalar MMD loss (sum over classes).
        """
        total_mmd = torch.tensor(0.0, device=h_mri.device)
        n_active = 0

        for k in range(self.num_classes):
            mask_m = y_mri == k
            mask_s = y_speech == k
            x_k = h_mri[mask_m]
            y_k = z_speech[mask_s]
            if x_k.size(0) < 2 or y_k.size(0) < 2:
                continue

            bw = (
                self.fixed_bandwidth
                if self.fixed_bandwidth
                else _median_bandwidth(x_k, y_k)
            )
            total_mmd = total_mmd + _mmd_squared(x_k, y_k, bw)
            n_active += 1

        if n_active > 0:
            total_mmd = total_mmd / n_active
        return total_mmd
