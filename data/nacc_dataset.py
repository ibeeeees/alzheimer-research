"""PyTorch datasets for NACC/SCAN MRI data.

NACCMRIDataset       — cross-sectional (one scan per sample), for Phase 1
NACCLongitudinalDataset — longitudinal (variable-length sequences), for Phase 2
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .label_construction import build_survival_targets, map_cdr_to_ordinal


class NACCMRIDataset(Dataset):
    """Cross-sectional MRI dataset (one scan = one sample).

    Used in Phase 1 (ordinal CDR pretraining).  Each item returns a
    single 3D volume and its ordinal label.

    The manifest CSV must have columns:
        naccid, visit_date, cdr, nii_path
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        volume_shape: Tuple[int, int, int] = (128, 128, 128),
        transform: Optional[Callable] = None,
        subject_ids: Optional[np.ndarray] = None,
    ):
        """
        Args:
            manifest_csv: Path to CSV with columns [naccid, visit_date, cdr, nii_path].
            volume_shape: Target (D, H, W) after preprocessing.
            transform: Optional augmentation callable (volume → volume).
            subject_ids: If provided, filter to these subjects only (for splits).
        """
        df = pd.read_csv(manifest_csv)
        if subject_ids is not None:
            df = df[df["naccid"].isin(subject_ids)].reset_index(drop=True)
        self.df = df
        self.volume_shape = volume_shape
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load and preprocess NIfTI volume
        volume = self._load_volume(row["nii_path"])
        label = map_cdr_to_ordinal(row["cdr"])

        if self.transform is not None:
            volume = self.transform(volume)

        return {
            "volume": volume,                          # (1, D, H, W)
            "label": torch.tensor(label, dtype=torch.long),
            "naccid": row["naccid"],
        }

    def _load_volume(self, path: str) -> torch.Tensor:
        """Load NIfTI, resize, normalise → (1, D, H, W) float32 tensor."""
        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)

        # Resize via zoom if needed
        if data.shape != self.volume_shape:
            from scipy.ndimage import zoom
            factors = tuple(t / s for t, s in zip(self.volume_shape, data.shape))
            data = zoom(data, factors, order=1)

        # Z-score normalise (brain voxels only)
        mask = data > 0
        if mask.any():
            mean = data[mask].mean()
            std = data[mask].std() + 1e-8
            data = (data - mean) / std
        data = np.clip(data, -3.0, 3.0)

        return torch.from_numpy(data).unsqueeze(0)  # (1, D, H, W)


class NACCLongitudinalDataset(Dataset):
    """Longitudinal MRI dataset (variable-length sequences per subject).

    Used in Phase 2.  Each item returns a padded sequence of volumes,
    time deltas, ordinal labels, and optionally survival targets.

    The manifest CSV must have columns:
        naccid, visit_date, cdr, nii_path
    and optionally: event, event_time_months (for survival).
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        volume_shape: Tuple[int, int, int] = (128, 128, 128),
        transform: Optional[Callable] = None,
        subject_ids: Optional[np.ndarray] = None,
        max_visits: int = 8,
        num_survival_intervals: int = 6,
        interval_months: int = 6,
    ):
        df = pd.read_csv(manifest_csv)
        df["visit_date"] = pd.to_datetime(df["visit_date"])
        if subject_ids is not None:
            df = df[df["naccid"].isin(subject_ids)].reset_index(drop=True)
        df = df.sort_values(["naccid", "visit_date"])

        self.volume_shape = volume_shape
        self.transform = transform
        self.max_visits = max_visits
        self.num_survival_intervals = num_survival_intervals
        self.interval_months = interval_months

        # Group by subject
        self.subjects: List[Dict] = []
        for naccid, group in df.groupby("naccid"):
            group = group.sort_values("visit_date").reset_index(drop=True)
            baseline_date = group.iloc[0]["visit_date"]
            visits = []
            for _, row in group.iterrows():
                dt_months = (row["visit_date"] - baseline_date).days / 30.44
                visits.append({
                    "nii_path": row["nii_path"],
                    "cdr": float(row["cdr"]),
                    "dt_months": dt_months,
                })
            # Truncate to max_visits (keep most recent)
            if len(visits) > max_visits:
                visits = visits[-max_visits:]
                # Recompute deltas relative to new first visit
                base_dt = visits[0]["dt_months"]
                for v in visits:
                    v["dt_months"] -= base_dt

            entry = {
                "naccid": naccid,
                "visits": visits,
                "baseline_cdr": visits[0]["cdr"],
                "latest_cdr": visits[-1]["cdr"],
            }

            # Survival targets (if available in CSV)
            if "event" in group.columns:
                event = int(group.iloc[0]["event"])
                event_time = float(group.iloc[0]["event_time_months"])
                entry["event"] = event
                entry["event_time_months"] = event_time

            self.subjects.append(entry)

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        subj = self.subjects[idx]
        visits = subj["visits"]
        n_visits = len(visits)

        # Load volumes and build tensors
        volumes = []
        time_deltas = []
        for v in visits:
            vol = self._load_volume(v["nii_path"])
            if self.transform is not None:
                vol = self.transform(vol)
            volumes.append(vol)
            time_deltas.append(v["dt_months"])

        # Pad to max_visits
        vol_shape = volumes[0].shape  # (1, D, H, W)
        while len(volumes) < self.max_visits:
            volumes.append(torch.zeros(vol_shape))
            time_deltas.append(0.0)

        result = {
            "volumes": torch.stack(volumes),                            # (T, 1, D, H, W)
            "time_deltas": torch.tensor(time_deltas, dtype=torch.float32),  # (T,)
            "lengths": torch.tensor(n_visits, dtype=torch.long),
            "label": torch.tensor(
                map_cdr_to_ordinal(subj["latest_cdr"]), dtype=torch.long
            ),
            "naccid": subj["naccid"],
        }

        # Survival targets
        if "event" in subj:
            ei, arm = build_survival_targets(
                subj["event"],
                subj["event_time_months"],
                self.num_survival_intervals,
                self.interval_months,
            )
            result["event_indicators"] = torch.from_numpy(ei)
            result["at_risk_mask"] = torch.from_numpy(arm)
            result["is_mci"] = torch.tensor(
                1 if subj["baseline_cdr"] == 0.5 else 0, dtype=torch.long
            )

        return result

    def _load_volume(self, path: str) -> torch.Tensor:
        """Load NIfTI, resize, normalise → (1, D, H, W) float32 tensor."""
        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)

        if data.shape != self.volume_shape:
            from scipy.ndimage import zoom
            factors = tuple(t / s for t, s in zip(self.volume_shape, data.shape))
            data = zoom(data, factors, order=1)

        mask = data > 0
        if mask.any():
            mean = data[mask].mean()
            std = data[mask].std() + 1e-8
            data = (data - mean) / std
        data = np.clip(data, -3.0, 3.0)

        return torch.from_numpy(data).unsqueeze(0)
