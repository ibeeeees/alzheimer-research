"""Label construction utilities for the longitudinal multi-task system.

Handles:
  - CDR → ordinal class mapping
  - MCI → AD conversion label construction with censoring
  - Discrete-time survival target tensors
  - Subject-level train/val/test splitting
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── CDR → Ordinal Mapping ────────────────────────────────────────────

CDR_TO_CLASS = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3, 3.0: 3}


def map_cdr_to_ordinal(cdr: float) -> int:
    """Map a CDR global score to ordinal class index {0,1,2,3}.

    CDR 3.0 is merged with CDR 2.0 into class 3 (ModerateToSevere).
    """
    if cdr in CDR_TO_CLASS:
        return CDR_TO_CLASS[cdr]
    # Handle edge cases (e.g. CDR 0.0 stored as int)
    cdr_f = float(cdr)
    if cdr_f in CDR_TO_CLASS:
        return CDR_TO_CLASS[cdr_f]
    raise ValueError(f"Unknown CDR value: {cdr}")


# ── Conversion Label Construction ────────────────────────────────────

def build_conversion_labels(
    visits_df: pd.DataFrame,
    window_months: int = 36,
    mci_cdr: float = 0.5,
    ad_cdr_threshold: float = 1.0,
    subject_col: str = "NACCID",
    date_col: str = "VISITDATE",
    cdr_col: str = "CDRGLOB",
    diagnosis_col: str = "NACCUDSD",
) -> pd.DataFrame:
    """Construct conversion labels from longitudinal visit data.

    For each subject whose baseline CDR == mci_cdr:
      - Converter:     CDR transitions to ≥ ad_cdr_threshold within window.
      - Stable MCI:    CDR stays < ad_cdr_threshold, follow-up ≥ window.
      - Right-censored: CDR stays < ad_cdr_threshold, follow-up < window.

    Args:
        visits_df: NACC UDS longitudinal dataframe (one row per visit).
            Must contain subject_col, date_col, cdr_col columns.
        window_months: Conversion observation window in months.
        mci_cdr: CDR value defining MCI at baseline.
        ad_cdr_threshold: CDR threshold defining conversion.
        subject_col: Column name for subject ID.
        date_col: Column name for visit date.
        cdr_col: Column name for global CDR score.
        diagnosis_col: Column name for clinical diagnosis.

    Returns:
        DataFrame indexed by subject with columns:
          baseline_date, event (0/1), event_time_months, last_followup_months,
          status ('converter', 'stable', 'censored')
    """
    df = visits_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([subject_col, date_col])

    records = []
    for subj_id, group in df.groupby(subject_col):
        group = group.sort_values(date_col).reset_index(drop=True)
        baseline_row = group.iloc[0]
        baseline_cdr = float(baseline_row[cdr_col])
        baseline_date = baseline_row[date_col]

        # Only process MCI-at-baseline subjects
        if baseline_cdr != mci_cdr:
            continue

        # Find conversion event
        event = 0
        event_time = None
        last_followup = 0.0

        for _, row in group.iloc[1:].iterrows():
            months_elapsed = (row[date_col] - baseline_date).days / 30.44
            last_followup = months_elapsed
            cdr_val = float(row[cdr_col])

            if cdr_val >= ad_cdr_threshold and months_elapsed <= window_months:
                event = 1
                event_time = months_elapsed
                break

        if event == 1:
            status = "converter"
            time = event_time
        elif last_followup >= window_months:
            status = "stable"
            time = window_months
        else:
            status = "censored"
            time = last_followup

        records.append({
            subject_col: subj_id,
            "baseline_date": baseline_date,
            "event": event,
            "event_time_months": time,
            "last_followup_months": last_followup,
            "status": status,
        })

    return pd.DataFrame(records)


# ── Discrete-Time Survival Targets ───────────────────────────────────

def build_survival_targets(
    event: int,
    event_time_months: float,
    num_intervals: int = 6,
    interval_months: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build discrete-time survival target arrays for a single subject.

    Args:
        event: 1 if converter, 0 if stable/censored.
        event_time_months: Time of event or last observation in months.
        num_intervals: Number of discrete intervals (default 6 for 36 months).
        interval_months: Duration of each interval in months (default 6).

    Returns:
        event_indicators: (J,) array — 1 only in the event interval.
        at_risk_mask:     (J,) array — 1 for intervals subject is at risk.
    """
    event_indicators = np.zeros(num_intervals, dtype=np.float32)
    at_risk_mask = np.zeros(num_intervals, dtype=np.float32)

    for j in range(num_intervals):
        interval_start = j * interval_months
        interval_end = (j + 1) * interval_months

        if event_time_months < interval_start:
            # Already converted or censored before this interval
            break

        # Subject is at risk in this interval
        at_risk_mask[j] = 1.0

        if event == 1 and event_time_months <= interval_end:
            # Event occurs in this interval
            event_indicators[j] = 1.0
            break

    return event_indicators, at_risk_mask


# ── Subject-Level Splitting ──────────────────────────────────────────

def subject_level_split(
    subjects: np.ndarray,
    labels: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Stratified subject-level split into train/val/test.

    Args:
        subjects: (N,) array of unique subject IDs.
        labels:   (N,) ordinal class labels for stratification.
        train_frac: Training fraction.
        val_frac:   Validation fraction.
        seed: Random seed.

    Returns:
        dict with keys 'train', 'val', 'test' → arrays of subject IDs.
    """
    rng = np.random.RandomState(seed)
    unique_labels = np.unique(labels)

    train_ids, val_ids, test_ids = [], [], []

    for label in unique_labels:
        mask = labels == label
        subj_in_class = subjects[mask].copy()
        rng.shuffle(subj_in_class)

        n = len(subj_in_class)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_ids.extend(subj_in_class[:n_train])
        val_ids.extend(subj_in_class[n_train : n_train + n_val])
        test_ids.extend(subj_in_class[n_train + n_val :])

    return {
        "train": np.array(train_ids),
        "val": np.array(val_ids),
        "test": np.array(test_ids),
    }
