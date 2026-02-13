# Longitudinal Multi-Task Alzheimer's Staging via Cross-Cohort Neuroimaging and Speech Biomarkers

A multi-task, longitudinal, cross-cohort predictive modeling system for Alzheimer's disease staging and progression, built on the SCAN (NACC) dataset and DementiaBank.

## Overview

This system jointly optimizes three clinically relevant objectives:

| Task | Method | Output |
|------|--------|--------|
| **Ordinal CDR severity** | CORAL ordinal regression (4 classes, 3 thresholds) | CDR stage {0, 0.5, 1.0, 2.0+} |
| **MCI-to-AD conversion** | Discrete-time survival model (6 intervals, 36-month window) | Time-varying conversion probability |
| **Cross-cohort alignment** | Class-conditioned MMD | Shared latent space across modalities |

### Data Sources

| Source | Modality | Scale | Subjects |
|--------|----------|-------|----------|
| SCAN (NACC) | 3D T1-weighted MRI | ~29,000 scans | ~10,000 (longitudinal) |
| DementiaBank (Pitt Corpus) | Speech recordings | ~550 recordings | ~270 |

Subjects do **not** overlap between cohorts. The system uses cross-cohort alignment via shared ordinal heads and MMD to bridge representations.

---

## Architecture

```
NACC MRI (longitudinal)                    DementiaBank (speech)
     │                                            │
  3D ResNet-18                              wav2vec 2.0 + SBERT
  (per-visit encoder)                       + handcrafted features
     │                                            │
  Time-aware GRU                           MLP projection
  (visit aggregation)                      (1382-D → 256-D)
     │                                            │
     └──────────── Shared Latent Space ───────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
     CORAL Ordinal   Survival Head   Amyloid Head
     (CDR staging)   (MCI→AD risk)   (Aβ+ binary)
```

Multi-task losses are balanced via learned homoscedastic uncertainty weights (Kendall et al., 2018).

---

## Project Structure

```
├── config.py                     # Central configuration
├── requirements.txt
├── ARCHITECTURE_PROPOSAL.md      # Full technical design document
│
├── models/
│   ├── mri_encoder.py            # 3D ResNet-18 (single-channel adapted)
│   ├── temporal_module.py        # Sinusoidal time encoding + GRU
│   ├── speech_encoder.py         # MLP on pre-extracted features
│   ├── task_heads.py             # OrdinalHead, SurvivalHead, AmyloidHead
│   ├── losses.py                 # CORAL, survival, amyloid, multi-task
│   ├── alignment.py              # Class-conditioned MMD
│   └── full_model.py             # AlzheimerMultiTaskModel (unified)
│
├── data/
│   ├── label_construction.py     # CDR mapping, conversion labels, splits
│   ├── nacc_dataset.py           # Cross-sectional + longitudinal datasets
│   ├── speech_dataset.py         # DementiaBank feature dataset
│   └── preprocessing.py          # Volume preprocessing, augmentation, feature extraction
│
├── training/
│   ├── trainer.py                # Phase1Trainer, Phase2Trainer
│   └── callbacks.py              # EarlyStopping, CheckpointManager
│
├── evaluation/
│   ├── metrics.py                # QWK, C-index, td-AUC, Brier, ECE, etc.
│   └── visualization.py          # Confusion matrices, survival curves, t-SNE
│
└── notebooks/
    ├── 01_preprocess_nacc_mri.ipynb
    ├── 02_extract_speech_features.ipynb
    └── 03_train_and_evaluate.ipynb
```

---

## Training

Training proceeds in two phases:

### Phase 1: MRI Ordinal Pretraining
- Single-task (CORAL ordinal loss only)
- Cross-sectional (one scan per sample)
- 30 epochs, batch 8, gradient accumulation to effective 16
- ~9 hours on Colab T4

### Phase 2: Full Multi-Task
- All losses active (ordinal + survival + alignment)
- Longitudinal sequences (variable-length, time-aware GRU)
- Speech batches interleaved for alignment
- Differential learning rates (backbone 1e-4, heads 5e-4)
- ~50 hours across 3-4 Colab Pro sessions with checkpointing

### Compute Requirements
- Google Colab T4 GPU (16 GB VRAM)
- ~2.4 GB VRAM at batch 4, 128³ resolution with AMP
- Mixed precision (fp16) throughout
- Checkpoint-resume for multi-session training

---

## Evaluation

| Task | Primary Metric | Additional |
|------|---------------|------------|
| CDR staging | QWK (Quadratic Weighted Kappa) | MAE, off-by-1, ECE, macro-F1 |
| MCI→AD conversion | C-index | Time-dependent AUC at 12/24/36 mo, Brier score |
| Amyloid positivity | AUROC | AUPRC, sensitivity @ 90% specificity |
| Alignment quality | Class-conditioned MMD | t-SNE visualization |

---

## Quick Start

```bash
pip install -r requirements.txt
```

Run notebooks in order:
1. `01_preprocess_nacc_mri.ipynb` — Build manifest CSV from NACC data
2. `02_extract_speech_features.ipynb` — Extract speech biomarkers
3. `03_train_and_evaluate.ipynb` — Train and evaluate (includes synthetic sanity check)

---

## Key Design Decisions

- **Cross-cohort modeling**: MRI and speech datasets don't share subjects. Alignment via shared CORAL head + class-conditioned MMD.
- **Ordinal regression**: CORAL preserves CDR ordering; misclassifying CDR 0 as CDR 2 is penalized more than CDR 0 as CDR 0.5.
- **Discrete-time survival**: Handles right-censored subjects natively. More informative than binary 36-month conversion.
- **Uncertainty-weighted losses**: Learned task weights prevent noisy tasks from destabilizing training.
- **SCAN advantage**: Centralized QC pipeline eliminates site-harmonization confounds present in ADNI/OASIS studies.
