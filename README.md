# Longitudinal Multi-Task Alzheimer's Staging via Cross-Cohort Neuroimaging and Speech Biomarkers

---

## TL;DR

This system stages Alzheimer's disease severity and predicts MCI-to-AD conversion by jointly learning from **brain MRI** (NACC/SCAN, ~29K scans) and **speech recordings** (DementiaBank, ~550 samples) — two datasets with **no shared subjects**. It bridges the modalities through a shared ordinal classification head and class-conditioned distribution alignment, producing calibrated severity scores and time-varying conversion risk estimates from either modality independently.

---

## Why This System Exists

### The Problem with Current Approaches

Most multimodal Alzheimer's models assume paired data: every subject has both MRI and speech (or PET, or genetics). In practice, this assumption discards 50-80% of available data and doesn't reflect clinical reality, where neuroimaging and cognitive-linguistic assessments are ordered for different patients.

### What We Do Differently

1. **Cross-cohort modeling**: Trains on two entirely separate populations and aligns their representations through shared label structure — no paired samples required.
2. **Ordinal severity, not binary classification**: CDR 0.5 misclassified as CDR 2.0 is clinically worse than CDR 0.5 misclassified as CDR 1.0. Our framework preserves this ordering.
3. **Progression modeling**: Rather than "does this patient have AD?", our system answers "when will this MCI patient convert?" — a fundamentally more useful clinical question.
4. **SCAN over ADNI**: SCAN's centralized acquisition eliminates the site-harmonization problem that plagues virtually all ADNI-based studies.

---

## Data Sources

| Source | Modality | Scale | Subjects | Key Advantage |
|--------|----------|-------|----------|---------------|
| **SCAN (NACC)** | 3D T1-weighted MRI | ~29,000 scans | ~10,000 (longitudinal) | Centralized QC, direct UDS linkage |
| **DementiaBank (Pitt)** | Speech recordings (Cookie Theft) | ~550 recordings | ~270 | Audio + transcripts, clinical staging |

Subjects do **not** overlap between cohorts. This is by design — see [How Cross-Cohort Fusion Works](#how-cross-cohort-fusion-works).

---

## How the MRI Pipeline Works

### Step-by-Step: From Raw Scan to Severity Score

```
Raw NIfTI (.nii.gz)
    │
    ▼
┌─────────────────────────────┐
│  1. PREPROCESSING            │
│  Brain extraction (BET)      │
│  MNI152 registration (FLIRT) │
│  Resample to 128³ voxels     │
│  Z-score normalize           │
│  Clip to [-3, +3] SD         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  2. MRI ENCODER              │
│  3D ResNet-18 (r3d_18)       │
│  Adapted: 3ch → 1ch input    │
│  33.5M params                │
│  Output: 256-D embedding     │
└─────────────┬───────────────┘
              │
    ┌─────────┴──────────┐
    │ Single visit?       │ Multiple visits?
    │                     │
    ▼                     ▼
  z_m ∈ ℝ²⁵⁶     ┌──────────────────┐
  (direct)        │  3. TEMPORAL GRU   │
                  │  Sinusoidal time   │
                  │  encoding (64-D)   │
                  │  handles irregular │
                  │  visit intervals   │
                  │  Output: h_m ∈ ℝ²⁵⁶│
                  └────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  4. TASK HEADS          │
              │  CDR staging (CORAL)    │
              │  MCI→AD survival        │
              │  Amyloid positivity     │
              └────────────────────────┘
```

### MRI Encoder Details

The encoder is a **3D ResNet-18** adapted from the video recognition domain (`torchvision.models.video.r3d_18`). Key adaptations:

- **Input**: Single-channel grayscale MRI volumes (128 x 128 x 128 voxels) instead of 3-channel RGB video frames
- **Channel adaptation**: Pretrained Kinetics-400 RGB weights are averaged across the 3 input channels to initialize the single-channel conv layer: `W_gray = (W_R + W_G + W_B) / 3`
- **Projection**: The 512-D backbone output is projected to 256-D via `Linear → LayerNorm → ReLU → Dropout(0.1)`
- **Parameters**: ~33.5M total

### Temporal Module for Longitudinal Visits

Patients in NACC often have 2-5 MRI scans taken months or years apart at **irregular intervals**. The temporal module handles this:

1. **Sinusoidal time encoding**: Each visit's time offset (in months from baseline) is encoded as a 64-D vector using sinusoidal positional encoding — the same idea as in Transformers, adapted for continuous time
2. **Concatenation**: Each visit's 256-D MRI embedding is concatenated with its 64-D time encoding → 320-D
3. **Input projection**: Linear layer maps 320-D → 256-D
4. **GRU**: Single-layer GRU (256 hidden) processes the visit sequence. The final hidden state captures the patient's longitudinal trajectory
5. **Single-visit fallback**: For patients with only one scan, the GRU simply passes the embedding through — no special handling needed

**Why GRU over Transformer?** With 2-5 visits per patient, the sequential inductive bias of a GRU is more appropriate than a Transformer's attention mechanism, and uses significantly less memory.

### Data Augmentation (Training Only)

| Augmentation | Range | Probability |
|-------------|-------|-------------|
| Random affine rotation | ±10° per axis | 0.5 |
| Random scaling | [0.95, 1.05] | 0.5 |
| Random translation | ±5 voxels | 0.5 |
| Intensity shift | ±0.1 | 0.3 |
| Intensity scale | [0.9, 1.1] | 0.3 |
| Gaussian noise | σ = 0.02 | 0.2 |
| Left-right flip | — | 0.5 |

**Implementation**: `data/preprocessing.py` (MRIPreprocessor, MRIAugmentation classes)

---

## How the Speech Pipeline Works

### Step-by-Step: From Audio to Severity Score

```
Cookie Theft Audio (.wav) + Transcript (.cha / ASR)
    │                              │
    ▼                              ▼
┌──────────────────┐    ┌──────────────────────┐
│ ACOUSTIC STREAM   │    │ LINGUISTIC STREAM     │
│                   │    │                       │
│ wav2vec 2.0 BASE  │    │ Sentence-BERT         │
│ → 768-D embedding │    │ (all-MiniLM-L6-v2)   │
│                   │    │ → 384-D embedding     │
│ Handcrafted:      │    │                       │
│ MFCCs, F0, jitter │    │ Handcrafted:          │
│ shimmer, HNR,     │    │ TTR, MATTR, syntax    │
│ speech rate, etc.  │    │ coherence, fluency    │
│ → 216-D vector    │    │ → 14-D vector         │
└────────┬─────────┘    └──────────┬────────────┘
         │                         │
         └────────┬────────────────┘
                  │
                  ▼
         Concatenate → 1382-D
                  │
                  ▼
    ┌──────────────────────────┐
    │  SPEECH ENCODER (MLP)     │
    │  Linear(1382, 512)        │
    │  → LayerNorm → GELU      │
    │  → Dropout(0.3)          │
    │  Linear(512, 256)         │
    │  → LayerNorm → GELU      │
    │  → Dropout(0.2)          │
    │  Output: z_s ∈ ℝ²⁵⁶      │
    └────────────┬─────────────┘
                 │
                 ▼
    ┌────────────────────────┐
    │  SHARED TASK HEADS      │
    │  (same heads as MRI)    │
    │  CDR staging (CORAL)    │
    └────────────────────────┘
```

### Feature Breakdown

**Acoustic features (216-D)**: Extracted via Parselmouth/OpenSMILE. 54 base features (MFCCs, prosody, voice quality, temporal) × 4 summary statistics (mean, std, skew, kurtosis).

**wav2vec 2.0 (768-D)**: Pretrained self-supervised speech model. Final transformer layer hidden states are average-pooled over time to produce a fixed-length embedding capturing high-level acoustic patterns.

**Linguistic features (14-D)**: From transcripts — lexical diversity (TTR, MATTR, Brunet's W, Honore's R), syntactic complexity (dependency distance, clause density, Yngve depth), semantic coherence, information content, and disfluency rates.

**Sentence-BERT (384-D)**: Encodes the full transcript into a dense semantic embedding capturing overall language quality and content.

**Why a simple MLP?** With only ~550 training samples, a complex architecture would overfit. The heavy lifting is done by the pretrained models (wav2vec 2.0, SBERT). The 2-layer MLP (~842K params) just learns to fuse and project these representations.

**Implementation**: `models/speech_encoder.py`, `data/preprocessing.py` (feature extraction), `data/speech_dataset.py`

---

## How Cross-Cohort Fusion Works

This is the core methodological innovation. The MRI cohort (NACC) and speech cohort (DementiaBank) share **zero subjects**. our system aligns them through two complementary mechanisms:

### Mechanism 1: Shared CORAL Ordinal Head

Both modalities feed into the **exact same** ordinal classification head with **shared thresholds**:

```
MRI embedding (h_m ∈ ℝ²⁵⁶)  ──→  Shared CORAL Head  ──→  CDR {0, 0.5, 1, 2+}
                                     (same weights,
                                      same thresholds)
Speech embedding (z_s ∈ ℝ²⁵⁶) ──→  Shared CORAL Head  ──→  CDR {0, 0.5, 1, 2+}
```

**Why this aligns representations**: The CORAL head learns 3 severity thresholds (b1, b2, b3) on a shared scale. For both modalities to correctly classify CDR using the same thresholds, the embeddings must encode severity on the same numerical scale. A "CDR 0.5 MRI patient" and a "CDR 0.5 speech patient" must produce similar severity scores — otherwise one modality's scores wouldn't work with the shared thresholds.

### Mechanism 2: Class-Conditioned MMD

The shared head provides implicit alignment through label structure, but doesn't guarantee that same-class embeddings from different modalities are actually close in latent space. Class-conditioned Maximum Mean Discrepancy (MMD) provides explicit distribution matching:

```
For each CDR class k ∈ {0, 0.5, 1, 2+}:

  MRI embeddings where CDR = k:    {h_m : y_m = k}
  Speech embeddings where CDR = k:  {z_s : y_s = k}

  MMD²(MRI_class_k, Speech_class_k) → minimize this!
```

The MMD uses a Gaussian kernel with bandwidth set by the median heuristic (median pairwise distance per batch). The total alignment loss sums MMD across all severity classes.

**Result**: After training, MRI and speech embeddings for same-severity patients occupy overlapping regions of the 256-D latent space — even though no patient appears in both datasets.

### Training Protocol: Alternating Mini-Batches

```
Each training step:
  1. Sample MRI batch (B_m=4 longitudinal sequences from NACC)
     → Forward through MRI encoder + GRU → h_m
     → Compute ordinal loss + survival loss + amyloid loss

  2. Sample speech batch (B_s=16 samples from DementiaBank)
     → Forward through speech encoder → z_s
     → Compute ordinal loss (shared head)

  3. Compute class-conditioned MMD between h_m and z_s

  4. Sum all losses → backprop through everything in one optimizer step
```

Both batches are in the same computational graph, so alignment gradients flow into both encoders simultaneously.

**Implementation**: `models/alignment.py` (ClassConditionedMMD), `models/full_model.py` (AlzheimerMultiTaskModel), `training/trainer.py` (Phase2Trainer)

---

## How Classification Results Are Produced

### Task 1: Ordinal CDR Staging (CORAL)

**Input**: Any 256-D embedding (from MRI or speech)
**Method**: CORAL ordinal regression with 3 learnable thresholds

```
Embedding → Linear(256,128) → ReLU → Dropout → Linear(128,1) → severity score s

Cumulative logits:  ℓ_k = s - b_k   for k = 1, 2, 3
Predicted class:    ŷ = Σ 𝟙(ℓ_k > 0)
Class probabilities: derived from sigmoid(ℓ_k) differences
```

| Predicted Class | CDR Score | Clinical Meaning |
|----------------|-----------|------------------|
| 0 | 0.0 | Cognitively normal |
| 1 | 0.5 | Very mild dementia / MCI |
| 2 | 1.0 | Mild dementia |
| 3 | 2.0+ | Moderate to severe dementia |

**Why CORAL over cross-entropy?** Standard cross-entropy treats all misclassifications equally. CORAL preserves ordinal structure — predicting CDR 0 for a CDR 2 patient is penalized more heavily than predicting CDR 1.

**Implementation**: `models/task_heads.py` (OrdinalHead), `models/losses.py` (coral_ordinal_loss)

### Task 2: MCI-to-AD Conversion (Discrete-Time Survival)

**Input**: 256-D longitudinal MRI embedding (MCI patients only, CDR=0.5 at baseline)
**Method**: Discrete-time survival model over 36-month window

```
Embedding → Linear(256,128) → ReLU → Dropout → Linear(128,6) → Sigmoid

Output: 6 interval-specific hazard probabilities (one per 6-month window)
  ĥ₁ = P(convert in months 0-6)
  ĥ₂ = P(convert in months 6-12 | survived past 6)
  ĥ₃ = P(convert in months 12-18 | survived past 12)
  ...
  ĥ₆ = P(convert in months 30-36 | survived past 30)

Cumulative conversion probability at month t:
  P(convert by t) = 1 - ∏(1 - ĥ_l) for l up to interval containing t
```

**Key advantage**: Handles right-censored patients (lost to follow-up before 36 months) natively — no imputation needed. Produces time-varying risk curves, not just a binary "will/won't convert" label.

**Implementation**: `models/task_heads.py` (SurvivalHead), `models/losses.py` (discrete_survival_loss)

### Task 3: Amyloid Positivity (Optional Auxiliary)

**Input**: 256-D MRI embedding (NACC patients with PET/CSF data only)
**Method**: Binary classification

```
Embedding → Linear(256,64) → ReLU → Dropout → Linear(64,1) → Sigmoid
Output: P(amyloid-positive) ∈ (0, 1)
```

Amyloid-positive if Centiloid > 20 (PET) or CSF Abeta42 < 192 pg/mL.

**Implementation**: `models/task_heads.py` (AmyloidHead), `models/losses.py` (amyloid_bce_loss)

### Multi-Task Loss Balancing

All task losses are combined using **homoscedastic uncertainty weighting** (Kendall et al., 2018):

```
L_total = (1/2σ₁²)·L_ord + log(σ₁)
        + (1/2σ₂²)·L_surv + log(σ₂)
        + (1/2σ₃²)·L_amy + log(σ₃)
        + λ·L_align
```

The uncertainty parameters σ₁, σ₂, σ₃ are **learned** during training. Tasks with noisier gradients automatically receive lower weight, preventing destabilization. The alignment loss weight λ is a fixed hyperparameter (default: 0.1).

**Implementation**: `models/losses.py` (MultiTaskLoss)

---

## Two-Phase Training

### Phase 1: MRI Ordinal Pretraining

| Setting | Value |
|---------|-------|
| Objective | CORAL ordinal loss only |
| Data | NACC cross-sectional (one scan per subject) |
| Epochs | 30 |
| Batch size | 8 (effective 16 with gradient accumulation) |
| LR | 3e-4 with cosine annealing |
| Early stopping | Patience 7 on validation QWK |
| Time | ~9 hours on Colab T4 |

**Purpose**: Establish a strong MRI encoder before adding multi-task and cross-cohort objectives, which can destabilize early training.

### Phase 2: Full Multi-Task + Longitudinal + Alignment

| Setting | Value |
|---------|-------|
| Objective | All losses (ordinal + survival + amyloid + alignment) |
| Data | NACC longitudinal + DementiaBank alternating batches |
| Epochs | 40 |
| LR | 1e-4 (backbone), 5e-4 (heads) |
| Alignment λ | Linearly warmed from 0 over 5 epochs |
| Time | ~50 hours across 3-4 Colab Pro sessions |

**Checkpoint-resume**: Model state, optimizer state, and training history are saved to Google Drive after every epoch, enabling seamless resume across Colab sessions.

**Implementation**: `training/trainer.py` (Phase1Trainer, Phase2Trainer), `training/callbacks.py` (CheckpointManager, EarlyStopping)

---

## Evaluation Metrics

| Task | Primary Metric | Additional Metrics |
|------|---------------|--------------------|
| **CDR staging** | QWK (Quadratic Weighted Kappa) | MAE, off-by-1 rate, ECE, macro-F1, confusion matrix |
| **MCI→AD conversion** | Harrell's C-index | Time-dependent AUC at 12/24/36 mo, Brier score, IBS |
| **Amyloid positivity** | AUROC | AUPRC, sensitivity @ 90% specificity |
| **Alignment quality** | Class-conditioned MMD | t-SNE visualization, cross-modal nearest neighbor |

All primary metrics reported with bootstrap 95% confidence intervals (1000 resamples).

**Implementation**: `evaluation/metrics.py`, `evaluation/visualization.py`

---

## Project Structure

```
neurovox/
├── config.py                        # Central configuration (all hyperparameters)
├── requirements.txt
├── README.md                        # This file
├── ARCHITECTURE_PROPOSAL.md                  # Full technical design document
│
├── models/
│   ├── mri_encoder.py               # 3D ResNet-18 (1-channel adapted, 33.5M params)
│   ├── temporal_module.py           # Sinusoidal time encoding + GRU
│   ├── speech_encoder.py            # 2-layer MLP (1382→512→256)
│   ├── task_heads.py                # OrdinalHead, SurvivalHead, AmyloidHead
│   ├── losses.py                    # CORAL, survival, amyloid, MultiTaskLoss
│   ├── alignment.py                 # ClassConditionedMMD
│   ├── full_model.py                # AlzheimerMultiTaskModel (unified)
│   └── __init__.py
│
├── data/
│   ├── label_construction.py        # CDR mapping, conversion labels, stratified splits
│   ├── nacc_dataset.py              # NACCMRIDataset + NACCLongitudinalDataset
│   ├── speech_dataset.py            # SpeechEmbeddingDataset + SyntheticSpeechDataset
│   ├── preprocessing.py             # MRI preprocessing, augmentation, feature extraction
│   └── __init__.py
│
├── training/
│   ├── trainer.py                   # Phase1Trainer, Phase2Trainer
│   ├── callbacks.py                 # EarlyStopping, CheckpointManager
│   └── __init__.py
│
├── evaluation/
│   ├── metrics.py                   # QWK, C-index, td-AUC, Brier, ECE
│   ├── visualization.py             # Confusion matrices, survival curves, t-SNE
│   └── __init__.py
│
└── notebooks/
    ├── 01_preprocess_nacc_mri.ipynb  # Build manifest CSV from NACC data
    ├── 02_extract_speech_features.ipynb  # Extract speech biomarkers
    └── 03_train_and_evaluate.ipynb   # Train and evaluate (includes synthetic sanity check)
```

---

## Quick Start

```bash
pip install -r requirements.txt
```

Run notebooks in order:

1. **`01_preprocess_nacc_mri.ipynb`** — Build manifest CSV from NACC UDS + NIfTI files, construct CDR labels and conversion targets
2. **`02_extract_speech_features.ipynb`** — Extract acoustic (wav2vec2 + handcrafted) and linguistic (SBERT + handcrafted) features from DementiaBank
3. **`03_train_and_evaluate.ipynb`** — Train Phase 1 → Phase 2, evaluate all tasks, generate visualizations (includes synthetic data sanity check)

### Compute Requirements

- Google Colab T4 GPU (16 GB VRAM)
- ~2.4 GB VRAM at batch 4, 128³ resolution with AMP
- Mixed precision (fp16) throughout
- Phase 1: ~9 hours (1 Colab session)
- Phase 2: ~50 hours (3-4 Colab Pro sessions with checkpoint-resume)

---

## Key References

1. Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank consistent ordinal regression for neural networks. *Pattern Recognition Letters*, 140, 325-331.
2. Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses. *CVPR 2018*.
3. Gensheimer, M. F., & Narasimhan, B. (2019). A scalable discrete-time survival model for neural networks. *PeerJ*, 7, e6257.
4. Baevski, A., et al. (2020). wav2vec 2.0: Self-supervised learning of speech representations. *NeurIPS 2020*.
5. Qiu, S., et al. (2020). Interpretable deep learning for Alzheimer's classification. *Brain*, 143(6), 1920-1933.
6. Hara, K., et al. (2018). Can spatiotemporal 3D CNNs retrace the history of 2D CNNs? *CVPR 2018*.
7. Gretton, A., et al. (2012). A kernel two-sample test. *JMLR*, 13, 723-773.
8. Ganin, Y., et al. (2016). Domain-adversarial training of neural networks. *JMLR*, 17(1).
9. Fraser, K. C., et al. (2016). Linguistic features identify Alzheimer's disease in narrative speech. *Journal of Alzheimer's Disease*, 49(2).
10. Beekly, D. L., et al. (2007). The NACC database: the Uniform Data Set. *Alzheimer Disease & Associated Disorders*, 21(3).
11. Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP 2019*.
12. de Jong, J., et al. (2019). Deep learning for clustering of multivariate clinical patient trajectories. *GigaScience*, 8(11).

See [ARCHITECTURE_PROPOSAL.md](ARCHITECTURE_PROPOSAL.md) for the complete reference list and detailed technical design.

---

## Citation

```bibtex
@software{ali2025longitudinal,
  title={Longitudinal Multi-Task Alzheimer's Staging via Cross-Cohort Neuroimaging and Speech Biomarkers},
  year={2025},
  url={https://github.com/your-repo/neurovox}
}
```

---

## License

This is a research system. It should not be used for clinical decision-making without prospective validation and regulatory review.
