# Longitudinal Multi-Task Alzheimer's Staging via Cross-Cohort Neuroimaging and Speech Biomarkers on the SCAN Dataset

**A Research Architecture Proposal**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction & Motivation](#2-introduction--motivation)
3. [Dataset Construction](#3-dataset-construction)
4. [System Architecture](#4-system-architecture)
5. [Cross-Cohort Alignment Strategy](#5-cross-cohort-alignment-strategy)
6. [Loss Functions](#6-loss-functions)
7. [Training Strategy & Compute Budget](#7-training-strategy--compute-budget)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Ablation Study Plan](#9-ablation-study-plan)
10. [Related Work & Differentiation](#10-related-work--differentiation)
11. [Limitations & Ethical Considerations](#11-limitations--ethical-considerations)
12. [References](#12-references)

---

## 1. Executive Summary

We propose a **multi-task, longitudinal, cross-cohort predictive modeling system** for Alzheimer's disease (AD) staging and progression, built on two complementary data sources:

| Component | Source | Scale | Modality |
|-----------|--------|-------|----------|
| Structural imaging | SCAN (NACC) | ~29,000 3D T1w MRI | Volumetric brain morphometry |
| Speech biomarkers | DementiaBank (Pitt Corpus) | ~550 recordings | Acoustic + linguistic features |

The system jointly optimizes three clinically relevant objectives:

1. **Ordinal CDR severity staging** (4-class ordinal regression via CORAL)
2. **MCI-to-AD conversion prediction** within a 24–36 month window (discrete-time survival)
3. **Cross-modal representation alignment** (label-conditioned domain bridging)

Because subjects do not overlap between NACC and DementiaBank, we frame the multimodal component as **cross-cohort biomarker modeling** — an approach that is both clinically realistic (not all patients receive both MRI and cognitive-linguistic assessment) and methodologically novel in the Alzheimer's deep learning literature.

The system is designed for training on a single Google Colab T4 GPU (16 GB VRAM) using mixed-precision arithmetic, gradient accumulation, and checkpoint-resume workflows.

---

## 2. Introduction & Motivation

### 2.1 The SCAN Dataset Advantage

The Standardized Centralized Alzheimer's & Related Dementias Neuroimaging (SCAN) initiative represents a paradigm shift in AD neuroimaging data infrastructure. Unlike ADNI and OASIS, which have dominated the literature for over a decade, SCAN provides:

- **Centralized acquisition protocols**: All participating ADRCs follow identical MRI and PET acquisition sequences, eliminating the site-harmonization problem that plagues multi-site studies.
- **Unified quality control**: Images are de-identified and defaced at the ADIR laboratory (Mayo Clinic), with centralized QC dashboards for each ADRC. This guarantees a level of data cleanliness that post-hoc harmonization (ComBat, etc.) on ADNI cannot achieve.
- **Direct UDS linkage**: Every imaging session is linked to NACC's Uniform Data Set, providing CDR scores, neuropsychological test batteries, diagnosis codes, and longitudinal visit histories.
- **Scale**: At ~29,000 T1w MRI scans with longitudinal coverage, SCAN offers roughly an order of magnitude more imaging data than typical ADNI-based studies (~2,000–3,000 subjects).

### 2.2 Why Cross-Cohort Multimodal Modeling

The dominant paradigm in multimodal AD modeling assumes paired data: every subject has both MRI and the secondary modality. This assumption is unrealistic in clinical practice and restricts training to the intersection of available modalities, often discarding 50–80% of the data.

We propose an alternative: **train modality-specific encoders on their respective cohorts, align them in a shared latent space via ordinal label structure, and deploy them independently or jointly at inference**. This is:

- **Clinically realistic**: A neurologist can use the MRI model alone; a speech pathologist can use the speech model alone.
- **Data-efficient**: Each encoder leverages its full dataset rather than being constrained to paired samples.
- **Methodologically novel**: Cross-cohort multimodal alignment for AD has not been explored in the literature.

### 2.3 Why Multi-Task Learning

Single-task diagnostic models (e.g., "classify AD vs. CN from MRI") conflate multiple clinical questions into one binary output. In contrast, our multi-task formulation:

- **Preserves ordinal structure**: CDR is not nominal; a CDR 0.5 misclassified as CDR 2.0 is clinically worse than misclassifying it as CDR 1.0.
- **Models progression, not just state**: MCI-to-AD conversion within a time window is a fundamentally different question from current severity — it requires the model to learn prognostic features, not just diagnostic features.
- **Provides implicit regularization**: Multi-task learning acts as an inductive bias, forcing the shared encoder to learn features that generalize across related clinical tasks.

---

## 3. Dataset Construction

### 3.1 NACC MRI: Organizing 29,000 Longitudinal Scans

#### 3.1.1 Data Structure

NACC UDS data is provided in **long format**: one row per participant per visit. Each row includes:

| Variable | Description | Use |
|----------|-------------|-----|
| `NACCID` | Unique subject identifier | Subject-level grouping |
| `VISITNUM` / `VISITDATE` | Visit sequence number and date | Temporal ordering |
| `CDRGLOB` | Global CDR score (0, 0.5, 1, 2, 3) | Primary ordinal label |
| `NACCUDSD` | Clinical diagnosis (1=NL, 2=impaired-not-MCI, 3=MCI, 4=Dementia) | Conversion definition |
| `NACCALZD` | Etiological diagnosis of AD (0=no, 1=yes) | AD-specific conversion |
| `NACCAGE` | Age at visit | Covariate / stratification |

Each MRI scan is archived by `NACCID` and scan date, allowing direct linkage to the UDS visit closest in time.

#### 3.1.2 Baseline vs. Follow-up Definition

For each subject $i$, define:

- **Baseline visit** $t_0^{(i)}$: The earliest visit with both a valid MRI scan and a CDR assessment.
- **Follow-up visits** $\{t_1^{(i)}, t_2^{(i)}, \ldots, t_{T_i}^{(i)}\}$: All subsequent visits with valid MRI + CDR.
- **Observation window** $\Delta t_{ij} = t_j^{(i)} - t_0^{(i)}$: Time elapsed from baseline, in months.

Subjects with only a single visit contribute to **cross-sectional tasks** (CDR staging) but not to **longitudinal tasks** (conversion prediction).

#### 3.1.3 Ordinal CDR Label Mapping

Map the 5-level CDR scale to 4 ordinal classes:

| CDR Score | Class Index | Label | Expected Proportion |
|-----------|-------------|-------|-------------------|
| 0.0 | 0 | NonDemented | ~40–50% |
| 0.5 | 1 | VeryMildDementia | ~25–30% |
| 1.0 | 2 | MildDementia | ~12–18% |
| 2.0+ | 3 | ModerateToSevere | ~5–10% |

CDR 3.0 is merged into class 3 due to low prevalence. This mapping produces $K = 4$ ordered classes with $K - 1 = 3$ ordinal thresholds.

#### 3.1.4 Conversion Label Construction

**Definition.** A subject converts from MCI to AD if:

1. At baseline visit $t_0$, the subject has $\text{CDR} = 0.5$ (MCI-equivalent), AND
2. At some follow-up visit $t_j$ within the conversion window $W$ (default: 36 months), the subject has $\text{CDR} \geq 1.0$ AND $\text{NACCALZD} = 1$ (AD etiology confirmed).

This produces three categories per MCI subject:

| Category | Definition | Label |
|----------|-----------|-------|
| **Converter** | CDR transitions from 0.5 to ≥1.0 within $W$ months | $\delta_i = 1$, event time $T_i = t_{\text{convert}} - t_0$ |
| **Stable MCI** | CDR remains 0.5 with follow-up $\geq W$ months | $\delta_i = 0$, $T_i = W$ |
| **Right-censored** | CDR remains 0.5 but last follow-up $< W$ months | $\delta_i = 0$, $T_i = t_{\text{last}} - t_0$ |

This is a standard right-censored survival setup. The discrete-time survival model (Section 6.2) handles censoring natively.

**Conversion window discretization.** Divide the 36-month window into $J = 6$ intervals of 6 months each:

$$\mathcal{T} = \{[0, 6), [6, 12), [12, 18), [18, 24), [24, 30), [30, 36]\} \text{ months}$$

For each MCI subject $i$, the discrete event indicator at interval $j$ is:

$$d_{ij} = \begin{cases} 1 & \text{if subject } i \text{ converts during interval } j \\ 0 & \text{otherwise} \end{cases}$$

The at-risk indicator is:

$$r_{ij} = \begin{cases} 1 & \text{if subject } i \text{ is still observed and unconverted at start of interval } j \\ 0 & \text{otherwise (already converted or censored before interval } j) \end{cases}$$

#### 3.1.5 Expected Scale Breakdown

Given ~29,000 scans across N subjects (estimated 8,000–12,000 unique subjects with 2–4 visits on average):

| Subset | Estimated N (subjects) | Scans |
|--------|----------------------|-------|
| All subjects | ~10,000 | ~29,000 |
| CDR 0 at baseline (cognitively normal) | ~4,500 | ~13,000 |
| CDR 0.5 at baseline (MCI) | ~3,000 | ~9,000 |
| CDR ≥ 1.0 at baseline (dementia) | ~2,500 | ~7,000 |
| MCI → AD converters (within 36 mo) | ~600–900 | — |
| MCI stable (≥36 mo follow-up) | ~1,200–1,800 | — |
| MCI censored (<36 mo follow-up) | ~500–800 | — |

The conversion prediction task operates on the ~3,000 MCI-at-baseline subjects.

### 3.2 DementiaBank Speech Data

#### 3.2.1 Pitt Corpus Structure

The DementiaBank Pitt Corpus contains audio recordings of the Cookie Theft picture description task:

| Group | Subjects | Recordings | Sessions/Subject |
|-------|----------|-----------|-----------------|
| Dementia (probable AD) | ~170 | ~310 | 1–3 |
| Control | ~100 | ~240 | 1–3 |
| **Total** | **~270** | **~550** | — |

Diagnosis severity is available via MMSE scores and clinical staging. We map to a simplified ordinal scheme:

| MMSE Range | Mapped Class | Clinical Interpretation |
|-----------|-------------|----------------------|
| 27–30 | 0 (NonDemented) | Cognitively normal |
| 21–26 | 1 (VeryMild/Mild) | Mild cognitive impairment |
| 11–20 | 2 (Moderate) | Moderate dementia |
| 0–10 | 3 (Severe) | Severe dementia |

**Note**: The DementiaBank severity distribution will differ from NACC. This distribution mismatch is explicitly modeled via the alignment strategy (Section 5).

#### 3.2.2 Feature Extraction Pipeline

Two complementary feature streams are extracted from each recording:

**Stream A — Acoustic Biomarkers (from raw audio)**:

| Feature Group | Features | Dimensionality |
|--------------|----------|---------------|
| Spectral | MFCCs (13) + Δ + ΔΔ | 39 |
| Prosodic | F0 mean/std/range, energy mean/std | 5 |
| Voice quality | Jitter (local/RAP), shimmer (local/apq3), HNR | 5 |
| Temporal | Speech rate, articulation rate, pause rate, mean pause duration, phonation ratio | 5 |
| **Subtotal** | | **54** |

Extracted via OpenSMILE (eGeMAPS feature set) or Parselmouth (Praat-based), aggregated per recording as summary statistics (mean, std, skew, kurtosis) → **216-dimensional handcrafted vector**.

Additionally, extract **learned acoustic embeddings** from a pretrained model:

- **wav2vec 2.0 BASE** (95M params): Extract hidden states from the final transformer layer, average-pool over time → **768-dimensional embedding**.
- Fine-tuning: Freeze the feature encoder (CNN layers), fine-tune the transformer layers on the DementiaBank classification objective.

**Stream B — Linguistic Biomarkers (from transcripts)**:

Transcripts are obtained via Whisper large-v3 ASR or from existing DementiaBank CHAT transcripts.

| Feature Group | Features | Dimensionality |
|--------------|----------|---------------|
| Lexical diversity | TTR, MATTR (window=25), Brunet's W, Honoré's R | 4 |
| Syntactic complexity | Mean dependency distance, clause density, Yngve depth | 3 |
| Semantic coherence | Cosine similarity between adjacent sentence embeddings (mean, min) | 2 |
| Information content | Information units (Cookie Theft specific), idea density (propositional density) | 2 |
| Fluency | Filler rate ("uh", "um"), repetition rate, revision rate | 3 |
| **Subtotal** | | **14** |

Additionally, extract **learned linguistic embeddings**:

- **Sentence-BERT** (all-MiniLM-L6-v2): Encode full transcript → **384-dimensional embedding**.
- Or: Mean-pool sentence-level embeddings across all sentences in the transcript.

**Combined speech feature vector**: Concatenate [handcrafted acoustic (216) + wav2vec2 (768) + handcrafted linguistic (14) + sentence-BERT (384)] → **1,382-dimensional** raw vector, projected to 256-D via a learned linear layer.

### 3.3 Subject-Level Splitting & Leakage Prevention

**Critical constraint**: All splits must be at the **subject level**. If subject $i$ appears in the training set, *all* visits of subject $i$ must be in the training set. This prevents temporal data leakage where the model memorizes subject-specific brain morphology from one visit and trivially predicts another.

#### 3.3.1 NACC MRI Splits

Stratified by baseline CDR class and conversion status:

| Split | Subjects | Scans (approx.) | Purpose |
|-------|----------|-----------------|---------|
| Train | 70% (~7,000) | ~20,300 | Model training |
| Validation | 15% (~1,500) | ~4,350 | Hyperparameter tuning, early stopping |
| Test | 15% (~1,500) | ~4,350 | Final evaluation (touched once) |

Stratification variables: `(baseline_CDR, has_followup, is_converter)` to ensure proportional representation in each split.

#### 3.3.2 DementiaBank Splits

Given the small size (~270 subjects), use a more conservative split:

| Split | Subjects | Recordings (approx.) |
|-------|----------|---------------------|
| Train | 60% (~162) | ~330 |
| Validation | 20% (~54) | ~110 |
| Test | 20% (~54) | ~110 |

For robust estimation, report results from **5-fold cross-validation** on DementiaBank in addition to the held-out test set.

#### 3.3.3 Leakage Prevention Checklist

- [ ] No subject appears in multiple splits.
- [ ] Longitudinal visits from the same subject are in the same split.
- [ ] Conversion labels are derived only from visits within the subject's designated split.
- [ ] Hyperparameter search uses validation set only; test set is evaluated once at the end.
- [ ] DementiaBank subjects with multiple sessions are grouped by subject.
- [ ] No information from test subjects influences preprocessing (e.g., normalization statistics are computed on training data only).

---

## 4. System Architecture

### 4.1 System Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CROSS-COHORT MULTI-TASK ARCHITECTURE                    ║
║                                                                            ║
║  ┌─── NACC/SCAN Cohort ──────────────────────────────────────────────┐     ║
║  │                                                                    │     ║
║  │  Visit t₁           Visit t₂           ...        Visit tₙ       │     ║
║  │  ┌──────────┐       ┌──────────┐                  ┌──────────┐    │     ║
║  │  │ 3D T1w   │       │ 3D T1w   │                  │ 3D T1w   │    │     ║
║  │  │ MRI Vol  │       │ MRI Vol  │                  │ MRI Vol  │    │     ║
║  │  │ 128³     │       │ 128³     │                  │ 128³     │    │     ║
║  │  └────┬─────┘       └────┬─────┘                  └────┬─────┘    │     ║
║  │       │                  │                              │          │     ║
║  │       ▼                  ▼                              ▼          │     ║
║  │  ┌─────────┐        ┌─────────┐                   ┌─────────┐    │     ║
║  │  │ 3D CNN  │        │ 3D CNN  │  (shared weights) │ 3D CNN  │    │     ║
║  │  │ Encoder │        │ Encoder │                   │ Encoder │    │     ║
║  │  │ fθ_m    │        │ fθ_m    │                   │ fθ_m    │    │     ║
║  │  └────┬────┘        └────┬────┘                   └────┬────┘    │     ║
║  │       │                  │                              │          │     ║
║  │       ▼                  ▼                              ▼          │     ║
║  │     z_m^(1)            z_m^(2)         ...          z_m^(n)       │     ║
║  │     ∈ ℝ^256            ∈ ℝ^256                      ∈ ℝ^256      │     ║
║  │       │                  │                              │          │     ║
║  │       └──────────┬───────┘──────── ... ─────────────────┘          │     ║
║  │                  │                                                 │     ║
║  │                  ▼                                                 │     ║
║  │        ┌──────────────────┐                                       │     ║
║  │        │  Temporal GRU    │                                       │     ║
║  │        │  + Positional    │                                       │     ║
║  │        │  Time Encoding   │                                       │     ║
║  │        └────────┬─────────┘                                       │     ║
║  │                 │                                                  │     ║
║  │                 ▼                                                  │     ║
║  │            h_m ∈ ℝ^256  (longitudinal MRI representation)         │     ║
║  │                 │                                                  │     ║
║  └─────────────────┼──────────────────────────────────────────────────┘     ║
║                    │                                                        ║
║              ┌─────┴─────┐                                                  ║
║              │  SHARED   │◄──── Alignment Loss (Section 5)                  ║
║              │  LATENT   │                                                  ║
║              │  SPACE    │                                                  ║
║              └─────┬─────┘                                                  ║
║                    │                                                        ║
║  ┌─── DementiaBank Cohort ───────────────────────────────────────────┐     ║
║  │                                                                    │     ║
║  │  ┌─────────────────┐    ┌──────────────────┐                      │     ║
║  │  │ Raw Audio       │    │ Transcript        │                     │     ║
║  │  │ (Cookie Theft)  │    │ (ASR / CHAT)      │                     │     ║
║  │  └───────┬─────────┘    └────────┬──────────┘                     │     ║
║  │          │                       │                                 │     ║
║  │          ▼                       ▼                                 │     ║
║  │  ┌──────────────┐      ┌──────────────────┐                       │     ║
║  │  │ wav2vec 2.0  │      │ Sentence-BERT    │                       │     ║
║  │  │ + Handcraft  │      │ + Handcraft      │                       │     ║
║  │  │ Acoustic     │      │ Linguistic       │                       │     ║
║  │  └──────┬───────┘      └────────┬─────────┘                       │     ║
║  │         │                       │                                  │     ║
║  │         └───────────┬───────────┘                                  │     ║
║  │                     │                                              │     ║
║  │                     ▼                                              │     ║
║  │           ┌──────────────────┐                                    │     ║
║  │           │ Speech Encoder   │                                    │     ║
║  │           │ gθ_s             │                                    │     ║
║  │           │ MLP: 1382→512→   │                                    │     ║
║  │           │       256        │                                    │     ║
║  │           └────────┬─────────┘                                    │     ║
║  │                    │                                               │     ║
║  │               z_s ∈ ℝ^256  (speech representation)                │     ║
║  │                    │                                               │     ║
║  └────────────────────┼──────────────────────────────────────────────┘     ║
║                       │                                                     ║
║         ┌─────────────┴──────────────┐                                     ║
║         │    SHARED MULTI-TASK HEADS │                                      ║
║         │                            │                                      ║
║         │  ┌──────────┐ ┌─────────┐ ┌────────────┐                        ║
║         │  │ CORAL    │ │ Binary  │ │ Discrete   │                         ║
║         │  │ Ordinal  │ │ Amyloid*│ │ Survival   │                         ║
║         │  │ Head     │ │ Head    │ │ Head       │                          ║
║         │  └────┬─────┘ └────┬────┘ └─────┬──────┘                         ║
║         │       │            │            │                                 ║
║         └───────┼────────────┼────────────┼─────────────────────────────   ║
║                 ▼            ▼            ▼                                  ║
║            CDR Stage    Aβ+ Status   P(convert                              ║
║            ŷ ∈ {0,1,2,3} ŷ ∈ {0,1}   by month t)                          ║
║                                                                             ║
║  * Amyloid head active only for NACC subjects with PET/CSF biomarkers      ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 4.2 MRI Encoder: 3D CNN

#### 4.2.1 Input Preprocessing

Each T1-weighted MRI volume undergoes the following standardized pipeline (SCAN's centralized QC handles steps 1–3; we perform 4–6):

1. **SCAN QC**: De-identification, defacing, visual inspection (centrally provided).
2. **Brain extraction**: FSL BET or SynthStrip (if not already skull-stripped by SCAN).
3. **Registration**: Affine registration to MNI152 1mm template via ANTs or FSL FLIRT.
4. **Resampling**: Trilinear interpolation to isotropic $128 \times 128 \times 128$ voxels (1.5mm effective resolution). This provides sufficient anatomical detail while being memory-efficient.
5. **Intensity normalization**: Per-volume z-score normalization (zero mean, unit variance) after masking to brain voxels only.
6. **Clipping**: Clip to $[-3, +3]$ standard deviations to remove outlier intensities.

**Alternative resolution**: If memory is a constraint, downsample to $96 \times 96 \times 96$ (2mm effective resolution). See Section 7 for memory analysis.

#### 4.2.2 Architecture: 3D ResNet-18

We use a 3D ResNet-18 backbone adapted from the video recognition domain (`torchvision.models.video.r3d_18`), modified for single-channel medical imaging:

```
Input: x ∈ ℝ^{B×1×128×128×128}

Stem:
  Conv3d(1, 64, kernel=7, stride=2, padding=3)     → B×64×64×64×64
  BatchNorm3d(64) → ReLU
  MaxPool3d(kernel=3, stride=2, padding=1)          → B×64×32×32×32

Layer1: 2× BasicBlock3d(64, 64, stride=1)           → B×64×32×32×32
Layer2: 2× BasicBlock3d(64, 128, stride=2)          → B×128×16×16×16
Layer3: 2× BasicBlock3d(128, 256, stride=2)         → B×256×8×8×8
Layer4: 2× BasicBlock3d(256, 512, stride=2)         → B×512×4×4×4

AdaptiveAvgPool3d(output_size=1)                     → B×512×1×1×1
Flatten                                               → B×512
Linear(512, 256) → LayerNorm(256) → ReLU → Dropout(0.1)  → B×256

Output: z_m ∈ ℝ^{B×256}
```

**Parameter count**: ~33.4M (backbone) + 131K (projection) ≈ **33.5M parameters**

**Weight initialization**: Load `r3d_18` pretrained on Kinetics-400. Adapt the first convolutional layer from 3-channel to 1-channel by averaging the RGB kernels:

$$W_{\text{gray}} = \frac{1}{3}(W_R + W_G + W_B) \in \mathbb{R}^{64 \times 1 \times 7 \times 7 \times 7}$$

#### 4.2.3 Data Augmentation (Training Only)

| Augmentation | Parameters | Probability |
|-------------|-----------|-------------|
| Random affine rotation | ±10° per axis | 0.5 |
| Random scaling | [0.95, 1.05] | 0.5 |
| Random translation | ±5 voxels per axis | 0.5 |
| Random intensity shift | ±0.1 | 0.3 |
| Random intensity scale | [0.9, 1.1] | 0.3 |
| Random Gaussian noise | σ = 0.02 | 0.2 |
| Random left-right flip | — | 0.5 |

Augmentations are applied on-the-fly using `torchio` or `MONAI` transforms. Left-right flipping is valid because the brain is approximately bilaterally symmetric in structural morphometry (excluding lateralized pathology studies).

### 4.3 Longitudinal Temporal Module

For subjects with multiple visits, per-visit MRI embeddings $\{z_m^{(1)}, z_m^{(2)}, \ldots, z_m^{(T_i)}\}$ are aggregated via a **time-aware GRU**:

#### 4.3.1 Temporal Encoding

Time intervals between visits are encoded as continuous positional features. For visit $t_j$ occurring $\Delta t_j$ months after baseline:

$$\text{TE}(\Delta t_j) = [\sin(\omega_1 \Delta t_j), \cos(\omega_1 \Delta t_j), \ldots, \sin(\omega_{d/2} \Delta t_j), \cos(\omega_{d/2} \Delta t_j)]$$

where $\omega_k = 1 / 10000^{2k/d}$ and $d = 64$. This allows the model to encode irregular visit intervals (a key advantage over fixed-step RNNs).

Each input to the GRU is the concatenation of the visit embedding and its temporal encoding:

$$\tilde{z}_m^{(j)} = [z_m^{(j)} \; \| \; \text{TE}(\Delta t_j)] \in \mathbb{R}^{320}$$

#### 4.3.2 GRU Architecture

```
Input projection: Linear(320, 256)                   → ℝ^{256} per step
GRU(input_size=256, hidden_size=256, num_layers=1, bidirectional=False)
Output: h_m = GRU hidden state at final timestep     → ℝ^{256}
```

For subjects with a single visit ($T_i = 1$), the GRU reduces to a single forward step, effectively passing the embedding through with a learned transformation. No special-casing is needed.

**Parameter count**: ~590K

**Why GRU over Transformer**: With typical sequence lengths of 2–5 visits, the inductive bias of a recurrent model (sequential processing, built-in temporal ordering) is more appropriate than a Transformer, which requires at minimum 2 attention heads to be meaningful and benefits from longer sequences (>10). The GRU also uses significantly less memory.

### 4.4 Speech Encoder

Given the small size of DementiaBank (~550 samples), the speech encoder must be **parameter-efficient** and rely heavily on **pretrained representations**:

```
Input: [acoustic_handcrafted(216), wav2vec2_embed(768),
        linguistic_handcrafted(14), sentbert_embed(384)]
       → concat → ℝ^{1382}

Speech Encoder (MLP):
  Linear(1382, 512) → LayerNorm(512) → GELU → Dropout(0.3)
  Linear(512, 256) → LayerNorm(256) → GELU → Dropout(0.2)

Output: z_s ∈ ℝ^{256}
```

**Parameter count**: ~842K

The heavy lifting is done by the frozen/fine-tuned pretrained models (wav2vec 2.0 and Sentence-BERT). The MLP learns to fuse and project these representations into the shared latent space.

**Fine-tuning strategy for wav2vec 2.0**:
- Freeze the CNN feature extractor (layers 0–6).
- Fine-tune the transformer layers (7–12) with a discriminative learning rate: $\text{lr}_{\text{transformer}} = 0.1 \times \text{lr}_{\text{head}}$.
- Add a mean-pooling layer over the temporal dimension.
- Train for 20 epochs with early stopping on validation accuracy.

### 4.5 Multi-Task Prediction Heads

All prediction heads operate on the **shared 256-dimensional representation** ($h_m$ for MRI subjects, $z_s$ for speech subjects).

#### 4.5.1 Ordinal CDR Head (CORAL)

```
Input: h ∈ ℝ^{256}   (h_m or z_s depending on modality)

Linear(256, 128) → ReLU → Dropout(0.2)
Linear(128, 1)           → scalar severity score s ∈ ℝ

OrdinalHead:
  Learnable thresholds: b = [b₁, b₂, b₃]  (nn.Parameter)
  Cumulative logits: ℓₖ = s - bₖ  for k = 1, 2, 3
  Predicted class: ŷ = Σₖ 𝟙(ℓₖ > 0)
```

**Active for**: All subjects in both cohorts.

#### 4.5.2 Conversion Survival Head

```
Input: h_m ∈ ℝ^{256}  (MRI longitudinal representation, MCI subjects only)

Linear(256, 128) → ReLU → Dropout(0.2)
Linear(128, J)           → raw hazard logits ∈ ℝ^J  (J=6 intervals)
Sigmoid                   → ĥⱼ ∈ (0, 1)  per interval
```

**Active for**: NACC MCI-at-baseline subjects only ($\text{CDR}_{\text{baseline}} = 0.5$).

The cumulative conversion probability at the end of interval $j$ is:

$$\hat{P}(\text{convert by } \tau_j \mid x) = 1 - \prod_{l=1}^{j}(1 - \hat{h}_l(x))$$

#### 4.5.3 Amyloid Positivity Head (Optional Auxiliary Task)

If amyloid PET SUVR or CSF Aβ42 data is available for a subset of NACC subjects:

```
Input: h_m ∈ ℝ^{256}

Linear(256, 64) → ReLU → Dropout(0.2)
Linear(64, 1)           → logit ∈ ℝ
Sigmoid                  → P(Aβ+) ∈ (0, 1)
```

**Active for**: NACC subjects with available amyloid biomarker data.

**Label definition**: Amyloid-positive if Centiloid > 20 (PET) or CSF Aβ42 < 192 pg/mL (NACC thresholds).

This head serves as both (a) a clinically useful prediction and (b) an auxiliary training signal that encourages the encoder to learn amyloid-sensitive features.

### 4.6 Complete Parameter Budget

| Component | Parameters | Memory (fp16) |
|-----------|-----------|---------------|
| 3D ResNet-18 backbone | 33.4M | 63.8 MB |
| MRI projection head | 131K | 0.3 MB |
| Temporal GRU | 590K | 1.1 MB |
| Speech encoder MLP | 842K | 1.6 MB |
| Ordinal CDR head | 33K | 0.1 MB |
| Conversion survival head | 33K | 0.1 MB |
| Amyloid head (optional) | 16K | <0.1 MB |
| Learnable task weights | 3 | <0.01 MB |
| **Total** | **~35.0M** | **~67 MB** |

This is comfortably within the T4's 16 GB VRAM budget. The dominant memory cost is activation storage during the forward pass, not parameter storage. See Section 7 for detailed memory analysis.

---

## 5. Cross-Cohort Alignment Strategy

### 5.1 The Alignment Problem

The MRI encoder $f_{\theta_m}$ and the speech encoder $g_{\theta_s}$ are trained on disjoint subject populations with potentially different severity distributions. Without explicit alignment, the learned representations $h_m$ and $z_s$ will occupy different regions of $\mathbb{R}^{256}$, even for subjects with the same clinical severity. The shared prediction heads cannot function correctly on unaligned representations.

### 5.2 Label-Conditioned Alignment via Shared Ordinal Head

**Core mechanism**: Both encoders feed into the **same** CORAL ordinal head with **shared thresholds**. This forces both representations to produce severity scores on the same scale, implicitly aligning the latent spaces.

Formally, for an MRI subject with embedding $h_m$ and a speech subject with embedding $z_s$, both producing severity scores $s_m = \phi(h_m)$ and $s_s = \phi(z_s)$ through the shared projection $\phi$, the ordinal loss enforces:

$$\text{If } y_m = y_s, \text{ then } s_m \approx s_s$$

because both must satisfy the same threshold constraints.

### 5.3 Explicit Distribution Alignment: Ordinal-Aware MMD

The shared head provides **class-conditional alignment** but does not guarantee that the **marginal distributions** $p(h_m)$ and $p(z_s)$ are similar. We add an explicit distribution matching term using Maximum Mean Discrepancy (MMD), conditioned on ordinal class:

$$\mathcal{L}_{\text{align}} = \sum_{k=0}^{K-1} \text{MMD}^2\left(\{h_m^{(i)} : y_m^{(i)} = k\}, \; \{z_s^{(j)} : y_s^{(j)} = k\}\right)$$

where the MMD with Gaussian kernel is:

$$\text{MMD}^2(P, Q) = \mathbb{E}_{x,x' \sim P}[\kappa(x, x')] - 2\mathbb{E}_{x \sim P, y \sim Q}[\kappa(x, y)] + \mathbb{E}_{y,y' \sim Q}[\kappa(y, y')]$$

$$\kappa(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$$

with kernel bandwidth $\sigma$ set via the median heuristic (median pairwise distance in each mini-batch).

In practice, this is computed on mini-batches by sampling equal numbers of MRI and speech embeddings per class. Classes with too few speech samples are oversampled.

### 5.4 Training Protocol for Alignment

Since the two datasets are independent, training proceeds in **alternating mini-batch** fashion:

```
For each training step:
  1. Sample a mini-batch of MRI data (B_m samples)
     → Forward through MRI encoder + temporal GRU → h_m
     → Compute L_ord(h_m, y_m) + L_surv(h_m, d_m) + L_amy(h_m, y_a)

  2. Sample a mini-batch of speech data (B_s samples)
     → Forward through speech encoder → z_s
     → Compute L_ord(z_s, y_s)

  3. Compute L_align between h_m and z_s (from current batches)

  4. Backprop total loss through all parameters
```

The MRI batch and speech batch are processed in the same optimizer step, allowing gradients from the alignment loss to flow into both encoders.

### 5.5 Justification for Cross-Cohort Modeling in a Publication

The paper framing should emphasize:

1. **Clinical realism**: "In real-world clinical workflows, neuroimaging and cognitive-linguistic assessments are often acquired on different patient populations. Our cross-cohort framework reflects this reality by not requiring paired multimodal data."

2. **Data efficiency**: "Rather than restricting training to the intersection of available modalities — which would exclude >95% of our imaging data — we train each encoder on its full cohort and align representations via shared ordinal structure."

3. **Complementary biomarkers**: "Structural brain atrophy (MRI) and speech degradation capture fundamentally different aspects of neurodegeneration. Their complementarity does not require subject-level pairing to be scientifically meaningful."

4. **Precedent**: Cross-domain alignment without paired samples is well-established in domain adaptation (Ganin et al., 2016), cross-lingual NLP (Conneau et al., 2018), and multi-omics integration (Wang et al., 2021). We adapt these principles to the neuroimaging–speech biomarker setting.

---

## 6. Loss Functions

### 6.1 CORAL Ordinal Loss

For $K = 4$ ordinal classes with $K - 1 = 3$ learnable thresholds $\mathbf{b} = [b_1, b_2, b_3]$, the CORAL loss for a single sample with severity score $s = \phi(h)$ and ordinal label $y \in \{0, 1, 2, 3\}$ is:

$$\mathcal{L}_{\text{ord}}(s, y) = -\frac{1}{K-1}\sum_{k=1}^{K-1}\left[y_k^{*}\log\sigma(\ell_k) + (1 - y_k^{*})\log(1 - \sigma(\ell_k))\right]$$

where:
- $\ell_k = s - b_k$ are the cumulative logits
- $y_k^{*} = \mathbb{1}[y \geq k]$ are the cumulative binary targets
- $\sigma(\cdot)$ is the sigmoid function

The class probability is derived as:

$$P(Y = k \mid x) = \begin{cases} 1 - \sigma(\ell_1) & \text{if } k = 0 \\ \sigma(\ell_k) - \sigma(\ell_{k+1}) & \text{if } 0 < k < K-1 \\ \sigma(\ell_{K-1}) & \text{if } k = K-1 \end{cases}$$

The ordinal prediction is:

$$\hat{y} = \sum_{k=1}^{K-1} \mathbb{1}[\ell_k > 0]$$

**Batch loss**: Average over all samples in the batch (both MRI and speech):

$$\mathcal{L}_{\text{ord}} = \frac{1}{N_m + N_s}\left(\sum_{i=1}^{N_m}\mathcal{L}_{\text{ord}}(s_m^{(i)}, y_m^{(i)}) + \sum_{j=1}^{N_s}\mathcal{L}_{\text{ord}}(s_s^{(j)}, y_s^{(j)})\right)$$

### 6.2 Discrete-Time Survival Loss (MCI-to-AD Conversion)

For the conversion prediction task, we adopt the discrete-time survival framework (Gensheimer & Narasimhan, 2019). The observation window $[0, W]$ (where $W = 36$ months) is divided into $J = 6$ intervals of 6 months each.

For each MCI subject $i$, the model predicts interval-specific hazards:

$$\hat{h}_j(x_i) = \sigma(g_j(h_m^{(i)})) \in (0, 1) \quad \text{for } j = 1, \ldots, J$$

where $g_j$ denotes the $j$-th output of the survival head.

The discrete survival function is:

$$\hat{S}(j \mid x_i) = \prod_{l=1}^{j}(1 - \hat{h}_l(x_i))$$

The likelihood for subject $i$ with event indicator $\delta_i$ and last observed interval $j_i$ is:

$$L_i = \left[\hat{h}_{j_i}(x_i)\right]^{\delta_i} \cdot \prod_{l=1}^{j_i - \delta_i}\left[1 - \hat{h}_l(x_i)\right]$$

This decomposes into a sum of binary cross-entropies over at-risk intervals:

$$\mathcal{L}_{\text{surv}} = -\frac{1}{N_{\text{MCI}}}\sum_{i=1}^{N_{\text{MCI}}}\sum_{l=1}^{j_i}\left[d_{il}\log\hat{h}_l(x_i) + (1 - d_{il})\log(1 - \hat{h}_l(x_i))\right]$$

where $d_{il} = 1$ iff $\delta_i = 1$ and $l = j_i$ (event occurred in interval $l$), and 0 otherwise.

**Key property**: Right-censored subjects naturally contribute only "no event" observations for their observed intervals. No imputation or weighting is required.

### 6.3 Amyloid Positivity Loss (Optional Auxiliary)

Standard binary cross-entropy for subjects with available amyloid labels:

$$\mathcal{L}_{\text{amy}} = -\frac{1}{N_{\text{amy}}}\sum_{i=1}^{N_{\text{amy}}}\left[y_a^{(i)}\log\hat{p}_a^{(i)} + (1 - y_a^{(i)})\log(1 - \hat{p}_a^{(i)})\right]$$

where $\hat{p}_a^{(i)} = \sigma(f_{\text{amy}}(h_m^{(i)}))$.

### 6.4 Cross-Modal Alignment Loss

As defined in Section 5.3:

$$\mathcal{L}_{\text{align}} = \sum_{k=0}^{K-1} \text{MMD}^2\left(\{h_m : y_m = k\}, \{z_s : y_s = k\}\right)$$

### 6.5 Multi-Task Weighting via Homoscedastic Uncertainty

Following Kendall, Gal & Cipolla (2018), we learn task-specific uncertainty parameters $\{s_t\}_{t=1}^{T}$ (where $s_t = \log \sigma_t^2$) that automatically balance the loss magnitudes:

$$\mathcal{L}_{\text{total}} = \frac{1}{2e^{s_1}}\mathcal{L}_{\text{ord}} + \frac{s_1}{2} + \frac{1}{2e^{s_2}}\mathcal{L}_{\text{surv}} + \frac{s_2}{2} + \frac{1}{2e^{s_3}}\mathcal{L}_{\text{amy}} + \frac{s_3}{2} + \lambda_{\text{align}}\mathcal{L}_{\text{align}}$$

where:
- $s_1, s_2, s_3$ are learnable scalars (initialized to 0, i.e., $\sigma_t^2 = 1$)
- $\lambda_{\text{align}}$ is a fixed hyperparameter (default: 0.1), as the alignment loss operates on a different scale

The $\frac{s_t}{2}$ regularization terms prevent the model from trivially minimizing the loss by increasing $\sigma_t^2 \to \infty$ (which would zero out the task loss). The learned $\sigma_t^2$ values are interpretable: higher uncertainty means the model assigns less weight to that task.

**Gradient analysis**: The effective weight on task $t$ is $w_t = \frac{1}{2\sigma_t^2}$. As training progresses, tasks with noisier gradients (higher intrinsic uncertainty) will have larger $\sigma_t^2$ and thus lower effective weight, automatically preventing noisy tasks from destabilizing training.

### 6.6 Task-Specific Loss Activation

Not all losses apply to all subjects:

| Loss | NACC CDR≥0 | NACC MCI (CDR=0.5) | NACC Aβ-labeled | DementiaBank |
|------|-----------|-------------------|----------------|-------------|
| $\mathcal{L}_{\text{ord}}$ | ✓ | ✓ | ✓ | ✓ |
| $\mathcal{L}_{\text{surv}}$ | — | ✓ | — | — |
| $\mathcal{L}_{\text{amy}}$ | — | — | ✓ | — |
| $\mathcal{L}_{\text{align}}$ | ✓ | ✓ | ✓ | ✓ |

Missing-task losses are masked to zero for subjects without applicable labels. The uncertainty-weighted formulation handles this gracefully: the effective batch size for each task varies, and the learned weights adapt accordingly.

---

## 7. Training Strategy & Compute Budget

### 7.1 GPU Memory Analysis (T4 — 16 GB VRAM)

#### 7.1.1 Forward Pass Memory (Single 3D MRI Volume, fp16)

| Layer | Output Shape | Elements | Memory (fp16) |
|-------|-------------|----------|---------------|
| Input | 1×128×128×128 | 2.10M | 4.0 MB |
| Stem conv | 64×64×64×64 | 16.78M | 32.0 MB |
| MaxPool | 64×32×32×32 | 2.10M | 4.0 MB |
| Layer1 | 64×32×32×32 | 2.10M | 4.0 MB |
| Layer2 | 128×16×16×16 | 0.52M | 1.0 MB |
| Layer3 | 256×8×8×8 | 0.13M | 0.3 MB |
| Layer4 | 512×4×4×4 | 0.03M | 0.1 MB |
| **Subtotal (activations)** | | **23.76M** | **45.4 MB** |

With residual block intermediate activations (roughly 2×): **~91 MB per volume**

For a batch of B=4 volumes: **~364 MB** forward activations.

With backward pass (gradients ≈ 2× forward for activations stored for backprop): **~728 MB**

#### 7.1.2 Total Memory Budget

| Component | Memory (fp16) |
|-----------|---------------|
| Model parameters | 67 MB |
| Optimizer states (AdamW: 2× params) | 134 MB |
| Gradient buffers | 67 MB |
| Forward activations (B=4) | 364 MB |
| Backward computation | 728 MB |
| PyTorch runtime overhead | ~500 MB |
| CUDA context | ~500 MB |
| **Total** | **~2.4 GB** |

**Conclusion**: B=4 at 128³ uses only ~2.4 GB. We have **ample headroom** on the T4. Realistic options:

| Resolution | Batch Size | Est. Memory | Feasible? |
|-----------|-----------|-------------|-----------|
| 96³ | 8 | ~2.0 GB | ✓ Comfortable |
| 128³ | 4 | ~2.4 GB | ✓ Comfortable |
| 128³ | 8 | ~4.2 GB | ✓ Comfortable |
| 128³ | 16 | ~7.8 GB | ✓ Feasible |
| 160³ | 4 | ~4.8 GB | ✓ Feasible |
| 160³ | 8 | ~9.0 GB | ✓ Tight |

**Recommendation**: Use **128³ resolution with batch size 8**, employing **gradient accumulation** (2 steps) for an effective batch size of 16. This strikes the best balance between spatial resolution, batch statistics, and memory headroom for the speech branch and alignment computations.

### 7.2 Mixed-Precision Training

Use PyTorch's native `torch.amp.autocast('cuda')` with `GradScaler`:

```python
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast('cuda', dtype=torch.float16):
    loss = forward_pass(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**All operations** in fp16 except:
- Loss computation (fp32 for numerical stability)
- BatchNorm running statistics (fp32, handled automatically by PyTorch)
- Softmax/sigmoid in the ordinal head (fp32 to prevent underflow)

### 7.3 Training Schedule

#### 7.3.1 Phase 1: MRI Encoder Pretraining (Ordinal CDR Only)

| Hyperparameter | Value |
|---------------|-------|
| Objective | $\mathcal{L}_{\text{ord}}$ only (single-task) |
| Data | NACC train split, cross-sectional (one scan per subject per epoch) |
| Epochs | 30 |
| Batch size | 8 (effective 16 with grad accumulation) |
| Optimizer | AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$) |
| Learning rate | $3 \times 10^{-4}$ with cosine annealing to $1 \times 10^{-6}$ |
| Weight decay | $10^{-4}$ |
| Warmup | 5% of total steps (linear) |
| Early stopping | Patience 7 epochs on val QWK |

**Estimated time**: ~7,000 subjects × 2.5 visits/subject (sampled) / 8 per batch ≈ 2,200 steps/epoch × 30 epochs ≈ 66K steps. At ~0.5s/step on T4 with AMP: **~9 hours**. Feasible in a single Colab Pro session.

**Purpose**: Establish a strong MRI representation before introducing multi-task and cross-cohort objectives, which can destabilize early training.

#### 7.3.2 Phase 2: Multi-Task + Longitudinal Training

| Hyperparameter | Value |
|---------------|-------|
| Objective | $\mathcal{L}_{\text{total}}$ (all tasks + alignment) |
| Data | NACC (longitudinal sequences) + DementiaBank (alternating batches) |
| Epochs | 40 |
| Batch size | MRI: 4 sequences × variable length; Speech: 16 samples |
| Learning rate | $1 \times 10^{-4}$ (backbone), $5 \times 10^{-4}$ (heads), cosine decay |
| Temporal GRU | Initialized randomly, full learning rate |
| Speech encoder | Full learning rate |
| Alignment $\lambda$ | 0.1, linearly warmed from 0 over first 5 epochs |

**Estimated time**: More complex batching, ~1.5s/step × ~3,000 steps/epoch × 40 epochs ≈ **~50 hours**. Requires 3–4 Colab Pro sessions with checkpointing.

#### 7.3.3 Phase 3: Threshold Optimization (Post-hoc)

After training, perform a grid search over CORAL thresholds on the validation set to maximize QWK, as in the current system. This is a CPU operation taking <1 minute.

### 7.4 Checkpointing and Resume Strategy

Save after every epoch:

```
checkpoint = {
    'epoch': epoch,
    'mri_encoder_state': mri_encoder.state_dict(),
    'temporal_gru_state': temporal_gru.state_dict(),
    'speech_encoder_state': speech_encoder.state_dict(),
    'ordinal_head_state': ordinal_head.state_dict(),
    'survival_head_state': survival_head.state_dict(),
    'amyloid_head_state': amyloid_head.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scaler_state': scaler.state_dict(),
    'task_weights': [s1, s2, s3],
    'best_val_qwk': best_qwk,
    'history': history,
}
```

Save to Google Drive for persistence across Colab sessions. Total checkpoint size: ~140 MB (fp16 parameters + fp32 optimizer states).

### 7.5 Hyperparameter Search Strategy

Given the computational budget, full grid search is infeasible. Use a **two-stage approach**:

**Stage 1: Rapid sweep on 20% of data (~6K scans, Phase 1 objective only)**

Search over:
| Hyperparameter | Values |
|---------------|--------|
| Learning rate | {1e-4, 3e-4, 1e-3} |
| Weight decay | {1e-5, 1e-4, 1e-3} |
| Dropout | {0.1, 0.2, 0.3} |
| Embed dim | {128, 256} |
| Volume resolution | {96³, 128³} |

Total: 3 × 3 × 3 × 2 × 2 = 108 configurations. Each trains for 15 epochs on 20% data (~30 min). Full sweep: ~54 hours (parallelizable across 4–5 Colab sessions).

**Alternatively**: Use Optuna with TPE sampler, budget of 30 trials.

**Stage 2: Train the top-3 configurations on the full dataset** and select by validation QWK.

---

## 8. Evaluation Framework

### 8.1 Task-Specific Metrics

#### 8.1.1 Ordinal CDR Staging

| Metric | Formula / Description | Rationale |
|--------|----------------------|-----------|
| **QWK** (Quadratic Weighted Kappa) | $\kappa_w = 1 - \frac{\sum_{i,j}w_{ij}O_{ij}}{\sum_{i,j}w_{ij}E_{ij}}$ where $w_{ij} = \frac{(i-j)^2}{(K-1)^2}$ | Primary metric. Penalizes distant misclassifications more than adjacent ones. Standard for ordinal scales. |
| **MAE** (Mean Absolute Error) | $\frac{1}{N}\sum_i \|y_i - \hat{y}_i\|$ | Interpretable ordinal error in CDR-stage units. |
| **Accuracy** | Standard classification accuracy | Secondary. Does not account for ordinal structure. |
| **Off-by-1 Rate** | Fraction of predictions within 1 class of ground truth | Clinical tolerance metric. |
| **ECE** (Expected Calibration Error) | $\sum_{b=1}^{B}\frac{n_b}{N}\|\text{acc}(b) - \text{conf}(b)\|$ | Calibration quality for clinical decision-making. |
| **Per-class F1** | Macro-averaged F1 across 4 classes | Handles class imbalance. |
| **Confusion Matrix** | $K \times K$ matrix | Visual inspection of error patterns. |

#### 8.1.2 MCI-to-AD Conversion (Survival)

| Metric | Formula / Description | Rationale |
|--------|----------------------|-----------|
| **C-index** (Harrell's concordance) | $C = \frac{\sum_{i,j}\mathbb{1}[\hat{T}_i < \hat{T}_j]\mathbb{1}[T_i < T_j]\delta_i}{\sum_{i,j}\mathbb{1}[T_i < T_j]\delta_i}$ | Primary metric for survival models. Measures ranking accuracy. |
| **Time-dependent AUC** | AUC at specific timepoints (12, 24, 36 months) | Clinical interpretability at fixed horizons. |
| **Brier Score** | $\frac{1}{N}\sum_i(\hat{P}(\text{convert by } t \mid x_i) - \mathbb{1}[T_i \leq t, \delta_i = 1])^2$ | Calibrated probability assessment. |
| **Integrated Brier Score** (IBS) | $\frac{1}{W}\int_0^W \text{BS}(t)\,dt$ | Overall calibration across the full time horizon. |
| **Kaplan-Meier Concordance** | Stratify subjects by predicted risk quartile, compare KM curves | Visual validation of risk stratification. |

#### 8.1.3 Amyloid Positivity (Binary)

| Metric | Description |
|--------|-------------|
| **AUROC** | Area under ROC curve (primary) |
| **AUPRC** | Area under precision-recall curve (if class imbalance) |
| **Sensitivity @ 90% Specificity** | Clinical screening threshold |
| **F1 Score** | Balanced precision-recall |

#### 8.1.4 Cross-Cohort Alignment Quality

| Metric | Description |
|--------|-------------|
| **MMD** | Post-training MMD between MRI and speech embeddings per class (should be near 0) |
| **t-SNE / UMAP Visualization** | Color by severity and modality; aligned representations should intermix |
| **Cross-modal Nearest Neighbor** | For each speech embedding, find k-nearest MRI neighbors; measure label agreement |

### 8.2 Statistical Rigor

- **Confidence intervals**: Bootstrap 95% CIs (1000 resamples) for all primary metrics.
- **Significance testing**: Paired DeLong test for AUC comparisons; permutation test for QWK comparisons.
- **Cross-validation**: 5-fold stratified CV for DementiaBank results (given small N). Single train/val/test split for NACC (given large N, with bootstrap CIs on the test set).

---

## 9. Ablation Study Plan

The ablation plan is designed to demonstrate the value of each architectural and methodological choice. All ablations use the same train/val/test splits.

### 9.1 Ablation Matrix

| ID | Experiment | What It Tests | Compared To |
|----|-----------|---------------|-------------|
| **A1** | MRI only (ordinal CDR) | MRI encoder baseline | Full system |
| **A2** | Speech only (ordinal CDR) | Speech encoder baseline | Full system |
| **A3** | MRI + Speech (with alignment) | Cross-cohort benefit | A1 and A2 alone |
| **A4** | No alignment loss ($\lambda_{\text{align}} = 0$) | Value of explicit MMD alignment | Full system |
| **A5** | Single-task ordinal only (no survival, no amyloid) | Value of multi-task learning | Full system |
| **A6** | Multi-task without uncertainty weighting (fixed equal weights) | Value of learned task balancing | Full system |
| **A7** | Cross-sectional only (no temporal GRU, latest visit only) | Value of longitudinal modeling | Full system |
| **A8** | CORAL ordinal vs. standard cross-entropy | Value of ordinal-aware loss | Full system (CORAL) |
| **A9** | Binary conversion (36-month) vs. discrete survival | Value of survival formulation | Full system (survival) |
| **A10** | Volume resolution: 96³ vs. 128³ vs. 160³ | Resolution-performance tradeoff | 128³ default |
| **A11** | wav2vec2 only vs. handcrafted only vs. both | Value of each speech feature stream | Full speech encoder |
| **A12** | Acoustic only vs. linguistic only vs. both | Acoustic vs. linguistic contribution | Full speech encoder |

### 9.2 Expected Outcomes and Hypotheses

| Ablation | Hypothesis |
|----------|-----------|
| A1 vs A3 | Cross-cohort alignment provides modest improvement on ordinal CDR by regularizing the shared latent space. |
| A5 vs Full | Multi-task learning improves ordinal CDR by 1–3% QWK through shared representation regularization. Survival head provides the strongest auxiliary signal. |
| A7 vs Full | Longitudinal modeling substantially improves conversion prediction (ΔC-index ≥ 0.05) but has minimal impact on cross-sectional CDR staging. |
| A8 | CORAL outperforms CE by reducing distant misclassifications (lower MAE, higher QWK at comparable accuracy). |
| A9 | Discrete survival outperforms binary conversion by properly handling censored subjects and providing calibrated time-varying risk. |
| A10 | 128³ provides the best accuracy-efficiency tradeoff. 160³ shows diminishing returns. 96³ shows meaningful degradation on fine-grained CDR distinctions (0 vs. 0.5). |
| A11–A12 | wav2vec2 embeddings provide the strongest individual acoustic signal. Linguistic features provide complementary information, particularly for distinguishing MCI from early dementia. Combined features outperform either alone. |

### 9.3 Data Efficiency Experiments

Train the MRI branch on {10%, 25%, 50%, 75%, 100%} of the NACC training set. Plot each metric vs. data fraction. This demonstrates:

- How much data is needed for the 3D CNN to converge.
- Whether multi-task learning improves sample efficiency (expect multi-task to show steeper learning curves at low data fractions).
- The advantage of SCAN's scale over typical ADNI-sized datasets.

### 9.4 Presentation

All ablation results should be presented as:

1. **Summary table** with mean ± bootstrap 95% CI for primary metrics.
2. **Bar charts** with error bars for visual comparison.
3. **Statistical significance** annotations (*, **, *** for p < 0.05, 0.01, 0.001).

---

## 10. Related Work & Differentiation

### 10.1 Landscape of Multimodal AD Deep Learning

| Study | Dataset | Modalities | Tasks | Longitudinal | Ordinal |
|-------|---------|-----------|-------|-------------|---------|
| Qiu et al. (2020) | NACC | MRI | AD diagnosis | ✗ | ✗ |
| Lu et al. (2018) | ADNI | MRI + PET | AD/MCI/NC classification | ✗ | ✗ |
| El-Sappagh et al. (2020) | ADNI | MRI + clinical | Multi-class classification | ✗ | ✗ |
| Li et al. (2021) | ADNI | MRI + PET + genetics | AD staging | Partial | ✗ |
| Mahim et al. (2024) | ADNI | MRI | Binary AD classification | ✗ | ✗ |
| DiaMond (2025) | ADNI | MRI + PET | Dementia diagnosis (ViT) | ✗ | ✗ |
| HAMMF (2024) | ADNI | MRI + clinical | Multi-class | ✗ | ✗ |
| **Ours** | **SCAN (NACC) + DementiaBank** | **MRI + Speech** | **Ordinal CDR + Survival + Biomarker** | **✓** | **✓** |

### 10.2 Key Differentiators

1. **First multi-task deep learning study on SCAN**: All prior deep learning work in this space uses ADNI, OASIS, or smaller institutional datasets. SCAN's centralized QC pipeline eliminates the site-harmonization confound that affects virtually all ADNI-based studies.

2. **Cross-cohort multimodal alignment**: Prior multimodal work assumes paired data from the same subjects. Our framework trains on disjoint populations and aligns via shared ordinal structure — a paradigm that is both novel and clinically practical.

3. **Joint ordinal + survival + biomarker modeling**: Most studies treat AD staging as a flat classification problem. We preserve the ordinal structure of CDR, model conversion as a censored time-to-event process, and optionally predict amyloid positivity — three clinically distinct questions in one unified framework.

4. **Longitudinal deep learning with irregular intervals**: We explicitly model the temporal dynamics of brain atrophy across visits with time-aware positional encoding, rather than treating each scan independently.

5. **Speech as a complementary biomarker**: While speech analysis for AD is an active area, its integration with neuroimaging in a cross-cohort framework is unexplored.

6. **Scale**: At ~29,000 MRI scans, this is among the largest 3D CNN studies for AD staging, enabled by SCAN's centralized data infrastructure.

---

## 11. Limitations & Ethical Considerations

### 11.1 Methodological Limitations

- **No paired multimodal data**: Cross-cohort alignment is a principled workaround, not a substitute for paired data. The alignment quality depends on the assumption that ordinal severity labels capture sufficient shared variance.
- **DementiaBank size**: ~550 recordings limit the speech encoder's capacity and generalization. Results on the speech branch should be interpreted with appropriate uncertainty bounds.
- **Conversion label noise**: CDR is a clinical rating with inter-rater variability (κ ≈ 0.8). Conversion labels constructed from CDR trajectories inherit this noise.
- **Selection bias**: NACC ADRCs are primarily academic memory clinics, not population-representative. SCAN's sample may overrepresent atypical presentations.
- **Censoring bias**: Subjects lost to follow-up may differ systematically from those retained (e.g., sicker subjects may drop out or die).

### 11.2 Ethical Considerations

- **Data governance**: All data access follows NACC and DementiaBank data use agreements. No attempt is made to re-identify subjects.
- **Algorithmic fairness**: Report performance disaggregated by age, sex, education, and race/ethnicity (available in UDS). If performance disparities are found, discuss implications.
- **Clinical deployment caveat**: This is a research system. It should not be used for clinical decision-making without prospective validation and regulatory review.

---

## 12. References

### Foundational Methods

1. Cao, W., Mirjalili, V., & Raschka, S. (2020). [Rank consistent ordinal regression for neural networks with application to age estimation.](https://doi.org/10.1016/j.patrec.2020.09.024) *Pattern Recognition Letters*, 140, 325–331.

2. Kendall, A., Gal, Y., & Cipolla, R. (2018). [Multi-task learning using uncertainty to weigh losses for scene geometry and semantics.](https://arxiv.org/abs/1705.07115) *CVPR 2018*.

3. Gensheimer, M. F., & Narasimhan, B. (2019). [A scalable discrete-time survival model for neural networks.](https://doi.org/10.7717/peerj.6257) *PeerJ*, 7, e6257.

4. Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). [A kernel two-sample test.](https://jmlr.org/papers/v13/gretton12a.html) *JMLR*, 13, 723–773.

5. Hara, K., Kataoka, H., & Satoh, Y. (2018). [Can spatiotemporal 3D CNNs retrace the history of 2D CNNs and ImageNet?](https://arxiv.org/abs/1711.09577) *CVPR 2018*.

6. He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Deep residual learning for image recognition.](https://arxiv.org/abs/1512.03385) *CVPR 2016*.

### Cross-Modal Alignment & Domain Adaptation

7. Ganin, Y., et al. (2016). [Domain-adversarial training of neural networks.](https://arxiv.org/abs/1505.07818) *JMLR*, 17(1), 2096–2030.

8. Conneau, A., et al. (2018). [Word translation without parallel data.](https://arxiv.org/abs/1710.04087) *ICLR 2018*.

9. Long, M., Cao, Y., Wang, J., & Jordan, M. (2015). [Learning transferable features with deep adaptation networks.](https://arxiv.org/abs/1502.02791) *ICML 2015*.

10. Tzeng, E., Hoffman, J., Saenko, K., & Darrell, T. (2017). [Adversarial discriminative domain adaptation.](https://arxiv.org/abs/1702.05464) *CVPR 2017*.

### Alzheimer's Disease Deep Learning

11. Qiu, S., et al. (2020). [Development and validation of an interpretable deep learning framework for Alzheimer's disease classification.](https://doi.org/10.1093/brain/awaa137) *Brain*, 143(6), 1920–1933.

12. Lu, D., et al. (2018). [Multimodal and multiscale deep neural networks for the early diagnosis of Alzheimer's disease using structural MR and FDG-PET images.](https://doi.org/10.1038/s41598-018-22871-z) *Scientific Reports*, 8(1), 5697.

13. El-Sappagh, S., Abuhmed, T., Islam, S. M. R., & Kwak, K. S. (2020). [Multimodal multitask deep learning model for Alzheimer's disease progression detection based on time series data.](https://doi.org/10.1016/j.neucom.2020.05.087) *Neurocomputing*, 412, 197–215.

14. Li, H., et al. (2021). [A deep learning model for early prediction of Alzheimer's disease dementia based on hippocampal MRI.](https://doi.org/10.1002/alz.12474) *Alzheimer's & Dementia*, 17(S6).

15. Mahim, S. M., et al. (2024). [Unlocking the potential of AI for Alzheimer's disease detection.](https://doi.org/10.1109/ACCESS.2024.3351112) *IEEE Access*, 12.

16. Yee, E., et al. (2025). [DiaMond: Dementia diagnosis with multi-modal vision Transformers using MRI and PET.](https://doi.org/10.1109/TMI.2024.3476571) *IEEE TMI*, 44(1).

17. Li, W., et al. (2024). [HAMMF: Hierarchical attention-based multi-modal fusion for AD diagnosis.](https://doi.org/10.1016/j.compbiomed.2024.108348) *Computers in Biology and Medicine*, 176.

### Speech & Language Biomarkers for AD

18. Fraser, K. C., Meltzer, J. A., & Rudzicz, F. (2016). [Linguistic features identify Alzheimer's disease in narrative speech.](https://doi.org/10.3233/JAD-150520) *Journal of Alzheimer's Disease*, 49(2), 407–422.

19. Baevski, A., et al. (2020). [wav2vec 2.0: A framework for self-supervised learning of speech representations.](https://arxiv.org/abs/2006.11477) *NeurIPS 2020*.

20. Radford, A., et al. (2023). [Robust speech recognition via large-scale weak supervision.](https://arxiv.org/abs/2212.04356) *ICML 2023*.

21. Reimers, N. & Gurevych, I. (2019). [Sentence-BERT: Sentence embeddings using Siamese BERT-networks.](https://arxiv.org/abs/1908.10084) *EMNLP 2019*.

22. Luz, S., et al. (2020). [Alzheimer's Dementia recognition through spontaneous speech: The ADReSS challenge.](https://arxiv.org/abs/2004.06833) *Interspeech 2020*.

23. Pappagari, R., et al. (2021). [Automatic detection and assessment of Alzheimer's disease using speech and language technologies in low-resource scenarios.](https://doi.org/10.21437/Interspeech.2021-1850) *Interspeech 2021*.

24. Martinc, M., & Pollak, S. (2020). [Leveraging pre-trained language models for Alzheimer's disease detection.](https://aclanthology.org/2020.lrec-1.176/) *LREC 2020*.

### Longitudinal & Survival Modeling

25. de Jong, J., et al. (2019). [Deep learning for clustering of multivariate clinical patient trajectories with missing values.](https://doi.org/10.1093/gigascience/giz134) *GigaScience*, 8(11).

26. Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M. (2018). [DeepHit: A deep learning approach to survival analysis with competing risks.](https://arxiv.org/abs/1801.05512) *AAAI 2018*.

27. Marinescu, R. V., et al. (2020). [TADPOLE Challenge: Accurate Alzheimer's disease prediction through crowdsourced forecasting of future data.](https://doi.org/10.1371/journal.pcbi.1008462) *PLOS Computational Biology*, 16(11).

### Data Sources

28. Beekly, D. L., et al. (2007). [The National Alzheimer's Coordinating Center (NACC) database: the Uniform Data Set.](https://doi.org/10.1097/WAD.0b013e318142774e) *Alzheimer Disease & Associated Disorders*, 21(3), 249–258.

29. Becker, J. T., et al. (1994). [The natural history of Alzheimer's disease: Description of study cohort and accuracy of diagnosis.](https://doi.org/10.1001/archneur.1994.00540180063015) *Archives of Neurology*, 51(6), 585–594.

30. Besser, L. M., et al. (2023). [The Standardized Centralized Alzheimer's and Related Dementias Neuroimaging (SCAN) database.](https://doi.org/10.1002/alz.068684) *Alzheimer's & Dementia*, 19(S15).

---

*This document is a research architecture specification. It is intended as a technical design reference, not a clinical recommendation.*
