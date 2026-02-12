# Multi-Modal Ordinal Alzheimer's Research Pipeline

## Context

A multi-modal ordinal regression framework that integrates MRI and speech data to predict Alzheimer's severity as a continuous latent score mapped to ordered CDR stages. Designed for free-tier Colab/Drive storage by persisting only compressed embeddings.

**Data sources**:
- **Primary**: NACC clinical data (approved access) — may link to volumetric 3D MRI
- **MRI images**: Kaggle 2D MRI slice datasets (44K JPGs, 4 severity classes)
- **Audio**: TBD (pipeline ready for ADReSS, DementiaBank, or similar)
- **Handwriting**: Darwin UCI dataset (451 tabular features) — potential extension

Label mapping for Kaggle/Mendeley → ordinal CDR stages:
- NonDemented → CDR 0
- VeryMildDemented → CDR 0.5
- MildDemented → CDR 1
- ModerateDemented → CDR 2+

This gives 4 ordered classes and 3 learned thresholds (b1, b2, b3).

---

## System Architecture

```
NACC Clinical Labels ──────────────────────────┐
MRI Data ──→ MRI CNN ──→ MRI Embedding ────────┤
Audio Data ──→ Audio CNN ──→ Audio Embedding ──→ Fusion Model ──→ Severity Score s(x)
                                                                    │
                                                         Learned Thresholds b1 b2 b3
                                                                    │
                                                         Ordered CDR Stage Prediction
```

---

## Free Storage Strategy

```
Remote Raw Data (S3 / Kaggle / Public Host)
    → Temporary Colab Runtime
        → Feature Extraction Phase
            → Compressed Embeddings Saved to Google Drive
```

**Storage policy**:
- Do not store raw MRI volumes
- Do not store raw WAV files
- Save only `.npz` compressed embeddings using float16
- Delete intermediate files after embedding extraction

Expected storage: MRI embeddings < 100MB, Audio embeddings < 100MB.

---

## Project Structure

```
project/
├── requirements.txt
├── .gitignore
├── config.py                    # Central hyperparameter config
├── data_embeddings/             # Created at runtime (gitignored)
│   └── .gitkeep
├── models/
│   ├── __init__.py
│   ├── mri_cnn.py               # 3D ResNet-18 + 2D ResNet-18 adapter
│   ├── audio_cnn.py             # Log-mel spectrogram + 2D CNN
│   ├── fusion_model.py          # Multi-modal fusion network
│   └── ordinal_utils.py         # CORAL ordinal loss, thresholds, metrics
├── experiments/
│   ├── __init__.py
│   ├── train_unimodal.py        # Train MRI-only or Audio-only
│   ├── train_multimodal.py      # Train fused model on embeddings
│   └── evaluation.py            # Full evaluation suite
└── notebooks/
    ├── 01_extract_mri_embeddings.ipynb
    └── 02_extract_audio_embeddings.ipynb
```

---

## Models

### MRI Feature Extractor (`models/mri_cnn.py`)
- **`MRIResNet3D`**: 3D ResNet-18 for volumetric NACC MRI data
  - Input: (B, 1, D, H, W) grayscale volume
  - Output: (B, embed_dim) embedding vector
- **`MRIResNet2D`**: 2D ResNet-18 for Kaggle/Mendeley 2D slices
  - Pretrained ImageNet weights, adapted first conv (3→1 channel)
  - Input: (B, 1, H, W) grayscale slice
  - Output: (B, embed_dim) embedding vector

### Audio Feature Extractor (`models/audio_cnn.py`)
- **`AudioCNN`**: Log-mel spectrogram + 2D ResNet-18 backbone
  - Built-in spectrogram transform (128 mel bins)
  - Input: raw waveform or pre-computed spectrogram
  - Output: (B, embed_dim) embedding vector

### Fusion Model (`models/fusion_model.py`)
- **`FusionModel`**: Early concatenation fusion
- **`GatedFusionModel`**: Gated fusion with learnable modality weights
- **`AttentionFusionModel`**: Cross-modal attention fusion
- All output a single scalar severity score s(x)

### Ordinal Regression (`models/ordinal_utils.py`)
- **CORAL ordinal loss**: Binary cross-entropy across K-1 threshold comparisons
- **Learnable thresholds**: P(CDR ≥ k) = sigmoid(s(x) - b_k)
- **Threshold optimization**: Post-hoc grid search on validation data
- **Metrics**: Accuracy, QWK, MAE, Off-by-k, Expected Calibration Error

---

## Experimental Design

This framework supports extensive experimentation:

1. **Model comparison**: Unimodal MRI vs Unimodal Audio vs Multimodal Fusion
2. **Architecture ablation**: CNN depth (ResNet-18 vs ResNet-34), embedding dim (128 vs 256), fusion layer sizes
3. **Data fraction experiments**: Train on 25%, 50%, 75%, 100% — plot learning curves
4. **Hyperparameter sweeps**: Learning rate, weight decay, dropout rate, batch size
5. **Ordinal vs categorical**: Compare CORAL ordinal regression vs standard cross-entropy
6. **Fusion strategies**: Early fusion (concatenation) vs gated fusion vs attention-based fusion
7. **Evaluation rigor**: 5-fold cross-validation, statistical significance, confusion matrices, calibration plots

---

## Quick Start

```bash
pip install -r requirements.txt

# Train unimodal MRI model
python experiments/train_unimodal.py --modality mri --data_fraction 1.0

# Train unimodal Audio model
python experiments/train_unimodal.py --modality audio --data_fraction 1.0

# Train multimodal fusion model
python experiments/train_multimodal.py --fusion_type concat

# Run full evaluation suite
python experiments/evaluation.py --experiment all
```

---

## Resource Requirements

- Google Colab GPU runtime
- Google Drive (15GB free storage)
- Python 3.8+ with PyTorch, torchaudio, torchvision, scikit-learn
- Compressed `.npz` embedding storage
