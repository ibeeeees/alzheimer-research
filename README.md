# Alzheimer's Disease Detection from Speech Using Multiple Machine Learning Models

**Authors:** Ibe Mohammed Ali, Kubra Sag, Poorav Rawat

## Overview

This project implements and compares multiple machine learning models for **binary classification** of Alzheimer's disease from speech audio. Given a raw audio recording of a speaker describing the Cookie Theft picture, the model predicts whether the speaker is cognitively healthy or has been diagnosed with Alzheimer's disease.

Clinical diagnosis of Alzheimer's disease relies on costly and time-consuming cognitive assessments. Speech offers a non-invasive, low-cost signal: Alzheimer's-related cognitive decline produces measurable changes in acoustic patterns, including longer pauses, reduced vocabulary, and altered prosody. This project investigates whether models trained on spectrogram representations of speech can reliably distinguish Alzheimer's patients from healthy controls, and how performance depends on input representation, model complexity, and the presence of silence in the clips.

## Dataset

We use the [Movement Disorders Voice Dataset](https://www.kaggle.com/) from Kaggle, which contains raw audio recordings of individuals diagnosed with Alzheimer's disease and healthy controls.

**Directory structure expected:**

```
data_raw/
├── alzheimer/    # Audio files from Alzheimer's patients
└── healthy/      # Audio files from healthy controls
```

Alternatively, pass `--synthetic` to any script to generate synthetic data for testing the pipeline without the real dataset.

## Preprocessing Pipeline

1. **Silence trimming** — Optional Voice Activity Detection (VAD) via WebRTC to remove non-speech segments
2. **Resampling** — Audio resampled to a consistent sample rate (16 kHz)
3. **Amplitude normalization** — Peak normalization to [-1, 1]
4. **Feature extraction** — Mel spectrograms, log-Mel spectrograms, or MFCC-based spectrograms via librosa

## Models

| Model | Description |
|-------|-------------|
| **CNN** | Treats the Mel spectrogram as a 2D image; detects localized frequency-time patterns indicative of cognitive decline |
| **RNN** | Treats audio as a sequential signal over time via bidirectional GRU; captures temporal patterns such as pauses and rhythm changes associated with Alzheimer's |
| **SVM** | Trained on handcrafted acoustic features; a well-established approach for speech classification tasks |
| **Logistic Regression** | Baseline trained on flattened MFCC feature vectors to establish a performance floor |

## Methodology

Audio recordings are preprocessed using librosa: silence is optionally trimmed via WebRTC VAD, audio is resampled to a consistent rate, and amplitude is normalized. We extract Mel spectrograms and MFCC-based spectrograms as 2D image-like representations. We compare four models:

- **Logistic Regression** serves as our baseline, trained on flattened MFCC feature vectors to establish a performance floor.
- **SVM** is trained on handcrafted acoustic features — a well-established approach for speech classification tasks.
- **RNN** treats the audio as a sequential signal over time and is naturally suited to capturing temporal patterns in speech such as pauses and rhythm changes associated with Alzheimer's.
- **CNN** treats the Mel spectrogram as a 2D image and detects localized frequency-time patterns indicative of cognitive decline.

## Project Structure

```
alzheimer-research/
├── main.py                         # Entry point: train and evaluate a single model
├── config.py                       # Hyperparameters and global settings
├── requirements.txt                # Python dependencies
├── data/
│   ├── dataset.py                  # Dataset loading, splitting, DataLoader creation
│   ├── preprocessing.py            # Audio preprocessing (resampling, normalization, VAD)
│   └── features.py                 # Spectrogram and MFCC feature extraction
├── models/
│   ├── cnn.py                      # SpectrogramCNN architecture
│   ├── rnn.py                      # SpectrogramRNN architecture (bidirectional GRU)
│   └── sklearn_models.py           # SVM and Logistic Regression wrappers
├── training/
│   └── trainer.py                  # Training loop with early stopping and checkpointing
├── evaluation/
│   └── metrics.py                  # Accuracy, precision, recall, F1, confusion matrix
├── experiments/
│   ├── model_comparison.py         # Compare all four models head-to-head
│   ├── feature_comparison.py       # Compare MFCC vs Mel vs log-Mel representations
│   ├── architecture_search.py      # CNN hyperparameter tuning (filters, layers, dropout)
│   └── dataset_size.py             # Ablation study on training set size
├── results/                        # Saved CSV results and plots
└── checkpoints/                    # Saved model weights
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run with synthetic data (no dataset required)

```bash
# Train and evaluate a single model
python3 main.py --synthetic                           # CNN (default)
python3 main.py --model rnn --synthetic               # RNN
python3 main.py --model svm --synthetic               # SVM
python3 main.py --model logreg --synthetic            # Logistic Regression

# Run experiments
python3 experiments/model_comparison.py --synthetic   # Compare all 4 models
python3 experiments/feature_comparison.py --synthetic  # Compare feature types
python3 experiments/architecture_search.py --synthetic # CNN architecture search
python3 experiments/dataset_size.py --synthetic        # Dataset size ablation
```

### Run with real data

1. Download the dataset from Kaggle
2. Extract audio files into `data_raw/alzheimer/` and `data_raw/healthy/`
3. Run:

```bash
python3 main.py                                       # CNN on real data
python3 experiments/model_comparison.py               # Full comparison on real data
```

## Experiments

### Model Comparison
Compares CNN, RNN, SVM, and Logistic Regression on the same train/val/test split using accuracy, precision, recall, and F1-score.

### Feature Comparison
Evaluates how input representation (MFCC, Mel spectrogram, log-Mel spectrogram) affects CNN classification performance.

### Architecture Search
Tunes CNN hyperparameters including number of convolutional filters, number of layers, and dropout rate.

### Dataset Size Ablation
Measures how training set size impacts model performance to understand data efficiency.

## Results

### Dataset

131 real audio recordings from the Kaggle Movement Disorders Voice dataset:
- **76** healthy controls (label 0)
- **55** Alzheimer's patients (label 1)
- Split: 91 train / 20 validation / 20 test (stratified, 70/15/15)
- Feature representation: log-Mel spectrogram (128 Mel bands, 16 kHz sample rate)

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **SVM** | **75.0%** | **71.4%** | 62.5% | **66.7%** |
| **CNN** | 70.0% | 62.5% | 62.5% | 62.5% |
| **Logistic Regression** | 65.0% | 55.6% | **62.5%** | 58.8% |
| **RNN** | 65.0% | 57.1% | 50.0% | 53.3% |

### Key Findings

1. **SVM achieved the best overall performance** (75.0% accuracy, 66.7% F1), outperforming the deep learning models. On small datasets like ours (131 samples), classical models with well-chosen kernels often generalize better than neural networks, which tend to overfit.

2. **CNN ranked second** (70.0% accuracy, 62.5% F1). The 2D convolutional approach to spectrograms successfully captured some frequency-time patterns, but the model's 10.3M parameters are excessive for 91 training samples, likely causing overfitting despite regularization.

3. **Logistic Regression and RNN performed comparably** at 65.0% accuracy. Logistic Regression had higher recall (62.5% vs 50.0%), meaning it missed fewer Alzheimer's cases — clinically important since false negatives are more dangerous than false positives.

4. **RNN had the lowest recall** (50.0%), missing half of Alzheimer's patients. The bidirectional GRU's 511K parameters, while fewer than CNN, still overfit on the limited sequential data.

5. **Dataset size is the primary bottleneck.** With only 131 total samples (55 Alzheimer's), all models are severely data-limited. The strong performance of SVM suggests that classical approaches should be preferred when labeled clinical speech data is scarce.

### Visualizations

All plots are saved in the `results/` directory:

| File | Description |
|------|-------------|
| `model_comparison_metrics.png` | Grouped bar chart comparing accuracy, precision, recall, and F1 across all 4 models |
| `model_comparison_confusion.png` | Side-by-side confusion matrices for all 4 models |
| `model_comparison_training.png` | Training and validation loss/accuracy curves for CNN and RNN |
| `model_comparison_f1.png` | Horizontal bar chart ranking models by F1-score |
| `model_comparison.csv` | Raw numerical results for all models |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall fraction of correct predictions |
| **Precision** | Fraction of predicted positives that are truly positive |
| **Recall** | Fraction of actual positives that are correctly identified |
| **F1-score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Detailed breakdown of true/false positives and negatives |

## Limitations and Future Work

- **Small dataset**: 131 samples is insufficient for deep learning models to generalize well. Data augmentation (time stretching, pitch shifting, noise injection) could help.
- **No cross-validation**: The single 70/15/15 split means results are sensitive to which samples land in each set. K-fold cross-validation would provide more robust estimates.
- **Binary classification only**: The current pipeline classifies healthy vs. Alzheimer's. Extending to multi-class staging (e.g., mild cognitive impairment) would be more clinically useful.
- **No linguistic features**: Only acoustic features (spectrograms) are used. Adding linguistic features from transcripts (vocabulary diversity, pause patterns, disfluency rates) could improve performance.
- **Hyperparameter tuning was limited**: SVM and Logistic Regression used default hyperparameters. Grid search or Bayesian optimization could further improve results.

## Resources

- **Training:** Google Colab Pro with T4 GPU
- **Software:** Python, PyTorch, librosa, NumPy, scikit-learn, matplotlib, seaborn
- **Version control:** GitHub

## Workload Distribution

- **Ibe Mohammed Ali** — Data preprocessing, spectrogram extraction, CNN model implementation
- **Kubra Sag** — SVM model implementation, experimental studies, hyperparameter tuning
- **Poorav Rawat** — RNN and Logistic Regression implementation, evaluation, ablation studies, results analysis
- **All members** — Experimental design and final manuscript preparation

## References

1. Fraser, K. C., Meltzer, J. A., and Rudzicz, F. (2016). Linguistic features identify Alzheimer's disease in narrative speech. *Journal of Alzheimer's Disease*, 49, 407-422.
2. Becker, J. T., Boller, F., Lopez, O. L., Saxton, J., and McGonigle, K. L. (1994). The natural history of Alzheimer's disease: Description of study cohort and accuracy of diagnosis. *Archives of Neurology*, 51, 585-594.
3. Boller, F. and Becker, J. (2005). DementiaBank database guide. University of Pittsburgh.
