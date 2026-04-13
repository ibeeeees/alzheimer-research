"""
config.py — All hyperparameters and settings in one place.

Change values here to control every aspect of the pipeline:
data processing, model architecture, training, and experiments.
"""


# =============================================================================
# DATA SETTINGS
# =============================================================================

# Path to the Kaggle "Movement Disorders Voice" dataset.
# Expected structure after download:
#   DATA_DIR/
#   ├── alzheimer/   ← Alzheimer's audio files (.wav)
#   └── healthy/     ← Healthy control audio files (.wav)
#
# Download from: https://www.kaggle.com/datasets/cycoool29/movement-disorders-voice
# Only the "alzheimer" and "healthy" folders are used.
DATA_DIR = "./data_raw"

# Audio settings
SAMPLE_RATE = 16000          # Resample all audio to 16 kHz
MAX_AUDIO_LENGTH_SEC = 10.0  # Truncate/pad audio to this length
CHUNK_LENGTH_SEC = 5.0       # Length of each chunk when splitting long audio
CHUNK_OVERLAP_SEC = 1.0      # Overlap between consecutive chunks

# Spectrogram settings
N_MELS = 128                 # Number of Mel filter banks
N_FFT = 1024                 # FFT window size
HOP_LENGTH = 512             # Hop length for STFT
N_MFCC = 40                  # Number of MFCC coefficients

# Feature type: one of "mfcc", "mel", "logmel"
FEATURE_TYPE = "logmel"

# Voice Activity Detection (silence removal)
USE_VAD = False              # Set True to remove silence before feature extraction
VAD_AGGRESSIVENESS = 2       # 0 (least aggressive) to 3 (most aggressive)


# =============================================================================
# MODEL SETTINGS
# =============================================================================

# Which model to train: "cnn", "rnn", "svm", "logreg"
MODEL_TYPE = "cnn"

# CNN architecture: "small", "medium", or "large"
CNN_ARCHITECTURE = "medium"

# Architecture-specific settings (used by cnn.py)
CNN_CONFIGS = {
    "small": {
        "conv_channels": [16, 32],           # 2 conv layers
        "kernel_size": 3,
        "pool_size": 2,
        "fc_size": 64,
        "dropout": 0.3,
    },
    "medium": {
        "conv_channels": [32, 64, 128],      # 3 conv layers
        "kernel_size": 3,
        "pool_size": 2,
        "fc_size": 128,
        "dropout": 0.4,
    },
    "large": {
        "conv_channels": [32, 64, 128, 256], # 4 conv layers
        "kernel_size": 3,
        "pool_size": 2,
        "fc_size": 256,
        "dropout": 0.5,
    },
}

# RNN settings
RNN_HIDDEN_SIZE = 128        # Hidden state size for GRU layers
RNN_NUM_LAYERS = 2           # Number of stacked GRU layers
RNN_DROPOUT = 0.3            # Dropout between GRU layers
RNN_FC_SIZE = 64             # Fully connected layer size after GRU
RNN_BIDIRECTIONAL = True     # Use bidirectional GRU

# SVM settings
SVM_KERNEL = "rbf"           # Kernel type: "rbf", "linear", "poly"
SVM_C = 1.0                  # Regularization parameter
SVM_GAMMA = "scale"          # Kernel coefficient

# Logistic Regression settings
LOGREG_C = 1.0               # Inverse regularization strength
LOGREG_MAX_ITER = 1000       # Maximum iterations for solver


# =============================================================================
# TRAINING SETTINGS
# =============================================================================

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 15
WEIGHT_DECAY = 1e-4          # L2 regularization

# Train/validation/test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
SEED = 42

# Device: "auto" means use GPU if available, else CPU
DEVICE = "auto"


# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

# Where to save trained models and results
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"
