"""
dataset.py — Load the Kaggle Movement Disorders Voice dataset for training.

Uses only the Alzheimer's and Healthy subsets for binary classification:
- 0: healthy control
- 1: Alzheimer's disease

Expected folder structure after downloading from Kaggle:
    data_raw/
    ├── alzheimer/   ← .wav files (label = 1)
    └── healthy/     ← .wav files (label = 0)

Download: https://www.kaggle.com/datasets/cycoool29/movement-disorders-voice
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
from data.features import extract_features, normalize_features, pad_or_truncate_features
from data.preprocessing import remove_silence, chunk_audio


class SpeechDataset(Dataset):
    """
    PyTorch Dataset that holds pre-computed spectrogram features and labels.

    Each item is a tuple:
        (spectrogram_tensor, label)
    where:
        spectrogram_tensor: shape (1, n_freq, n_time) — like a grayscale image
        label: 0 (healthy) or 1 (Alzheimer's)
    """

    def __init__(self, features_list, labels_list):
        self.features = features_list
        self.labels = labels_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = self.features[idx]
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0)
        label_tensor = torch.FloatTensor([self.labels[idx]])
        return spec_tensor, label_tensor


# =============================================================================
# DATA LOADING
# =============================================================================


def _load_audio_files(data_dir):
    """
    Load audio files from the Kaggle dataset folders.

    Scans for "alzheimer" and "healthy" subdirectories under data_dir.

    Returns:
        list of (audio_array, label) tuples
    """
    import librosa

    # Map folder names to labels — only Alzheimer's and Healthy
    class_folders = {
        "healthy": 0,
        "alzheimer": 1,
    }

    samples = []

    for folder_name, label in class_folders.items():
        folder = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Expected folder '{folder}' not found.\n"
                f"Download the dataset from Kaggle and place it so the structure is:\n"
                f"  {data_dir}/\n"
                f"  ├── alzheimer/   (.wav files)\n"
                f"  └── healthy/     (.wav files)\n"
            )

        # Find all audio files
        audio_files = sorted(
            glob.glob(os.path.join(folder, "*.wav"))
            + glob.glob(os.path.join(folder, "*.mp3"))
            + glob.glob(os.path.join(folder, "*.flac"))
        )

        print(f"  Found {len(audio_files)} files in {folder_name}/")

        for fpath in audio_files:
            try:
                audio, sr = librosa.load(fpath, sr=config.SAMPLE_RATE, mono=True)
                samples.append((audio, label))
            except Exception as e:
                print(f"  WARNING: Could not load {fpath}: {e}")

    if len(samples) == 0:
        raise RuntimeError(
            f"No audio files found in {data_dir}.\n"
            f"Make sure the dataset is downloaded and the folder structure is correct."
        )

    return samples


def _generate_synthetic(n_samples=200):
    """
    Generate fake audio data for testing the pipeline without real data.

    - Class 0 (healthy): lower-frequency dominant signal
    - Class 1 (Alzheimer's): higher-frequency signal + noise + pauses
    """
    np.random.seed(config.SEED)
    samples = []
    duration = config.MAX_AUDIO_LENGTH_SEC
    n_samples_audio = int(duration * config.SAMPLE_RATE)

    for i in range(n_samples):
        label = i % 2
        t = np.linspace(0, duration, n_samples_audio, dtype=np.float32)

        if label == 0:
            freq = np.random.uniform(100, 300)
            audio = 0.5 * np.sin(2 * np.pi * freq * t)
            audio += 0.1 * np.random.randn(n_samples_audio).astype(np.float32)
        else:
            freq = np.random.uniform(400, 800)
            audio = 0.3 * np.sin(2 * np.pi * freq * t)
            audio += 0.3 * np.random.randn(n_samples_audio).astype(np.float32)
            pause_start = np.random.randint(0, n_samples_audio // 2)
            pause_len = np.random.randint(config.SAMPLE_RATE // 4, config.SAMPLE_RATE)
            audio[pause_start:pause_start + pause_len] *= 0.01

        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        samples.append((audio.astype(np.float32), label))

    return samples


# =============================================================================
# MAIN LOADING FUNCTION
# =============================================================================


def load_dataset(feature_type=None, use_vad=None, use_chunks=False,
                 data_dir=None, synthetic=False):
    """
    Load audio data and convert to spectrograms.

    Pipeline: raw audio -> (optional VAD) -> (optional chunking) -> spectrogram -> normalize -> pad/truncate

    Args:
        feature_type: "mfcc", "mel", or "logmel" (default: from config)
        use_vad: whether to remove silence (default: from config)
        use_chunks: whether to split audio into chunks
        data_dir: path to dataset folder (default: from config)
        synthetic: if True, use synthetic data for testing

    Returns:
        features_list: list of 2D numpy arrays
        labels_list: list of integer labels
    """
    if feature_type is None:
        feature_type = config.FEATURE_TYPE
    if use_vad is None:
        use_vad = config.USE_VAD
    if data_dir is None:
        data_dir = config.DATA_DIR

    print(f"Loading dataset...")
    print(f"  Feature type: {feature_type}")
    print(f"  VAD (silence removal): {use_vad}")
    print(f"  Chunking: {use_chunks}")

    # Step 1: Load raw audio
    if synthetic:
        print("  Using SYNTHETIC data (for testing only)")
        raw_samples = _generate_synthetic(n_samples=200)
    else:
        print(f"  Source: {data_dir}")
        raw_samples = _load_audio_files(data_dir)

    print(f"  Loaded {len(raw_samples)} raw audio samples")

    # Step 2: Process each audio sample
    features_list = []
    labels_list = []

    target_samples = int(config.MAX_AUDIO_LENGTH_SEC * config.SAMPLE_RATE)
    target_time_steps = target_samples // config.HOP_LENGTH + 1

    if use_chunks:
        chunk_samples = int(config.CHUNK_LENGTH_SEC * config.SAMPLE_RATE)
        target_time_steps = chunk_samples // config.HOP_LENGTH + 1

    for audio, label in tqdm(raw_samples, desc="Extracting features"):
        # Normalize audio amplitude
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        # Optional: remove silence
        if use_vad:
            audio = remove_silence(audio, config.SAMPLE_RATE, config.VAD_AGGRESSIVENESS)

        # Optional: split into chunks
        if use_chunks:
            audio_segments = chunk_audio(
                audio, config.SAMPLE_RATE,
                config.CHUNK_LENGTH_SEC, config.CHUNK_OVERLAP_SEC
            )
        else:
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            else:
                audio = np.pad(audio, (0, max(0, target_samples - len(audio))))
            audio_segments = [audio]

        for segment in audio_segments:
            spec = extract_features(
                segment,
                config.SAMPLE_RATE,
                feature_type=feature_type,
                n_mels=config.N_MELS,
                n_fft=config.N_FFT,
                hop_length=config.HOP_LENGTH,
                n_mfcc=config.N_MFCC,
            )

            spec = normalize_features(spec)
            spec = pad_or_truncate_features(spec, target_time_steps)

            features_list.append(spec)
            labels_list.append(label)

    n_healthy = labels_list.count(0)
    n_alz = labels_list.count(1)
    print(f"  Processed {len(features_list)} samples "
          f"(healthy: {n_healthy}, alzheimer: {n_alz})")

    return features_list, labels_list


def create_data_loaders(features_list, labels_list, data_fraction=1.0,
                        batch_size=None, seed=None):
    """
    Split data into train/val/test sets and create PyTorch DataLoaders.

    Args:
        features_list: list of 2D numpy arrays (spectrograms)
        labels_list: list of labels
        data_fraction: use only this fraction of the data (for ablation)
        batch_size: batch size (default: from config)
        seed: random seed (default: from config)

    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if seed is None:
        seed = config.SEED

    if data_fraction < 1.0:
        n_samples = max(1, int(len(features_list) * data_fraction))
        np.random.seed(seed)
        indices = np.random.choice(len(features_list), n_samples, replace=False)
        features_list = [features_list[i] for i in indices]
        labels_list = [labels_list[i] for i in indices]
        print(f"Using {data_fraction*100:.0f}% of data: {n_samples} samples")

    indices = list(range(len(features_list)))
    labels_array = np.array(labels_list)

    # First split: separate out the test set
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=config.TEST_RATIO,
        stratify=labels_array,
        random_state=seed,
    )

    # Second split: separate train and validation
    val_relative_size = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative_size,
        stratify=labels_array[train_val_idx],
        random_state=seed,
    )

    print(f"Split sizes — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    full_dataset = SpeechDataset(features_list, labels_list)

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
