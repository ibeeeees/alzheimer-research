"""
features.py — Extract audio features (spectrograms) from raw audio.

Supports three feature types:
1. MFCC — Mel-Frequency Cepstral Coefficients (compact, widely used in speech)
2. Mel spectrogram — frequency representation using Mel scale
3. log-Mel spectrogram — Mel spectrogram in log scale (better dynamic range)

All features are returned as 2D numpy arrays (frequency x time) that look
like "images" — perfect for feeding into a CNN.
"""

import numpy as np
import librosa


def extract_features(audio, sample_rate, feature_type="logmel",
                     n_mels=128, n_fft=1024, hop_length=512, n_mfcc=40):
    """
    Convert raw audio waveform into a 2D spectrogram feature.

    Args:
        audio: numpy array of audio samples (float32, mono)
        sample_rate: sample rate in Hz
        feature_type: "mfcc", "mel", or "logmel"
        n_mels: number of Mel bands (height of the spectrogram)
        n_fft: FFT window size (controls frequency resolution)
        hop_length: step size between FFT windows (controls time resolution)
        n_mfcc: number of MFCC coefficients (only used when feature_type="mfcc")

    Returns:
        2D numpy array of shape (n_features, time_steps)
    """
    if feature_type == "mfcc":
        # MFCC: extracts the "shape" of the spectrum
        # Result shape: (n_mfcc, time_steps), e.g., (40, 157)
        features = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    elif feature_type == "mel":
        # Mel spectrogram: shows energy at each Mel-frequency band over time
        # Result shape: (n_mels, time_steps), e.g., (128, 157)
        features = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )

    elif feature_type == "logmel":
        # Log-Mel spectrogram: same as Mel but in log scale
        # Log scale compresses the huge range of energy values,
        # making quiet sounds more visible (like adjusting brightness)
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        # Add small epsilon to avoid log(0)
        features = librosa.power_to_db(mel_spec, ref=np.max)

    else:
        raise ValueError(
            f"Unknown feature_type '{feature_type}'. "
            f"Choose from: 'mfcc', 'mel', 'logmel'"
        )

    return features


def normalize_features(features):
    """
    Normalize features to zero mean and unit variance.

    Why normalize?
    - Different audio recordings have different volumes
    - Normalization puts all samples on the same scale
    - Helps the neural network learn faster and more stably

    Args:
        features: 2D numpy array (frequency x time)

    Returns:
        Normalized 2D numpy array
    """
    mean = features.mean()
    std = features.std()

    # Avoid division by zero (silent audio has std=0)
    if std < 1e-6:
        return features - mean

    return (features - mean) / std


def pad_or_truncate_features(features, target_length):
    """
    Make all spectrograms the same width (time dimension).

    Neural networks need fixed-size inputs, but different audio clips
    produce different-width spectrograms. This function pads short ones
    with zeros or truncates long ones.

    Args:
        features: 2D numpy array (frequency x time)
        target_length: desired number of time steps

    Returns:
        2D numpy array of shape (frequency, target_length)
    """
    current_length = features.shape[1]

    if current_length >= target_length:
        # Truncate: take only the first target_length columns
        return features[:, :target_length]
    else:
        # Pad: add zeros on the right
        pad_width = target_length - current_length
        return np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
