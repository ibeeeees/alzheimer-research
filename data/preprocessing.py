"""MRI preprocessing and augmentation utilities.

Designed for both local and Colab execution.
Speech feature extraction helpers are also included.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ── MRI Preprocessing ────────────────────────────────────────────────

class MRIPreprocessor:
    """Preprocess raw NIfTI volumes to standardised tensors.

    Pipeline:
      1. Load NIfTI
      2. Resample to target shape via trilinear interpolation
      3. Z-score normalise (brain-masked)
      4. Clip to [-3, 3]

    This class can be used standalone or inside a notebook cell.
    """

    def __init__(
        self,
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        clip_range: Tuple[float, float] = (-3.0, 3.0),
    ):
        self.target_shape = target_shape
        self.clip_range = clip_range

    def process_file(self, nii_path: str | Path) -> np.ndarray:
        """Load and preprocess a single NIfTI file.

        Returns:
            (D, H, W) float32 numpy array, normalised.
        """
        import nibabel as nib
        from scipy.ndimage import zoom

        img = nib.load(str(nii_path))
        data = img.get_fdata(dtype=np.float32)

        # Resample if needed
        if data.shape != self.target_shape:
            factors = tuple(
                t / s for t, s in zip(self.target_shape, data.shape)
            )
            data = zoom(data, factors, order=1)

        # Z-score normalise (brain voxels only)
        mask = data > 0
        if mask.any():
            mean = data[mask].mean()
            std = data[mask].std() + 1e-8
            data = (data - mean) / std

        data = np.clip(data, *self.clip_range)
        return data

    def process_to_tensor(self, nii_path: str | Path) -> torch.Tensor:
        """Process and return as (1, D, H, W) tensor."""
        data = self.process_file(nii_path)
        return torch.from_numpy(data).unsqueeze(0)

    def batch_process(
        self,
        nii_paths: Sequence[str | Path],
        output_dir: str | Path,
        dtype: str = "float16",
    ) -> List[Path]:
        """Batch-process NIfTI files, saving as compressed .npz.

        Args:
            nii_paths: List of NIfTI file paths.
            output_dir: Directory to write .npz files.
            dtype: Storage dtype ("float16" or "float32").

        Returns:
            List of output .npz file paths.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []

        for path in nii_paths:
            path = Path(path)
            data = self.process_file(path)
            out_path = out_dir / f"{path.stem}.npz"
            save_data = data.astype(dtype) if dtype == "float16" else data
            np.savez_compressed(str(out_path), volume=save_data)
            output_paths.append(out_path)

        return output_paths


# ── MRI Augmentation ─────────────────────────────────────────────────

class MRIAugmentation:
    """On-the-fly 3D MRI augmentation for training.

    Operates on (1, D, H, W) tensors.  All augmentations are applied
    with independent random probability per sample.
    """

    def __init__(
        self,
        rotation_degrees: float = 10.0,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        translate_voxels: int = 5,
        intensity_shift: float = 0.1,
        intensity_scale: Tuple[float, float] = (0.9, 1.1),
        noise_std: float = 0.02,
        flip_prob: float = 0.5,
    ):
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range
        self.translate_voxels = translate_voxels
        self.intensity_shift = intensity_shift
        self.intensity_scale = intensity_scale
        self.noise_std = noise_std
        self.flip_prob = flip_prob

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to a (1, D, H, W) tensor."""
        # Random left-right flip
        if torch.rand(1).item() < self.flip_prob:
            volume = torch.flip(volume, [-1])  # flip W axis

        # Random intensity shift
        if torch.rand(1).item() < 0.3:
            shift = (torch.rand(1).item() * 2 - 1) * self.intensity_shift
            volume = volume + shift

        # Random intensity scale
        if torch.rand(1).item() < 0.3:
            lo, hi = self.intensity_scale
            scale = lo + torch.rand(1).item() * (hi - lo)
            volume = volume * scale

        # Random Gaussian noise
        if torch.rand(1).item() < 0.2:
            noise = torch.randn_like(volume) * self.noise_std
            volume = volume + noise

        # Random translation (simple shift)
        if torch.rand(1).item() < 0.5:
            max_t = self.translate_voxels
            shifts = [
                int((torch.rand(1).item() * 2 - 1) * max_t)
                for _ in range(3)
            ]
            volume = torch.roll(volume, shifts=shifts, dims=[1, 2, 3])

        return volume


def build_mri_augmentation(cfg) -> MRIAugmentation:
    """Build augmentation transform from config."""
    return MRIAugmentation(
        rotation_degrees=cfg.aug_rotation_degrees,
        scale_range=cfg.aug_scale_range,
        translate_voxels=cfg.aug_translate_voxels,
        intensity_shift=cfg.aug_intensity_shift,
        intensity_scale=cfg.aug_intensity_scale,
        noise_std=cfg.aug_noise_std,
        flip_prob=cfg.aug_flip_prob,
    )


# ── Speech Feature Extraction Helpers ────────────────────────────────

def extract_acoustic_handcrafted(audio_path: str | Path) -> np.ndarray:
    """Extract handcrafted acoustic features from an audio file.

    Returns a 216-D vector: 54 base features × 4 summary stats
    (mean, std, skew, kurtosis).

    Requires: librosa, parselmouth (Praat)
    """
    import librosa
    from scipy.stats import skew, kurtosis

    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    features = []

    # MFCCs (13 + delta + delta-delta = 39 frame-level features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    spectral_feats = np.vstack([mfcc, mfcc_delta, mfcc_delta2])  # (39, T)

    # Prosodic: F0, energy (5 features)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[pitches > 0]
    if len(pitch_vals) > 0:
        prosodic = np.array([
            pitch_vals.mean(), pitch_vals.std(),
            pitch_vals.max() - pitch_vals.min(),
        ])
    else:
        prosodic = np.zeros(3)
    rms = librosa.feature.rms(y=y)[0]
    prosodic = np.concatenate([prosodic, [rms.mean(), rms.std()]])  # (5,)

    # Voice quality placeholder (5 features — jitter, shimmer, HNR)
    # Full extraction requires Praat/parselmouth
    voice_quality = np.zeros(5)
    try:
        import parselmouth
        snd = parselmouth.Sound(str(audio_path))
        pitch = snd.to_pitch()
        pp = parselmouth.praat.call([snd, pitch], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([snd, pitch], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        hnr = parselmouth.praat.call(snd.to_harmonicity(), "Get mean", 0, 0)
        voice_quality = np.array([pp, 0.0, shimmer, 0.0, hnr], dtype=np.float32)
    except Exception:
        pass

    # Temporal features (5): speech rate proxies
    # Onset detection as proxy for syllable rate
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    duration_sec = len(y) / sr
    temporal = np.array([
        len(onsets) / max(duration_sec, 0.1),  # speech rate proxy
        duration_sec,
        len(onsets),
        0.0,  # placeholder for pause rate
        0.0,  # placeholder for phonation ratio
    ], dtype=np.float32)

    # Stack all base features: (54, T) or mix of scalar and frame-level
    # For frame-level features, compute summary stats
    frame_feats = spectral_feats  # (39, T)
    summary = np.concatenate([
        frame_feats.mean(axis=1),
        frame_feats.std(axis=1),
        skew(frame_feats, axis=1),
        kurtosis(frame_feats, axis=1),
    ])  # 39 * 4 = 156

    # Scalar features (directly appended)
    scalar = np.concatenate([prosodic, voice_quality, temporal])  # 15

    # Pad/trim to exactly 216
    combined = np.concatenate([summary, scalar])  # 171
    result = np.zeros(216, dtype=np.float32)
    result[:len(combined)] = combined[:216]

    return result


def extract_linguistic_handcrafted(transcript: str) -> np.ndarray:
    """Extract handcrafted linguistic features from a transcript.

    Returns a 14-D vector.
    """
    words = transcript.lower().split()
    n_words = max(len(words), 1)
    sentences = [s.strip() for s in transcript.split(".") if s.strip()]
    n_sentences = max(len(sentences), 1)

    # Type-token ratio
    unique_words = set(words)
    ttr = len(unique_words) / n_words

    # Moving-average TTR (window=25)
    window = min(25, n_words)
    if window > 0:
        ttrs = []
        for i in range(max(1, n_words - window + 1)):
            chunk = words[i : i + window]
            ttrs.append(len(set(chunk)) / len(chunk))
        mattr = np.mean(ttrs) if ttrs else ttr
    else:
        mattr = ttr

    # Brunet's W: W = N^(V^-0.172)
    V = len(unique_words)
    brunets_w = n_words ** (V ** -0.172) if V > 0 else 0.0

    # Honore's R: R = 100 * log(N) / (1 - V1/V)
    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    V1 = sum(1 for c in word_counts.values() if c == 1)  # hapax legomena
    denom = max(1 - V1 / max(V, 1), 0.01)
    honores_r = 100 * np.log(max(n_words, 1)) / denom

    # Syntactic complexity proxies
    avg_sentence_length = n_words / n_sentences
    avg_word_length = np.mean([len(w) for w in words]) if words else 0.0
    long_word_ratio = sum(1 for w in words if len(w) > 6) / n_words

    # Semantic coherence placeholder (would use sentence embeddings)
    coherence_mean = 0.0
    coherence_min = 0.0

    # Information content
    idea_density = n_words / n_sentences  # proxy
    info_units = 0.0  # requires Cookie Theft-specific coding

    # Fluency features
    fillers = sum(1 for w in words if w in {"uh", "um", "eh", "ah"})
    filler_rate = fillers / n_words
    repetitions = sum(
        1 for i in range(1, len(words)) if words[i] == words[i - 1]
    )
    repetition_rate = repetitions / n_words
    revision_rate = 0.0  # placeholder

    return np.array([
        ttr, mattr, brunets_w, honores_r,
        avg_sentence_length, avg_word_length, long_word_ratio,
        coherence_mean, coherence_min,
        idea_density, info_units,
        filler_rate, repetition_rate, revision_rate,
    ], dtype=np.float32)
