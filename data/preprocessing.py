"""
preprocessing.py — Audio preprocessing utilities.

Two main functions:
1. remove_silence() — Uses Voice Activity Detection (VAD) to strip silent
   segments from audio. This can help the model focus on actual speech.
2. chunk_audio() — Splits long recordings into shorter overlapping chunks
   so each sample has a consistent length.
"""

import numpy as np
import struct


def remove_silence(audio, sample_rate, aggressiveness=2):
    """
    Remove silence from audio using WebRTC Voice Activity Detection.

    How it works:
    - Splits audio into small frames (30ms each)
    - Uses WebRTC VAD to classify each frame as speech or silence
    - Keeps only the speech frames and concatenates them

    Args:
        audio: numpy array of audio samples (float32, mono)
        sample_rate: sample rate in Hz (must be 8000, 16000, 32000, or 48000)
        aggressiveness: 0-3, higher = more aggressive silence removal

    Returns:
        numpy array with silence removed
    """
    try:
        import webrtcvad
    except ImportError:
        print("WARNING: webrtcvad not installed. Skipping silence removal.")
        print("Install it with: pip install webrtcvad")
        return audio

    # WebRTC VAD requires 16-bit PCM audio at specific sample rates
    valid_rates = [8000, 16000, 32000, 48000]
    if sample_rate not in valid_rates:
        print(f"WARNING: VAD requires sample rate in {valid_rates}, got {sample_rate}.")
        print("Skipping silence removal.")
        return audio

    # Create the VAD object
    vad = webrtcvad.Vad(aggressiveness)

    # Convert float audio [-1, 1] to 16-bit PCM bytes
    # (VAD works on raw bytes, not floats)
    audio_int16 = (audio * 32767).astype(np.int16)
    raw_bytes = audio_int16.tobytes()

    # Frame size: VAD accepts 10, 20, or 30 ms frames
    frame_duration_ms = 30
    frame_size = int(sample_rate * frame_duration_ms / 1000)  # samples per frame
    frame_bytes = frame_size * 2  # 2 bytes per int16 sample

    # Process each frame and keep only voiced frames
    voiced_frames = []
    for i in range(0, len(raw_bytes) - frame_bytes + 1, frame_bytes):
        frame = raw_bytes[i:i + frame_bytes]
        # is_speech returns True if the frame contains speech
        if vad.is_speech(frame, sample_rate):
            voiced_frames.append(frame)

    if len(voiced_frames) == 0:
        # If VAD removed everything, return original audio
        print("WARNING: VAD removed all audio. Returning original.")
        return audio

    # Convert back to numpy float32
    voiced_bytes = b"".join(voiced_frames)
    voiced_audio = np.frombuffer(voiced_bytes, dtype=np.int16).astype(np.float32) / 32767.0

    return voiced_audio


def chunk_audio(audio, sample_rate, chunk_length_sec=5.0, overlap_sec=1.0):
    """
    Split a long audio recording into shorter overlapping chunks.

    Why chunk?
    - Neural networks expect fixed-size inputs
    - Long recordings can be memory-intensive
    - Overlapping chunks provide data augmentation

    Example with chunk_length=5s, overlap=1s:
        [0s----5s]
              [4s----9s]
                    [8s----13s]  ...

    Args:
        audio: numpy array of audio samples
        sample_rate: sample rate in Hz
        chunk_length_sec: length of each chunk in seconds
        overlap_sec: overlap between consecutive chunks in seconds

    Returns:
        list of numpy arrays, each one chunk of audio
    """
    chunk_samples = int(chunk_length_sec * sample_rate)
    overlap_samples = int(overlap_sec * sample_rate)
    step = chunk_samples - overlap_samples  # how far to advance each chunk

    chunks = []
    start = 0

    while start < len(audio):
        end = start + chunk_samples
        chunk = audio[start:end]

        # Pad the last chunk with zeros if it's too short
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        chunks.append(chunk)
        start += step

    return chunks
