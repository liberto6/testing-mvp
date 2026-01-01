"""
Audio Processing Utilities
Handles audio format conversion, resampling, and buffering
"""

import io
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AudioBuffer:
    """Thread-safe audio buffer for streaming"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer = []
        self.total_samples = 0

    def add(self, audio: np.ndarray):
        """Add audio samples to buffer"""
        self.buffer.append(audio)
        self.total_samples += len(audio)

    def get_all(self) -> Optional[np.ndarray]:
        """Get all buffered audio and clear buffer"""
        if not self.buffer:
            return None

        audio = np.concatenate(self.buffer)
        self.clear()
        return audio

    def clear(self):
        """Clear buffer"""
        self.buffer = []
        self.total_samples = 0

    def duration_seconds(self) -> float:
        """Get current buffer duration in seconds"""
        return self.total_samples / self.sample_rate


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate

    Args:
        audio: Audio array (float32, -1.0 to 1.0)
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio

    # Calculate number of samples in resampled audio
    num_samples = int(len(audio) * target_sr / orig_sr)

    # Use scipy's resample for high-quality resampling
    resampled = signal.resample(audio, num_samples)

    return resampled.astype(np.float32)


def normalize_audio(audio: np.ndarray, target_level: float = 0.95) -> np.ndarray:
    """
    Normalize audio to target peak level

    Args:
        audio: Audio array (float32)
        target_level: Target peak level (0.0 to 1.0)

    Returns:
        Normalized audio
    """
    peak = np.abs(audio).max()

    if peak == 0:
        return audio

    return audio * (target_level / peak)


def bytes_to_numpy(audio_bytes: bytes, dtype=np.int16) -> np.ndarray:
    """
    Convert audio bytes to numpy array

    Args:
        audio_bytes: Raw audio bytes
        dtype: Data type of audio (int16, float32, etc.)

    Returns:
        Numpy array (float32, -1.0 to 1.0)
    """
    # Convert to numpy
    audio_np = np.frombuffer(audio_bytes, dtype=dtype)

    # Convert to float32 in range -1.0 to 1.0
    if dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / 32768.0
    elif dtype == np.int32:
        audio_np = audio_np.astype(np.float32) / 2147483648.0

    return audio_np


def numpy_to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int = 24000,
    dtype=np.int16
) -> bytes:
    """
    Convert numpy array to WAV bytes

    Args:
        audio: Audio array (float32, -1.0 to 1.0 or int16)
        sample_rate: Sample rate
        dtype: Output dtype (int16 or int32)

    Returns:
        WAV file as bytes
    """
    # Convert float32 to int16 if needed
    if audio.dtype == np.float32:
        if dtype == np.int16:
            audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        elif dtype == np.int32:
            audio = (audio * 2147483647).clip(-2147483648, 2147483647).astype(np.int32)

    # Write to WAV in memory
    byte_io = io.BytesIO()
    wavfile.write(byte_io, sample_rate, audio)

    return byte_io.getvalue()


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """
    Apply gain to audio

    Args:
        audio: Audio array
        gain_db: Gain in decibels

    Returns:
        Audio with applied gain
    """
    gain_linear = 10 ** (gain_db / 20.0)
    return audio * gain_linear


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    min_silence_duration: float = 0.3
) -> np.ndarray:
    """
    Trim silence from beginning and end of audio

    Args:
        audio: Audio array (float32)
        sample_rate: Sample rate
        threshold_db: Silence threshold in dB
        min_silence_duration: Minimum silence duration to trim (seconds)

    Returns:
        Trimmed audio
    """
    # Calculate threshold in linear scale
    threshold = 10 ** (threshold_db / 20.0)

    # Find samples above threshold
    mask = np.abs(audio) > threshold

    # Find first and last non-silent samples
    nonsilent_indices = np.where(mask)[0]

    if len(nonsilent_indices) == 0:
        return audio

    start_idx = nonsilent_indices[0]
    end_idx = nonsilent_indices[-1] + 1

    return audio[start_idx:end_idx]


def compute_rms(audio: np.ndarray) -> float:
    """
    Compute RMS (Root Mean Square) energy of audio

    Args:
        audio: Audio array

    Returns:
        RMS value
    """
    return np.sqrt(np.mean(audio ** 2))


def compute_db(audio: np.ndarray) -> float:
    """
    Compute decibel level of audio

    Args:
        audio: Audio array

    Returns:
        dB level
    """
    rms = compute_rms(audio)
    if rms == 0:
        return -np.inf
    return 20 * np.log10(rms)


def split_audio_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration_ms: int = 20
) -> list:
    """
    Split audio into fixed-size chunks

    Args:
        audio: Audio array
        sample_rate: Sample rate
        chunk_duration_ms: Chunk duration in milliseconds

    Returns:
        List of audio chunks
    """
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    chunks = []

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        # Pad last chunk if needed
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    # Test audio utilities
    print("🧪 Testing Audio Utilities...\n")

    # Create test audio (1 second sine wave at 440 Hz)
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    print(f"Original audio: {len(audio)} samples @ {sample_rate} Hz")
    print(f"Duration: {duration}s")
    print(f"RMS: {compute_rms(audio):.4f}")
    print(f"dB: {compute_db(audio):.2f}")

    # Test resampling
    resampled = resample_audio(audio, sample_rate, 24000)
    print(f"\nResampled to 24kHz: {len(resampled)} samples")

    # Test normalization
    normalized = normalize_audio(audio * 0.5)
    print(f"\nNormalized peak: {np.abs(normalized).max():.4f}")

    # Test WAV conversion
    wav_bytes = numpy_to_wav_bytes(audio, sample_rate)
    print(f"\nWAV bytes: {len(wav_bytes)} bytes")

    print("\n✅ All tests passed!")
