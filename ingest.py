"""
Audio ingest module.
Accepts mp3, wav, m4a, ogg, flac, aac.
Converts internally to a mono numpy array at 22050 Hz.
"""

import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf

SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
TARGET_SR = 22050


def load(filepath: str) -> tuple:
    """
    Loads an audio file and converts it to mono float32 at TARGET_SR.

    Returns:
        y: normalised mono float32 numpy array
        sr: sample rate (always TARGET_SR)
        meta: dict with info about the original file
    """
    path = os.path.abspath(filepath)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {ext}. "
            f"Valid formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    # Load directly with soundfile (wav, flac, ogg)
    # For mp3, m4a, aac — use ffmpeg as intermediary
    if ext in {".mp3", ".m4a", ".aac"}:
        wav_path = _convert_with_ffmpeg(path)
        try:
            y, sr_orig = sf.read(wav_path)
        finally:
            os.unlink(wav_path)
    else:
        y, sr_orig = sf.read(path)

    meta = {
        "filename": os.path.basename(path),
        "original_sr": sr_orig,
        "original_channels": y.ndim if y.ndim > 1 else 1,
        "duration_s": round(len(y) / sr_orig, 2) if y.ndim == 1 else round(y.shape[0] / sr_orig, 2),
        "format": ext,
    }

    # Convert to mono
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Resample if needed
    if sr_orig != TARGET_SR:
        import librosa
        y = librosa.resample(y.astype(np.float32), orig_sr=sr_orig, target_sr=TARGET_SR)

    y = y.astype(np.float32)

    # RMS normalisation
    y = _rms_normalize(y)

    return y, TARGET_SR, meta


def _convert_with_ffmpeg(input_path: str) -> str:
    """Converts audio to a temporary WAV file using ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-ac", "1",
                "-ar", str(TARGET_SR),
                "-f", "wav",
                tmp.name,
            ],
            capture_output=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg error:\n{result.stderr.decode(errors='replace')}"
            )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install it with: sudo apt install ffmpeg  |  brew install ffmpeg"
        )
    return tmp.name


def _rms_normalize(y: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normalises audio to a target RMS level."""
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-8:
        return y
    return y * (target_rms / rms)
