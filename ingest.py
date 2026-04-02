"""
Módulo de ingesta de audio.
Acepta mp3, wav, m4a, ogg, flac, aac.
Convierte internamente a numpy array mono 22050 Hz.
"""

import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf

SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
TARGET_SR = 22050


def load(filepath: str) -> tuple[np.ndarray, int, dict]:
    """
    Carga un archivo de audio y lo convierte a mono float32 a TARGET_SR.

    Returns:
        y: numpy array mono float32 normalizado
        sr: sample rate (siempre TARGET_SR)
        meta: dict con info del archivo original
    """
    path = os.path.abspath(filepath)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Formato no soportado: {ext}. "
            f"Formatos válidos: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    # Intentar cargar directamente con soundfile (wav, flac, ogg)
    # Para mp3, m4a, aac — usar ffmpeg como intermediario
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

    # Mono
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Resample si es necesario
    if sr_orig != TARGET_SR:
        import librosa
        y = librosa.resample(y.astype(np.float32), orig_sr=sr_orig, target_sr=TARGET_SR)

    y = y.astype(np.float32)

    # Normalización RMS
    y = _rms_normalize(y)

    return y, TARGET_SR, meta


def _convert_with_ffmpeg(input_path: str) -> str:
    """Convierte a WAV temporal usando ffmpeg."""
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
            "ffmpeg no encontrado. Instálalo con: sudo apt install ffmpeg  |  brew install ffmpeg"
        )
    return tmp.name


def _rms_normalize(y: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normaliza el audio a un RMS objetivo."""
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-8:
        return y
    return y * (target_rms / rms)
