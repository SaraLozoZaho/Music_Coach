"""
Separación de fuentes con Demucs (htdemucs_6s).
Stems: drums, bass, guitar, piano, vocals, other

El modelo se descarga automáticamente la primera vez (~2 GB).
Requiere: pip install demucs
"""

import os
import tempfile
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

# Nombres de stems del modelo htdemucs_6s y su etiqueta en español
STEM_LABELS = {
    "drums":  "Batería",
    "bass":   "Bajo",
    "guitar": "Guitarra",
    "piano":  "Piano",
    "vocals": "Voces",
    "other":  "Percusión / Otros",
}

MODEL = "htdemucs_6s"
TARGET_SR = 22050


def separate(audio_path: str, device: str = "cpu") -> dict[str, np.ndarray]:
    """
    Separa el audio en stems usando Demucs.

    Args:
        audio_path: ruta al archivo de audio original (cualquier formato)
        device: "cpu" o "cuda" si hay GPU disponible

    Returns:
        dict {stem_name: np.ndarray mono float32 a TARGET_SR}
    """
    out_dir = tempfile.mkdtemp(prefix="music_coach_demucs_")
    try:
        _run_demucs(audio_path, out_dir, device)
        stems = _load_stems(out_dir, audio_path)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    return stems


def _run_demucs(audio_path: str, out_dir: str, device: str) -> None:
    """Ejecuta Demucs como subproceso."""
    cmd = [
        "python", "-m", "demucs",
        "-n", MODEL,
        "-d", device,
        "-o", out_dir,
        "--mp3",
        audio_path,
    ]
    print(f"     Ejecutando Demucs ({MODEL}) en {device}... puede tardar varios minutos.")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Demucs falló:\n{result.stderr[-2000:]}"
        )


def _load_stems(out_dir: str, audio_path: str) -> dict:
    """Carga los stems generados por Demucs como numpy arrays mono."""
    track_name = Path(audio_path).stem
    stems_dir = Path(out_dir) / MODEL / track_name

    if not stems_dir.exists():
        # Demucs a veces usa el nombre sin extensión o con variaciones
        candidates = list(Path(out_dir).rglob("*.mp3")) + list(Path(out_dir).rglob("*.wav"))
        if not candidates:
            raise FileNotFoundError(f"No se encontraron stems en {out_dir}")
        stems_dir = candidates[0].parent

    stems = {}
    for stem_name in STEM_LABELS:
        # Buscar archivo del stem (mp3 o wav)
        for ext in (".mp3", ".wav"):
            stem_file = stems_dir / f"{stem_name}{ext}"
            if stem_file.exists():
                y, sr = sf.read(str(stem_file))
                if y.ndim > 1:
                    y = y.mean(axis=1)
                y = y.astype(np.float32)
                # Resample si necesario
                if sr != TARGET_SR:
                    import librosa
                    y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
                stems[stem_name] = y
                break

    if not stems:
        raise FileNotFoundError(f"No se encontraron stems válidos en {stems_dir}")
    return stems
