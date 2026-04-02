"""
Módulo de análisis de dinámica.
Métricas: curva RMS, rango dinámico, clipping, espectrograma, LUFS.
"""

import numpy as np
import librosa


def analyze(y: np.ndarray, sr: int, config: dict) -> dict:
    """
    Analiza la dinámica de la señal de audio.

    Returns dict con:
      - rms_times: tiempos del envelope RMS (s)
      - rms_db: energía RMS en dB
      - dynamic_range_db: diferencia p95-p5 del RMS (dB)
      - clipping_ratio: fracción de muestras por encima del umbral de clipping
      - lufs: loudness integrado EBU R128 (si pyloudnorm disponible)
      - spectrogram: dict {times, freqs, magnitude_db} submuestreado
    """
    min_range = config.get("min_dynamic_range_db", 10)
    clip_thr_db = config.get("clipping_threshold_db", -1)

    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    # Convertir a dB evitando log(0)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Rango dinámico
    p5 = float(np.percentile(rms_db, 5))
    p95 = float(np.percentile(rms_db, 95))
    dynamic_range_db = round(p95 - p5, 1)

    # Clipping: muestras que superan el umbral en dB
    clip_linear = librosa.db_to_amplitude(clip_thr_db)
    clipping_ratio = float(np.mean(np.abs(y) >= clip_linear))

    # LUFS
    lufs = _compute_lufs(y, sr)

    # Espectrograma de amplitud (submuestreado)
    spec = np.abs(librosa.stft(y, hop_length=hop))
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)
    # Submuestrear frecuencias y tiempo para HTML ligero
    freq_step = max(1, spec_db.shape[0] // 128)
    time_step = max(1, spec_db.shape[1] // 300)
    spec_sub = spec_db[::freq_step, ::time_step]
    freqs = librosa.fft_frequencies(sr=sr)[::freq_step]
    spec_times = librosa.frames_to_time(
        np.arange(spec_db.shape[1]), sr=sr, hop_length=hop
    )[::time_step]

    # Submuestrear RMS para HTML
    rms_step = max(1, len(rms_times) // 1000)

    return {
        "rms_times": rms_times[::rms_step].tolist(),
        "rms_db": rms_db[::rms_step].tolist(),
        "dynamic_range_db": dynamic_range_db,
        "clipping_ratio": round(clipping_ratio, 4),
        "lufs": lufs,
        "spectrogram": {
            "times": spec_times.tolist(),
            "freqs": freqs.tolist(),
            "magnitude_db": spec_sub.tolist(),
        },
        "min_dynamic_range_db": min_range,
    }


def _compute_lufs(y: np.ndarray, sr: int):
    """Calcula LUFS integrado (EBU R128) con pyloudnorm."""
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        # pyloudnorm espera (samples, channels)
        audio_2d = y[:, np.newaxis] if y.ndim == 1 else y.T
        loudness = meter.integrated_loudness(audio_2d)
        return round(float(loudness), 1)
    except Exception:
        return None
