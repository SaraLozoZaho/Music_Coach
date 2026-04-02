"""
Módulo de análisis de calidad global.
Métricas: SNR, spectral centroid, zero-crossing rate, mel spectrogram, MFCC.
"""

import numpy as np
import librosa


def analyze(y: np.ndarray, sr: int, config: dict) -> dict:
    """
    Analiza la calidad global de la grabación.

    Returns dict con:
      - snr_db: Signal-to-Noise Ratio estimado (dB)
      - spectral_centroid_mean: centroide espectral medio (Hz)
      - spectral_rolloff_mean: rolloff espectral medio (Hz)
      - zcr_mean: zero-crossing rate media
      - silence_ratio: fracción del tiempo en silencio
      - mel_spectrogram: dict {times, freqs_mel, magnitude_db} submuestreado
      - mfcc_mean: vector de 13 coeficientes MFCC medios
    """
    min_snr = config.get("min_snr_db", 20)
    hop = 512

    # SNR estimado: relación entre RMS total y RMS del ruido de fondo
    # El ruido de fondo se estima como el percentil 10 del RMS
    rms_frames = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms_frames, ref=np.max)
    noise_floor = float(np.percentile(rms_db, 10))
    signal_level = float(np.percentile(rms_db, 90))
    snr_db = round(signal_level - noise_floor, 1)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)[0]

    # Silencios: frames con RMS < -60 dB (relativo al máximo)
    silence_ratio = float(np.mean(rms_db < -40))

    # Mel spectrogram (submuestreado)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop, n_mels=64)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    time_step = max(1, mel_db.shape[1] // 300)
    mel_times = librosa.frames_to_time(
        np.arange(mel_db.shape[1]), sr=sr, hop_length=hop
    )[::time_step]
    mel_freqs = librosa.mel_frequencies(n_mels=64, fmin=0, fmax=sr // 2)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    mfcc_mean = mfcc.mean(axis=1).tolist()

    return {
        "snr_db": snr_db,
        "spectral_centroid_mean": round(float(np.mean(centroid)), 1),
        "spectral_rolloff_mean": round(float(np.mean(rolloff)), 1),
        "zcr_mean": round(float(np.mean(zcr)), 4),
        "silence_ratio": round(silence_ratio, 3),
        "mel_spectrogram": {
            "times": mel_times.tolist(),
            "freqs_mel": mel_freqs.tolist(),
            "magnitude_db": mel_db[:, ::time_step].tolist(),
        },
        "mfcc_mean": [round(v, 2) for v in mfcc_mean],
        "min_snr_db": min_snr,
    }
