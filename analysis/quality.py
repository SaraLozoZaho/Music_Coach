"""
Global quality analysis module.
Metrics: SNR, spectral centroid, zero-crossing rate, mel spectrogram, MFCC.
"""

import numpy as np
import librosa


def analyze(y: np.ndarray, sr: int, config: dict) -> dict:
    """
    Analyses the global quality of a recording.

    Returns dict with:
      - snr_db: estimated Signal-to-Noise Ratio (dB)
      - spectral_centroid_mean: mean spectral centroid (Hz)
      - spectral_rolloff_mean: mean spectral rolloff (Hz)
      - zcr_mean: mean zero-crossing rate
      - silence_ratio: fraction of time in silence
      - mel_spectrogram: dict {times, freqs_mel, magnitude_db} downsampled
      - mfcc_mean: vector of 13 mean MFCC coefficients
    """
    min_snr = config.get("min_snr_db", 20)
    hop = 512

    # Estimated SNR: ratio between total RMS and background noise RMS
    # Background noise is estimated as the 10th percentile of the RMS
    rms_frames = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms_frames, ref=np.max)
    noise_floor = float(np.percentile(rms_db, 10))
    signal_level = float(np.percentile(rms_db, 90))
    snr_db = round(signal_level - noise_floor, 1)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)[0]

    # Silences: frames with RMS < -40 dB (relative to max)
    silence_ratio = float(np.mean(rms_db < -40))

    # Mel spectrogram (downsampled)
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
