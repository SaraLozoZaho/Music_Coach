"""
Dynamics analysis module.
Metrics: RMS curve, dynamic range, clipping, spectrogram, LUFS.
"""

import numpy as np
import librosa


def analyze(y: np.ndarray, sr: int, config: dict) -> dict:
    """
    Analyses the dynamics of an audio signal.

    Returns dict with:
      - rms_times: RMS envelope time positions (s)
      - rms_db: RMS energy in dB
      - dynamic_range_db: p95-p5 difference of RMS (dB)
      - clipping_ratio: fraction of samples above the clipping threshold
      - lufs: integrated loudness EBU R128 (if pyloudnorm is available)
      - spectrogram: dict {times, freqs, magnitude_db} downsampled
    """
    min_range = config.get("min_dynamic_range_db", 10)
    clip_thr_db = config.get("clipping_threshold_db", -1)

    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    # Convert to dB avoiding log(0)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Dynamic range
    p5 = float(np.percentile(rms_db, 5))
    p95 = float(np.percentile(rms_db, 95))
    dynamic_range_db = round(p95 - p5, 1)

    # Clipping: samples that exceed the threshold in dB
    clip_linear = librosa.db_to_amplitude(clip_thr_db)
    clipping_ratio = float(np.mean(np.abs(y) >= clip_linear))

    # LUFS
    lufs = _compute_lufs(y, sr)

    # Amplitude spectrogram (downsampled)
    spec = np.abs(librosa.stft(y, hop_length=hop))
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)
    # Downsample frequencies and time to keep HTML light
    freq_step = max(1, spec_db.shape[0] // 128)
    time_step = max(1, spec_db.shape[1] // 300)
    spec_sub = spec_db[::freq_step, ::time_step]
    freqs = librosa.fft_frequencies(sr=sr)[::freq_step]
    spec_times = librosa.frames_to_time(
        np.arange(spec_db.shape[1]), sr=sr, hop_length=hop
    )[::time_step]

    # Downsample RMS for HTML
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
    """Calculates integrated LUFS (EBU R128) using pyloudnorm."""
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        # pyloudnorm expects (samples, channels)
        audio_2d = y[:, np.newaxis] if y.ndim == 1 else y.T
        loudness = meter.integrated_loudness(audio_2d)
        return round(float(loudness), 1)
    except Exception:
        return None
