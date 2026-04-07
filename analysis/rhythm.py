"""
Rhythm analysis module using librosa.
Metrics: BPM, tempo variation, onset precision, IOI.
"""

import numpy as np
import librosa


def analyze(y: np.ndarray, sr: int, config: dict) -> dict:
    """
    Analyses the rhythm of an audio signal.

    Returns dict with:
      - tempo_global: estimated global BPM
      - tempo_curve: dict {times, bpm} — tempo variation over time
      - onset_times: list of onset times (s)
      - onset_deviations_ms: deviation of each onset from the beat grid (ms)
      - mean_deviation_ms: mean deviation (ms)
      - ioi_stats: dict {mean_ms, std_ms, cv} — inter-onset interval stats
      - onset_strength: dict {times, strength} — onset strength envelope
    """
    onset_dev_thr = config.get("onset_deviation_ms", 30)

    # Beat and global tempo detection
    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    tempo_global = float(tempo_arr[0]) if hasattr(tempo_arr, '__len__') else float(tempo_arr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Local tempo curve (overlapping windows)
    tempo_curve = _tempo_curve(y, sr)

    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

    # Onset deviations from beat grid
    deviations_ms = _onset_deviations(onset_times, beat_times.tolist())
    mean_dev = float(np.mean(np.abs(deviations_ms))) if deviations_ms else 0.0

    # Inter-onset intervals
    ioi_stats = _ioi_stats(onset_times)

    # Onset strength envelope (downsampled to avoid bloating the HTML)
    hop = 512
    strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    strength_times = librosa.frames_to_time(np.arange(len(strength)), sr=sr, hop_length=hop)
    # Downsample to max 1000 points
    step = max(1, len(strength) // 1000)
    onset_strength = {
        "times": strength_times[::step].tolist(),
        "strength": strength[::step].tolist(),
    }

    return {
        "tempo_global": tempo_global,
        "tempo_curve": tempo_curve,
        "beat_times": beat_times.tolist(),
        "onset_times": onset_times,
        "onset_deviations_ms": deviations_ms,
        "mean_deviation_ms": mean_dev,
        "ioi_stats": ioi_stats,
        "onset_strength": onset_strength,
        "onset_dev_threshold_ms": onset_dev_thr,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tempo_curve(y: np.ndarray, sr: int) -> dict:
    """Estimates tempo variation in overlapping windows."""
    hop = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    # tempogram: rows = periods, columns = time
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop)
    # Dominant tempo per window
    tempo_per_frame = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop, aggregate=None)
    times = librosa.frames_to_time(np.arange(len(tempo_per_frame)), sr=sr, hop_length=hop)
    step = max(1, len(times) // 500)
    return {
        "times": times[::step].tolist(),
        "bpm": tempo_per_frame[::step].tolist(),
    }


def _onset_deviations(onset_times: list, beat_times: list) -> list:
    """For each onset, calculates the deviation to the nearest beat (ms)."""
    if not beat_times:
        return []
    beats = np.array(beat_times)
    deviations = []
    for t in onset_times:
        nearest = beats[np.argmin(np.abs(beats - t))]
        deviations.append(round((t - nearest) * 1000, 1))
    return deviations


def _ioi_stats(onset_times: list) -> dict:
    """Inter-onset interval statistics."""
    if len(onset_times) < 2:
        return {"mean_ms": 0.0, "std_ms": 0.0, "cv": 0.0}
    ioi = np.diff(onset_times) * 1000  # ms
    mean = float(np.mean(ioi))
    std = float(np.std(ioi))
    cv = std / mean if mean > 0 else 0.0
    return {"mean_ms": round(mean, 1), "std_ms": round(std, 1), "cv": round(cv, 3)}
