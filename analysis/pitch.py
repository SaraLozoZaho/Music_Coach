"""
Pitch analysis module using CREPE.
CREPE is monophonic — works best with a solo voice or single instrument.
For polyphonic signals, source separation with Demucs is recommended first.
"""

import numpy as np


def analyze(y: np.ndarray, sr: int, config: dict) -> dict:
    """
    Analyses the pitch of an audio signal.

    Returns dict with:
      - times: array of time positions (s)
      - frequencies: array of estimated frequencies (Hz)
      - confidence: CREPE confidence array [0,1]
      - cents_deviation: deviation from the nearest note (cents)
      - stability_std: global pitch standard deviation (cents)
      - in_tune_ratio: fraction of time within ±tolerance_cents of the note
      - problematic_segments: list of {t_start, t_end, mean_deviation_cents}
    """
    try:
        import crepe
        audio_f32 = y.astype(np.float32)
        times, frequencies, confidence, _ = crepe.predict(
            audio_f32, sr, viterbi=True, verbose=0
        )
    except (ImportError, ModuleNotFoundError, Exception):
        return _fallback_pitch(y, sr, config)

    tolerance = config.get("tolerance_cents", 20)
    stability_thr = config.get("stability_threshold", 15)

    # Filter low-confidence frames
    mask = confidence > 0.5
    if mask.sum() == 0:
        return _empty_result()

    freqs_valid = frequencies.copy()
    freqs_valid[~mask] = np.nan

    # Convert Hz → cents relative to nearest MIDI note
    cents_dev = _hz_to_cents_deviation(freqs_valid)

    # Global stability
    valid_cents = cents_dev[~np.isnan(cents_dev)]
    stability_std = float(np.std(valid_cents)) if len(valid_cents) > 0 else 0.0

    # In-tune ratio
    in_tune_ratio = float(
        np.mean(np.abs(valid_cents) <= tolerance)
    ) if len(valid_cents) > 0 else 0.0

    # Problematic segments: 2s windows where mean deviation > tolerance
    problematic = _find_problematic_segments(times, cents_dev, tolerance, window_s=2.0)

    return {
        "times": times.tolist(),
        "frequencies": np.where(np.isnan(freqs_valid), None, freqs_valid).tolist(),
        "confidence": confidence.tolist(),
        "cents_deviation": np.where(np.isnan(cents_dev), None, cents_dev).tolist(),
        "stability_std": stability_std,
        "in_tune_ratio": in_tune_ratio,
        "problematic_segments": problematic,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hz_to_cents_deviation(frequencies: np.ndarray) -> np.ndarray:
    """Returns the deviation in cents from the nearest MIDI note."""
    cents_dev = np.full_like(frequencies, np.nan)
    valid = ~np.isnan(frequencies) & (frequencies > 0)
    f = frequencies[valid]
    # Continuous MIDI note
    midi_float = 69 + 12 * np.log2(f / 440.0)
    midi_nearest = np.round(midi_float)
    # Frequency of the nearest note
    f_nearest = 440.0 * 2 ** ((midi_nearest - 69) / 12)
    cents_dev[valid] = 1200 * np.log2(f / f_nearest)
    return cents_dev


def _find_problematic_segments(
    times: np.ndarray,
    cents_dev: np.ndarray,
    tolerance: float,
    window_s: float = 2.0,
) -> list:
    """Finds windows where the mean deviation exceeds the tolerance."""
    if len(times) == 0:
        return []

    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    window_frames = max(1, int(window_s / dt))
    segments = []

    i = 0
    while i < len(times):
        window = cents_dev[i : i + window_frames]
        valid = window[~np.isnan(window.astype(float))]
        if len(valid) > window_frames * 0.5:
            mean_dev = float(np.mean(np.abs(valid)))
            if mean_dev > tolerance:
                t_start = float(times[i])
                t_end = float(times[min(i + window_frames - 1, len(times) - 1)])
                segments.append(
                    {"t_start": t_start, "t_end": t_end, "mean_deviation_cents": round(mean_dev, 1)}
                )
                i += window_frames
                continue
        i += window_frames // 2 or 1

    return segments


def _fallback_pitch(y: np.ndarray, sr: int, config: dict) -> dict:
    """Fallback using librosa's pYIN if CREPE is not installed."""
    import librosa

    tolerance = config.get("tolerance_cents", 20)

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    times = librosa.times_like(f0, sr=sr)
    freqs = np.where(voiced_flag, f0, np.nan)
    cents_dev = _hz_to_cents_deviation(freqs)
    valid_cents = cents_dev[~np.isnan(cents_dev)]
    stability_std = float(np.std(valid_cents)) if len(valid_cents) > 0 else 0.0
    in_tune_ratio = float(np.mean(np.abs(valid_cents) <= tolerance)) if len(valid_cents) > 0 else 0.0
    problematic = _find_problematic_segments(times, cents_dev, tolerance)

    return {
        "times": times.tolist(),
        "frequencies": np.where(np.isnan(freqs), None, freqs).tolist(),
        "confidence": voiced_flag.astype(float).tolist(),
        "cents_deviation": np.where(np.isnan(cents_dev), None, cents_dev).tolist(),
        "stability_std": stability_std,
        "in_tune_ratio": in_tune_ratio,
        "problematic_segments": problematic,
        "status": "fallback_pyin",
    }


def _empty_result() -> dict:
    return {
        "times": [],
        "frequencies": [],
        "confidence": [],
        "cents_deviation": [],
        "stability_std": 0.0,
        "in_tune_ratio": 0.0,
        "problematic_segments": [],
        "status": "no_pitch_detected",
    }
