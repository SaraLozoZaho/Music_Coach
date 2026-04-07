"""
Per-instrument/stem analysis.

Each stem receives the most relevant analysis for its nature:
  - Drums:            rhythm (timing, tempo, precision)
  - Bass:             rhythm (vs drums) + pitch
  - Guitar:           pitch + dynamics
  - Piano:            pitch + dynamics
  - Vocals:           pitch + dynamics
  - Percussion/Other: rhythm + dynamics

The result includes a timing comparison between bass and drums to detect
whether the bassist is ahead of or behind the drummer.
"""

import numpy as np

from analysis import pitch as pitch_mod
from analysis import rhythm as rhythm_mod
from analysis import dynamics as dynamics_mod
from analysis.separation import STEM_LABELS


# Which analyses to run per stem
STEM_ANALYSES = {
    "drums":  ["rhythm", "dynamics"],
    "bass":   ["rhythm", "pitch", "dynamics"],
    "guitar": ["pitch", "dynamics"],
    "piano":  ["pitch", "dynamics"],
    "vocals": ["pitch", "dynamics"],
    "other":  ["rhythm", "dynamics"],
}

SR = 22050


def analyze_all(stems: dict, config: dict) -> dict:
    """
    Analyses all available stems.

    Returns:
        dict {stem_name: {"label": str, "rhythm": ..., "pitch": ..., "dynamics": ...}}
    """
    results = {}
    for stem_name, y in stems.items():
        analyses = STEM_ANALYSES.get(stem_name, ["dynamics"])
        result = {"label": STEM_LABELS.get(stem_name, stem_name)}

        if "rhythm" in analyses:
            result["rhythm"] = rhythm_mod.analyze(y, SR, config.get("rhythm", {}))
        if "pitch" in analyses:
            result["pitch"] = pitch_mod.analyze(y, SR, config.get("pitch", {}))
        if "dynamics" in analyses:
            result["dynamics"] = dynamics_mod.analyze(y, SR, config.get("dynamics", {}))

        results[stem_name] = result

    # Timing comparison: bass vs drums
    if "bass" in results and "drums" in results:
        results["_timing_comparison"] = _compare_timing(
            results["bass"]["rhythm"],
            results["drums"]["rhythm"],
        )

    return results


def _compare_timing(bass_rhythm: dict, drums_rhythm: dict) -> dict:
    """
    Compares bass onsets against drum onsets to detect whether the bassist
    is behind (+) or ahead (-) of the drummer.

    Returns dict with statistics and list of offsets.
    """
    bass_onsets = np.array(bass_rhythm.get("onset_times", []))
    drums_onsets = np.array(drums_rhythm.get("onset_times", []))

    if len(bass_onsets) == 0 or len(drums_onsets) == 0:
        return {"mean_offset_ms": 0.0, "std_offset_ms": 0.0, "offsets_ms": []}

    # For each bass onset, find the nearest drum onset
    offsets = []
    for bt in bass_onsets:
        nearest_drum = drums_onsets[np.argmin(np.abs(drums_onsets - bt))]
        offset_ms = (bt - nearest_drum) * 1000
        # Only consider close pairs (< 200 ms)
        if abs(offset_ms) < 200:
            offsets.append(round(float(offset_ms), 1))

    if not offsets:
        return {"mean_offset_ms": 0.0, "std_offset_ms": 0.0, "offsets_ms": []}

    arr = np.array(offsets)
    mean_offset = float(np.mean(arr))
    std_offset = float(np.std(arr))

    # Verdict
    if abs(mean_offset) < 20:
        verdict = "in sync"
    elif mean_offset > 0:
        verdict = f"behind the drums by {abs(mean_offset):.0f} ms"
    else:
        verdict = f"ahead of the drums by {abs(mean_offset):.0f} ms"

    return {
        "mean_offset_ms": round(mean_offset, 1),
        "std_offset_ms": round(std_offset, 1),
        "offsets_ms": offsets[::max(1, len(offsets)//500)],  # downsampled
        "verdict": verdict,
    }


def build_stem_summaries(stem_results: dict, config: dict) -> dict:
    """
    Builds a traffic-light status and short label for each instrument.
    """
    summaries = {}
    pitch_cfg = config.get("pitch", {})
    rhythm_cfg = config.get("rhythm", {})
    dyn_cfg = config.get("dynamics", {})

    for stem_name, data in stem_results.items():
        if stem_name.startswith("_"):
            continue

        issues = []
        color = "green"

        # Rhythm
        if "rhythm" in data:
            dev = data["rhythm"].get("mean_deviation_ms", 0)
            thr = rhythm_cfg.get("onset_deviation_ms", 30)
            if dev > thr * 2:
                issues.append(f"very irregular rhythm ({dev:.0f} ms)")
                color = "red"
            elif dev > thr:
                issues.append(f"minor rhythmic irregularities ({dev:.0f} ms)")
                if color == "green":
                    color = "yellow"

        # Pitch
        if "pitch" in data:
            in_tune = data["pitch"].get("in_tune_ratio", 1)
            if in_tune < 0.6:
                issues.append(f"poor tuning ({in_tune*100:.0f}% in tune)")
                color = "red"
            elif in_tune < 0.8:
                issues.append(f"tuning needs work ({in_tune*100:.0f}% in tune)")
                if color == "green":
                    color = "yellow"

        # Dynamics (clipping)
        if "dynamics" in data:
            clip = data["dynamics"].get("clipping_ratio", 0)
            if clip > 0.01:
                issues.append("clipping detected")
                if color == "green":
                    color = "yellow"

        label = "No issues detected" if not issues else " · ".join(issues)
        summaries[stem_name] = {
            "color": color,
            "label": label,
            "instrument": STEM_LABELS.get(stem_name, stem_name),
        }

    # Add timing comparison if available
    if "_timing_comparison" in stem_results:
        tc = stem_results["_timing_comparison"]
        summaries["_timing_comparison"] = tc

    return summaries
