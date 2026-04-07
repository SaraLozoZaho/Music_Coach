"""
Per-instrument/stem analysis.

Each stem receives the most relevant analysis for its nature:
  - Drums:            rhythm (timing, tempo, precision)
  - Bass:             rhythm (vs drums) + pitch + dynamics
  - Guitar:           rhythm (vs drums) + pitch + dynamics
  - Piano:            rhythm (vs drums) + pitch + dynamics
  - Vocals:           rhythm (vs drums) + pitch + dynamics
  - Percussion/Other: rhythm (vs drums) + dynamics

Drums are always analysed first so their beat grid and tempo can be passed
to all other instruments as an authoritative reference.
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
    "guitar": ["rhythm", "pitch", "dynamics"],
    "piano":  ["rhythm", "pitch", "dynamics"],
    "vocals": ["rhythm", "pitch", "dynamics"],
    "other":  ["rhythm", "dynamics"],
}

SR = 22050


def analyze_all(stems: dict, config: dict) -> dict:
    """
    Analyses all available stems.

    Drums are analysed first with full beat tracking.
    All other instruments receive the drums' beat grid AND exact tempo so that
    tempo is identical across the band and deviations are measured against
    the actual drum beat — not an independent per-instrument estimate.
    """
    results = {}
    drums_beat_times = None
    drums_tempo = None

    # ── Drums first — they define the beat grid for the whole band ──────────
    if "drums" in stems:
        result = {"label": STEM_LABELS["drums"]}
        result["rhythm"] = rhythm_mod.analyze(stems["drums"], SR, config.get("rhythm", {}))
        result["dynamics"] = dynamics_mod.analyze(stems["drums"], SR, config.get("dynamics", {}))
        results["drums"] = result
        drums_beat_times = result["rhythm"]["beat_times"]
        drums_tempo = result["rhythm"]["tempo_global"]

    # ── All other stems use drums beat grid + exact drums tempo ─────────────
    for stem_name, y in stems.items():
        if stem_name == "drums":
            continue
        analyses = STEM_ANALYSES.get(stem_name, ["dynamics"])
        result = {"label": STEM_LABELS.get(stem_name, stem_name)}

        if "rhythm" in analyses:
            result["rhythm"] = rhythm_mod.analyze(
                y, SR, config.get("rhythm", {}),
                reference_beats=drums_beat_times,
                reference_tempo=drums_tempo,
            )
        if "pitch" in analyses:
            result["pitch"] = pitch_mod.analyze(y, SR, config.get("pitch", {}))
        if "dynamics" in analyses:
            result["dynamics"] = dynamics_mod.analyze(y, SR, config.get("dynamics", {}))

        results[stem_name] = result

    # ── Timing comparisons: each melodic instrument vs drums ────────────────
    if "drums" in results:
        comparisons = {}
        for stem_name in ("bass", "guitar", "piano", "vocals"):
            if stem_name in results and "rhythm" in results[stem_name]:
                comparisons[stem_name] = _compare_timing(
                    results[stem_name]["rhythm"],
                    results["drums"]["rhythm"],
                )
        results["_timing_comparisons"] = comparisons

    return results


def _compare_timing(inst_rhythm: dict, drums_rhythm: dict) -> dict:
    """
    Compares instrument onsets against drum onsets.
    Returns whether the instrument is behind (+) or ahead (-) of the drummer.
    """
    inst_onsets = np.array(inst_rhythm.get("onset_times", []))
    drums_onsets = np.array(drums_rhythm.get("onset_times", []))

    if len(inst_onsets) == 0 or len(drums_onsets) == 0:
        return {"mean_offset_ms": 0.0, "std_offset_ms": 0.0, "offsets_ms": []}

    offsets = []
    for bt in inst_onsets:
        nearest_drum = drums_onsets[np.argmin(np.abs(drums_onsets - bt))]
        offset_ms = (bt - nearest_drum) * 1000
        if abs(offset_ms) < 200:
            offsets.append(round(float(offset_ms), 1))

    if not offsets:
        return {"mean_offset_ms": 0.0, "std_offset_ms": 0.0, "offsets_ms": []}

    arr = np.array(offsets)
    mean_offset = float(np.mean(arr))
    std_offset = float(np.std(arr))

    if abs(mean_offset) < 20:
        verdict = "in sync"
    elif mean_offset > 0:
        verdict = f"behind the drums by {abs(mean_offset):.0f} ms"
    else:
        verdict = f"ahead of the drums by {abs(mean_offset):.0f} ms"

    return {
        "mean_offset_ms": round(mean_offset, 1),
        "std_offset_ms": round(std_offset, 1),
        "offsets_ms": offsets[::max(1, len(offsets)//500)],
        "verdict": verdict,
    }


def build_stem_summaries(stem_results: dict, config: dict) -> dict:
    """
    Builds a traffic-light status and short label for each instrument.
    """
    summaries = {}
    rhythm_cfg = config.get("rhythm", {})

    for stem_name, data in stem_results.items():
        if stem_name.startswith("_"):
            continue

        issues = []
        color = "green"

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

        if "pitch" in data:
            in_tune = data["pitch"].get("in_tune_ratio", 1)
            if in_tune < 0.6:
                issues.append(f"poor tuning ({in_tune*100:.0f}% in tune)")
                color = "red"
            elif in_tune < 0.8:
                issues.append(f"tuning needs work ({in_tune*100:.0f}% in tune)")
                if color == "green":
                    color = "yellow"

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

    if "_timing_comparisons" in stem_results:
        summaries["_timing_comparisons"] = stem_results["_timing_comparisons"]

    return summaries
