"""
Análisis por instrumento/stem.

Cada stem recibe el análisis más relevante según su naturaleza:
  - Batería:          ritmo (timing, tempo, precisión)
  - Bajo:             ritmo (vs batería) + pitch
  - Guitarra:         pitch + dinámica
  - Piano:            pitch + dinámica
  - Voces:            pitch + dinámica
  - Percusión/Otros:  ritmo + dinámica

El resultado incluye comparativa de timing bajo vs batería para detectar
si el bajista va adelantado o atrasado.
"""

import numpy as np

from analysis import pitch as pitch_mod
from analysis import rhythm as rhythm_mod
from analysis import dynamics as dynamics_mod
from analysis.separation import STEM_LABELS


# Qué análisis hacer por stem
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
    Analiza todos los stems disponibles.

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

    # Comparativa de timing: bajo vs batería
    if "bass" in results and "drums" in results:
        results["_timing_comparison"] = _compare_timing(
            results["bass"]["rhythm"],
            results["drums"]["rhythm"],
        )

    return results


def _compare_timing(bass_rhythm: dict, drums_rhythm: dict) -> dict:
    """
    Compara los onsets del bajo con los de la batería para detectar
    si el bajista va adelantado (+) o atrasado (-) respecto al drummer.

    Returns dict con estadísticas y lista de desfases.
    """
    bass_onsets = np.array(bass_rhythm.get("onset_times", []))
    drums_onsets = np.array(drums_rhythm.get("onset_times", []))

    if len(bass_onsets) == 0 or len(drums_onsets) == 0:
        return {"mean_offset_ms": 0.0, "std_offset_ms": 0.0, "offsets_ms": []}

    # Para cada onset del bajo, encontrar el onset de batería más cercano
    offsets = []
    for bt in bass_onsets:
        nearest_drum = drums_onsets[np.argmin(np.abs(drums_onsets - bt))]
        offset_ms = (bt - nearest_drum) * 1000
        # Solo considerar pares cercanos (< 200 ms)
        if abs(offset_ms) < 200:
            offsets.append(round(float(offset_ms), 1))

    if not offsets:
        return {"mean_offset_ms": 0.0, "std_offset_ms": 0.0, "offsets_ms": []}

    arr = np.array(offsets)
    mean_offset = float(np.mean(arr))
    std_offset = float(np.std(arr))

    # Veredicto
    if abs(mean_offset) < 20:
        verdict = "sincronizado"
    elif mean_offset > 0:
        verdict = f"atrasado {abs(mean_offset):.0f} ms respecto a la batería"
    else:
        verdict = f"adelantado {abs(mean_offset):.0f} ms respecto a la batería"

    return {
        "mean_offset_ms": round(mean_offset, 1),
        "std_offset_ms": round(std_offset, 1),
        "offsets_ms": offsets[::max(1, len(offsets)//500)],  # submuestreado
        "verdict": verdict,
    }


def build_stem_summaries(stem_results: dict, config: dict) -> dict:
    """
    Genera un semáforo y recomendación corta para cada instrumento.
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

        # Ritmo
        if "rhythm" in data:
            dev = data["rhythm"].get("mean_deviation_ms", 0)
            thr = rhythm_cfg.get("onset_deviation_ms", 30)
            if dev > thr * 2:
                issues.append(f"ritmo muy irregular ({dev:.0f} ms)")
                color = "red"
            elif dev > thr:
                issues.append(f"pequeñas irregularidades rítmicas ({dev:.0f} ms)")
                if color == "green":
                    color = "yellow"

        # Pitch
        if "pitch" in data:
            in_tune = data["pitch"].get("in_tune_ratio", 1)
            if in_tune < 0.6:
                issues.append(f"afinación deficiente ({in_tune*100:.0f}% en tono)")
                color = "red"
            elif in_tune < 0.8:
                issues.append(f"afinación mejorable ({in_tune*100:.0f}% en tono)")
                if color == "green":
                    color = "yellow"

        # Dinámica (clipping)
        if "dynamics" in data:
            clip = data["dynamics"].get("clipping_ratio", 0)
            if clip > 0.01:
                issues.append("saturación detectada")
                if color == "green":
                    color = "yellow"

        label = "Sin problemas detectados" if not issues else " · ".join(issues)
        summaries[stem_name] = {
            "color": color,
            "label": label,
            "instrument": STEM_LABELS.get(stem_name, stem_name),
        }

    # Añadir timing comparison si existe
    if "_timing_comparison" in stem_results:
        tc = stem_results["_timing_comparison"]
        summaries["_timing_comparison"] = tc

    return summaries
