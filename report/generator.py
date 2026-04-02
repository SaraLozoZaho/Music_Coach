"""
Generador de informe HTML interactivo con Jinja2 + Plotly.
El HTML resultante es autocontenido y compatible con GitHub Pages.
"""

import os
import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "report.html.j2"


def generate(
    meta: dict,
    pitch: dict,
    rhythm: dict,
    dynamics: dict,
    quality: dict,
    config: dict,
    output_dir: str = "docs/reports",
    stem_results: dict = None,
    stem_summaries: dict = None,
) -> str:
    """
    Genera el informe HTML y lo guarda en output_dir.
    Devuelve la ruta absoluta del archivo generado.
    """
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.html"
    output_path = os.path.join(output_dir, filename)

    summary = _build_summary(pitch, rhythm, dynamics, quality, config)
    recommendations = _build_recommendations(pitch, rhythm, dynamics, quality, config)

    # Recomendaciones por instrumento (si hay separación de fuentes)
    stem_recommendations = {}
    if stem_results:
        stem_recommendations = _build_stem_recommendations(stem_results, stem_summaries, config)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,
    )
    env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)

    template = env.get_template(TEMPLATE_NAME)
    html = template.render(
        meta=meta,
        date=date_str,
        pitch=pitch,
        rhythm=rhythm,
        dynamics=dynamics,
        quality=quality,
        config=config,
        summary=summary,
        recommendations=recommendations,
        stem_results=stem_results or {},
        stem_summaries=stem_summaries or {},
        stem_recommendations=stem_recommendations,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Semáforo
# ---------------------------------------------------------------------------

def _build_summary(pitch, rhythm, dynamics, quality, config) -> dict:
    pitch_cfg = config.get("pitch", {})
    rhythm_cfg = config.get("rhythm", {})
    dyn_cfg = config.get("dynamics", {})
    qual_cfg = config.get("quality", {})

    # Afinación: % tiempo en afinación
    in_tune = pitch.get("in_tune_ratio", 0)
    pitch_color = "green" if in_tune >= 0.8 else ("yellow" if in_tune >= 0.6 else "red")
    pitch_label = {"green": "Buena", "yellow": "Mejorable", "red": "Requiere atención"}[pitch_color]

    # Ritmo: desviación media de onsets
    dev_ms = rhythm.get("mean_deviation_ms", 999)
    thr = rhythm_cfg.get("onset_deviation_ms", 30)
    rhythm_color = "green" if dev_ms <= thr else ("yellow" if dev_ms <= thr * 2 else "red")
    rhythm_label = {"green": "Preciso", "yellow": "Mejorable", "red": "Inestable"}[rhythm_color]

    # Dinámica: rango dinámico
    dr = dynamics.get("dynamic_range_db", 0)
    min_dr = dyn_cfg.get("min_dynamic_range_db", 10)
    clip = dynamics.get("clipping_ratio", 0)
    if clip > 0.01:
        dyn_color = "red"
    elif dr >= min_dr:
        dyn_color = "green"
    else:
        dyn_color = "yellow"
    dyn_label = {"green": "Buena", "yellow": "Rango bajo", "red": "Saturación"}[dyn_color]

    # Calidad: SNR
    snr = quality.get("snr_db", 0)
    min_snr = qual_cfg.get("min_snr_db", 20)
    qual_color = "green" if snr >= min_snr else ("yellow" if snr >= min_snr * 0.7 else "red")
    qual_label = {"green": "Buena", "yellow": "Aceptable", "red": "Ruido excesivo"}[qual_color]

    return {
        "pitch": {"color": pitch_color, "label": pitch_label},
        "rhythm": {"color": rhythm_color, "label": rhythm_label},
        "dynamics": {"color": dyn_color, "label": dyn_label},
        "quality": {"color": qual_color, "label": qual_label},
    }


# ---------------------------------------------------------------------------
# Recomendaciones textuales
# ---------------------------------------------------------------------------

def _build_recommendations(pitch, rhythm, dynamics, quality, config) -> list:
    recs = []
    pitch_cfg = config.get("pitch", {})
    rhythm_cfg = config.get("rhythm", {})
    dyn_cfg = config.get("dynamics", {})
    qual_cfg = config.get("quality", {})

    in_tune = pitch.get("in_tune_ratio", 1)
    tol = pitch_cfg.get("tolerance_cents", 20)
    if in_tune < 0.6:
        recs.append({
            "icon": "🎵",
            "title": "Afinación — Atención urgente",
            "text": f"Solo el {in_tune*100:.1f}% del tiempo estás dentro de ±{tol} cents. "
                    "Practica con un afinador de referencia (clic o drone) para interiorizar la afinación."
        })
    elif in_tune < 0.8:
        recs.append({
            "icon": "🎵",
            "title": "Afinación — Mejorable",
            "text": f"{in_tune*100:.1f}% del tiempo en afinación. "
                    "Presta especial atención a los pasajes marcados en la tabla superior."
        })
    else:
        recs.append({
            "icon": "✅",
            "title": "Afinación — Correcta",
            "text": f"Excelente: {in_tune*100:.1f}% del tiempo dentro de la tolerancia de ±{tol} cents."
        })

    stab = pitch.get("stability_std", 0)
    stab_thr = pitch_cfg.get("stability_threshold", 15)
    if stab > stab_thr:
        recs.append({
            "icon": "〰️",
            "title": "Estabilidad tonal — Vibrato excesivo o fluctuación",
            "text": f"Desviación estándar del pitch: {stab:.1f} cents (umbral: {stab_thr} cents). "
                    "Trabaja el control de la columna de aire y la relajación en la embocadura."
        })

    dev_ms = rhythm.get("mean_deviation_ms", 0)
    thr_ms = rhythm_cfg.get("onset_deviation_ms", 30)
    if dev_ms > thr_ms * 2:
        recs.append({
            "icon": "🥁",
            "title": "Ritmo — Inestabilidad rítmica",
            "text": f"Desviación media de onsets: {dev_ms:.1f} ms (umbral: {thr_ms} ms). "
                    "Practica con metrónomo a tempo reducido antes de incrementar la velocidad."
        })
    elif dev_ms > thr_ms:
        recs.append({
            "icon": "🥁",
            "title": "Ritmo — Pequeñas irregularidades",
            "text": f"Desviación media: {dev_ms:.1f} ms. "
                    "Estás cerca del umbral. Revisa los pasajes rápidos con metrónomo."
        })
    else:
        recs.append({
            "icon": "✅",
            "title": "Ritmo — Preciso",
            "text": f"Desviación media de onsets: {dev_ms:.1f} ms. ¡Buen pulso!"
        })

    dr = dynamics.get("dynamic_range_db", 0)
    min_dr = dyn_cfg.get("min_dynamic_range_db", 10)
    clip = dynamics.get("clipping_ratio", 0)
    if clip > 0.01:
        recs.append({
            "icon": "⚠️",
            "title": "Dinámica — Saturación detectada",
            "text": f"{clip*100:.1f}% de las muestras superan el umbral de clipping. "
                    "Aleja el micrófono o reduce el nivel de grabación en los pasajes fuertes."
        })
    elif dr < min_dr:
        recs.append({
            "icon": "🔉",
            "title": "Dinámica — Rango dinámico bajo",
            "text": f"Rango dinámico: {dr} dB (mínimo deseable: {min_dr} dB). "
                    "Trabaja los contrastes dinámicos: atrévete a tocar más piano y más forte."
        })

    snr = quality.get("snr_db", 0)
    min_snr = qual_cfg.get("min_snr_db", 20)
    if snr < min_snr:
        recs.append({
            "icon": "🎙️",
            "title": "Calidad de grabación — Ruido de fondo",
            "text": f"SNR estimado: {snr} dB (mínimo recomendado: {min_snr} dB). "
                    "Graba en un espacio más silencioso, cierra puertas/ventanas y aleja el móvil de fuentes de ruido."
        })

    sil = quality.get("silence_ratio", 0)
    if sil > 0.3:
        recs.append({
            "icon": "⏸️",
            "title": "Muchos silencios detectados",
            "text": f"{sil*100:.1f}% del tiempo es silencio o muy bajo nivel. "
                    "Comprueba que la grabación es completa y que no hay cortes."
        })

    if not recs:
        recs.append({
            "icon": "🏆",
            "title": "¡Excelente ensayo!",
            "text": "Todas las métricas están dentro de los umbrales óptimos. ¡Sigue así!"
        })

    return recs


# ---------------------------------------------------------------------------
# Recomendaciones por instrumento
# ---------------------------------------------------------------------------

STEM_ADVICE = {
    "drums": {
        "rhythm_bad":  "La batería establece el pulso del grupo. Practica con metrónomo hasta que el tempo sea constante.",
        "rhythm_ok":   "Buen pulso. El grupo puede apoyarse en ti como referencia rítmica.",
    },
    "bass": {
        "rhythm_bad":  "El bajo debe estar perfectamente alineado con la batería. Practica el patrón rítmico junto al baterista antes del próximo ensayo.",
        "rhythm_ok":   "Buen timing. Mantén ese bloqueo con la batería.",
        "pitch_bad":   "Revisa la afinación del instrumento y trabaja los cambios de posición con lentitud.",
        "pitch_ok":    "Afinación correcta.",
        "timing_late": "Tiendes a ir atrasado respecto a la batería. Anticipa mentalmente cada nota antes de tocarla.",
        "timing_early":"Tiendes a adelantarte. Escucha más la batería y deja que ella marque el tempo.",
    },
    "guitar": {
        "pitch_bad":   "Afina antes de cada ensayo y comprueba la afinación cada 15-20 minutos, especialmente en cuerdas agudas.",
        "pitch_ok":    "Afinación correcta.",
        "rhythm_bad":  "Trabaja los cambios de acorde con metrónomo para que no haya pausas entre ellos.",
    },
    "piano": {
        "pitch_bad":   "El piano puede necesitar afinación por un técnico si las notas suenan descentradas.",
        "pitch_ok":    "Afinación correcta.",
        "rhythm_bad":  "Presta atención al tempo en los pasajes rápidos. Practica a velocidad reducida.",
    },
    "vocals": {
        "pitch_bad":   "Calienta la voz antes del ensayo y practica con un drone de referencia para interiorizar la afinación.",
        "pitch_ok":    "Afinación vocal correcta.",
        "rhythm_bad":  "Trabaja la dicción rítmica — cada sílaba debe caer en el tiempo correcto.",
        "stability":   "El vibrato o la fluctuación de pitch es excesiva. Trabaja el apoyo en el diafragma.",
    },
    "other": {
        "rhythm_bad":  "La percusión adicional debe estar sincronizada con la batería. Practica juntos.",
        "rhythm_ok":   "Buen ritmo en los elementos de percusión.",
    },
}


def _build_stem_recommendations(stem_results: dict, stem_summaries: dict, config: dict) -> dict:
    """Genera recomendaciones específicas por instrumento."""
    recs = {}
    rhythm_cfg = config.get("rhythm", {})
    pitch_cfg = config.get("pitch", {})
    thr_ms = rhythm_cfg.get("onset_deviation_ms", 30)
    tol_cents = pitch_cfg.get("tolerance_cents", 20)
    stab_thr = pitch_cfg.get("stability_threshold", 15)

    advice = STEM_ADVICE

    for stem_name, data in stem_results.items():
        if stem_name.startswith("_"):
            continue
        stem_recs = []
        a = advice.get(stem_name, {})

        # Ritmo
        if "rhythm" in data:
            dev = data["rhythm"].get("mean_deviation_ms", 0)
            if dev > thr_ms:
                text = a.get("rhythm_bad", f"Desviación rítmica media: {dev:.0f} ms. Practica con metrónomo.")
                stem_recs.append({"icon": "🥁", "title": "Ritmo", "text": text, "severity": "red" if dev > thr_ms*2 else "yellow"})
            else:
                text = a.get("rhythm_ok", f"Ritmo preciso ({dev:.0f} ms de desviación media).")
                stem_recs.append({"icon": "✅", "title": "Ritmo", "text": text, "severity": "green"})

        # Pitch
        if "pitch" in data:
            in_tune = data["pitch"].get("in_tune_ratio", 1)
            stab = data["pitch"].get("stability_std", 0)
            if in_tune < 0.7:
                text = a.get("pitch_bad", f"Solo {in_tune*100:.0f}% del tiempo en afinación.")
                stem_recs.append({"icon": "🎵", "title": "Afinación", "text": text, "severity": "red"})
            elif in_tune < 0.85:
                stem_recs.append({"icon": "🎵", "title": "Afinación", "text": f"Afinación mejorable: {in_tune*100:.0f}% en tono (±{tol_cents} cents).", "severity": "yellow"})
            else:
                text = a.get("pitch_ok", f"Afinación correcta ({in_tune*100:.0f}% en tono).")
                stem_recs.append({"icon": "✅", "title": "Afinación", "text": text, "severity": "green"})
            if stab > stab_thr and stem_name == "vocals":
                stem_recs.append({"icon": "〰️", "title": "Estabilidad tonal", "text": a.get("stability", f"Fluctuación de pitch: σ={stab:.1f} cents."), "severity": "yellow"})

        # Dinámica: clipping
        if "dynamics" in data:
            clip = data["dynamics"].get("clipping_ratio", 0)
            if clip > 0.01:
                stem_recs.append({"icon": "⚠️", "title": "Saturación", "text": f"Saturación detectada ({clip*100:.1f}% del tiempo). Revisa el nivel de grabación.", "severity": "red"})

        recs[stem_name] = stem_recs

    # Timing bajo vs batería
    tc = stem_results.get("_timing_comparison", {})
    if tc.get("verdict") and tc["verdict"] != "sincronizado":
        mean_off = tc["mean_offset_ms"]
        key = "timing_late" if mean_off > 0 else "timing_early"
        bass_advice = STEM_ADVICE.get("bass", {}).get(key, tc["verdict"])
        if "bass" in recs:
            recs["bass"].insert(0, {
                "icon": "🎸",
                "title": f"Timing vs Batería — {tc['verdict']}",
                "text": bass_advice,
                "severity": "red" if abs(mean_off) > 50 else "yellow",
            })

    return recs
