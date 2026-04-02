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

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,
    )
    # Filtro tojson para pasar datos a JavaScript de forma segura
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
