"""
Interactive HTML report generator using Jinja2 + Plotly.
The resulting HTML is self-contained and compatible with GitHub Pages.
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
    Generates the HTML report and saves it to output_dir.
    Returns the absolute path of the generated file.
    """
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.html"
    output_path = os.path.join(output_dir, filename)

    summary = _build_summary(pitch, rhythm, dynamics, quality, config)
    recommendations = _build_recommendations(pitch, rhythm, dynamics, quality, config)

    # Per-instrument recommendations (if source separation was used)
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
# Traffic light summary
# ---------------------------------------------------------------------------

def _build_summary(pitch, rhythm, dynamics, quality, config) -> dict:
    pitch_cfg = config.get("pitch", {})
    rhythm_cfg = config.get("rhythm", {})
    dyn_cfg = config.get("dynamics", {})
    qual_cfg = config.get("quality", {})

    # Tuning: % time in tune
    in_tune = pitch.get("in_tune_ratio", 0)
    pitch_color = "green" if in_tune >= 0.8 else ("yellow" if in_tune >= 0.6 else "red")
    pitch_label = {"green": "Good", "yellow": "Needs work", "red": "Needs attention"}[pitch_color]

    # Rhythm: mean onset deviation
    dev_ms = rhythm.get("mean_deviation_ms", 999)
    thr = rhythm_cfg.get("onset_deviation_ms", 30)
    rhythm_color = "green" if dev_ms <= thr else ("yellow" if dev_ms <= thr * 2 else "red")
    rhythm_label = {"green": "Precise", "yellow": "Needs work", "red": "Unstable"}[rhythm_color]

    # Dynamics: dynamic range
    dr = dynamics.get("dynamic_range_db", 0)
    min_dr = dyn_cfg.get("min_dynamic_range_db", 10)
    clip = dynamics.get("clipping_ratio", 0)
    if clip > 0.01:
        dyn_color = "red"
    elif dr >= min_dr:
        dyn_color = "green"
    else:
        dyn_color = "yellow"
    dyn_label = {"green": "Good", "yellow": "Low range", "red": "Clipping"}[dyn_color]

    # Quality: SNR
    snr = quality.get("snr_db", 0)
    min_snr = qual_cfg.get("min_snr_db", 20)
    qual_color = "green" if snr >= min_snr else ("yellow" if snr >= min_snr * 0.7 else "red")
    qual_label = {"green": "Good", "yellow": "Acceptable", "red": "Too much noise"}[qual_color]

    return {
        "pitch": {"color": pitch_color, "label": pitch_label},
        "rhythm": {"color": rhythm_color, "label": rhythm_label},
        "dynamics": {"color": dyn_color, "label": dyn_label},
        "quality": {"color": qual_color, "label": qual_label},
    }


# ---------------------------------------------------------------------------
# Global text recommendations
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
            "title": "Tuning — Urgent attention needed",
            "text": f"Only {in_tune*100:.1f}% of the time within ±{tol} cents. "
                    "Practise with a tuning reference (click track or drone) to internalise pitch."
        })
    elif in_tune < 0.8:
        recs.append({
            "icon": "🎵",
            "title": "Tuning — Needs improvement",
            "text": f"{in_tune*100:.1f}% of the time in tune. "
                    "Pay special attention to the passages marked in the table above."
        })
    else:
        recs.append({
            "icon": "✅",
            "title": "Tuning — Good",
            "text": f"Excellent: {in_tune*100:.1f}% of the time within ±{tol} cents tolerance."
        })

    stab = pitch.get("stability_std", 0)
    stab_thr = pitch_cfg.get("stability_threshold", 15)
    if stab > stab_thr:
        recs.append({
            "icon": "〰️",
            "title": "Tonal stability — Excessive vibrato or pitch fluctuation",
            "text": f"Pitch standard deviation: {stab:.1f} cents (threshold: {stab_thr} cents). "
                    "Work on breath support and embouchure relaxation."
        })

    dev_ms = rhythm.get("mean_deviation_ms", 0)
    thr_ms = rhythm_cfg.get("onset_deviation_ms", 30)
    if dev_ms > thr_ms * 2:
        recs.append({
            "icon": "🥁",
            "title": "Rhythm — Rhythmic instability",
            "text": f"Mean onset deviation: {dev_ms:.1f} ms (threshold: {thr_ms} ms). "
                    "Practise with a metronome at a reduced tempo before increasing speed."
        })
    elif dev_ms > thr_ms:
        recs.append({
            "icon": "🥁",
            "title": "Rhythm — Minor irregularities",
            "text": f"Mean deviation: {dev_ms:.1f} ms. "
                    "Close to the threshold. Review fast passages with a metronome."
        })
    else:
        recs.append({
            "icon": "✅",
            "title": "Rhythm — Precise",
            "text": f"Mean onset deviation: {dev_ms:.1f} ms. Great pulse!"
        })

    dr = dynamics.get("dynamic_range_db", 0)
    min_dr = dyn_cfg.get("min_dynamic_range_db", 10)
    clip = dynamics.get("clipping_ratio", 0)
    if clip > 0.01:
        recs.append({
            "icon": "⚠️",
            "title": "Dynamics — Clipping detected",
            "text": f"{clip*100:.1f}% of samples exceed the clipping threshold. "
                    "Move the microphone further away or reduce the recording level in loud passages."
        })
    elif dr < min_dr:
        recs.append({
            "icon": "🔉",
            "title": "Dynamics — Low dynamic range",
            "text": f"Dynamic range: {dr} dB (minimum desired: {min_dr} dB). "
                    "Work on dynamic contrast: dare to play softer and louder."
        })

    snr = quality.get("snr_db", 0)
    min_snr = qual_cfg.get("min_snr_db", 20)
    if snr < min_snr:
        recs.append({
            "icon": "🎙️",
            "title": "Recording quality — Background noise",
            "text": f"Estimated SNR: {snr} dB (recommended minimum: {min_snr} dB). "
                    "Record in a quieter space, close doors/windows and keep the phone away from noise sources."
        })

    sil = quality.get("silence_ratio", 0)
    if sil > 0.3:
        recs.append({
            "icon": "⏸️",
            "title": "Many silences detected",
            "text": f"{sil*100:.1f}% of the time is silence or very low level. "
                    "Check that the recording is complete and there are no cuts."
        })

    if not recs:
        recs.append({
            "icon": "🏆",
            "title": "Excellent rehearsal!",
            "text": "All metrics are within optimal thresholds. Keep it up!"
        })

    return recs


# ---------------------------------------------------------------------------
# Per-instrument recommendations
# ---------------------------------------------------------------------------

STEM_ADVICE = {
    "drums": {
        "rhythm_bad":  "The drums set the pulse for the whole band. Practise with a metronome until the tempo is steady.",
        "rhythm_ok":   "Great pulse. The band can use you as the rhythmic reference.",
    },
    "bass": {
        "rhythm_bad":  "The bass must be perfectly locked with the drums. Practise the rhythm pattern together with the drummer before the next rehearsal.",
        "rhythm_ok":   "Good timing. Keep that lock with the drums.",
        "pitch_bad":   "Check the instrument's tuning and work through position changes slowly.",
        "pitch_ok":    "Tuning is correct.",
        "timing_late": "You tend to play behind the drums. Mentally anticipate each note before you play it.",
        "timing_early":"You tend to rush ahead. Listen more to the drums and let them set the tempo.",
    },
    "guitar": {
        "pitch_bad":   "Tune before each rehearsal and check tuning every 15-20 minutes, especially on the higher strings.",
        "pitch_ok":    "Tuning is correct.",
        "rhythm_bad":  "Work on chord changes with a metronome so there are no gaps between them.",
    },
    "piano": {
        "pitch_bad":   "The piano may need tuning by a technician if notes sound off-centre.",
        "pitch_ok":    "Tuning is correct.",
        "rhythm_bad":  "Pay attention to tempo in fast passages. Practise at reduced speed.",
    },
    "vocals": {
        "pitch_bad":   "Warm up your voice before rehearsal and practise with a drone reference to internalise pitch.",
        "pitch_ok":    "Vocal tuning is correct.",
        "rhythm_bad":  "Work on rhythmic diction — each syllable must fall on the right beat.",
        "stability":   "Vibrato or pitch fluctuation is excessive. Work on diaphragm support.",
    },
    "other": {
        "rhythm_bad":  "Additional percussion must be in sync with the drums. Practise together.",
        "rhythm_ok":   "Good rhythm on the percussion elements.",
    },
}


def _build_stem_recommendations(stem_results: dict, stem_summaries: dict, config: dict) -> dict:
    """Generates specific recommendations per instrument."""
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

        # Rhythm
        if "rhythm" in data:
            dev = data["rhythm"].get("mean_deviation_ms", 0)
            if dev > thr_ms:
                text = a.get("rhythm_bad", f"Mean rhythmic deviation: {dev:.0f} ms. Practise with a metronome.")
                stem_recs.append({"icon": "🥁", "title": "Rhythm", "text": text, "severity": "red" if dev > thr_ms*2 else "yellow"})
            else:
                text = a.get("rhythm_ok", f"Precise rhythm ({dev:.0f} ms mean deviation).")
                stem_recs.append({"icon": "✅", "title": "Rhythm", "text": text, "severity": "green"})

        # Pitch
        if "pitch" in data:
            in_tune = data["pitch"].get("in_tune_ratio", 1)
            stab = data["pitch"].get("stability_std", 0)
            if in_tune < 0.7:
                text = a.get("pitch_bad", f"Only {in_tune*100:.0f}% of the time in tune.")
                stem_recs.append({"icon": "🎵", "title": "Tuning", "text": text, "severity": "red"})
            elif in_tune < 0.85:
                stem_recs.append({"icon": "🎵", "title": "Tuning", "text": f"Tuning needs work: {in_tune*100:.0f}% in tune (±{tol_cents} cents).", "severity": "yellow"})
            else:
                text = a.get("pitch_ok", f"Tuning is correct ({in_tune*100:.0f}% in tune).")
                stem_recs.append({"icon": "✅", "title": "Tuning", "text": text, "severity": "green"})
            if stab > stab_thr and stem_name == "vocals":
                stem_recs.append({"icon": "〰️", "title": "Tonal stability", "text": a.get("stability", f"Pitch fluctuation: σ={stab:.1f} cents."), "severity": "yellow"})

        # Dynamics: clipping
        if "dynamics" in data:
            clip = data["dynamics"].get("clipping_ratio", 0)
            if clip > 0.01:
                stem_recs.append({"icon": "⚠️", "title": "Clipping", "text": f"Clipping detected ({clip*100:.1f}% of the time). Check the recording level.", "severity": "red"})

        recs[stem_name] = stem_recs

    # Bass vs drums timing
    tc = stem_results.get("_timing_comparison", {})
    if tc.get("verdict") and tc["verdict"] != "in sync":
        mean_off = tc["mean_offset_ms"]
        key = "timing_late" if mean_off > 0 else "timing_early"
        bass_advice = STEM_ADVICE.get("bass", {}).get(key, tc["verdict"])
        if "bass" in recs:
            recs["bass"].insert(0, {
                "icon": "🎸",
                "title": f"Timing vs Drums — {tc['verdict']}",
                "text": bass_advice,
                "severity": "red" if abs(mean_off) > 50 else "yellow",
            })

    return recs
