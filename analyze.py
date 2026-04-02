#!/usr/bin/env python3
"""
Music Coach — Punto de entrada CLI.

Uso:
    python analyze.py grabacion.mp3
    python analyze.py grabacion.mp3 --publish
    python analyze.py grabacion.mp3 --separate          # análisis por instrumento
    python analyze.py grabacion.mp3 --separate --publish
    python analyze.py grabacion.mp3 --output-dir mi_carpeta/
    python analyze.py grabacion.mp3 --config mi_config.yaml
"""

import argparse
import sys
import time
import yaml

import ingest
from analysis import pitch as pitch_mod
from analysis import rhythm as rhythm_mod
from analysis import dynamics as dynamics_mod
from analysis import quality as quality_mod
from report import generator as report_gen


DEFAULT_CONFIG = "config.yaml"


def load_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[warn] config.yaml no encontrado en '{path}', usando valores por defecto.")
        return {}


def _step(label: str):
    print(f"  → {label}...", end=" ", flush=True)
    return time.time()


def _done(t0: float):
    print(f"listo ({time.time() - t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Music Coach — Analiza una grabación de ensayo y genera un informe HTML."
    )
    parser.add_argument("audio_file", help="Ruta al archivo de audio (.mp3, .wav, .m4a, .ogg, .flac, .aac)")
    parser.add_argument("--publish", action="store_true", help="Publicar en GitHub Pages tras el análisis")
    parser.add_argument("--separate", action="store_true", help="Separar instrumentos con Demucs y analizar por pista")
    parser.add_argument("--output-dir", default=None, help="Carpeta de salida para el informe (por defecto: docs/reports/)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Ruta al archivo de configuración YAML")
    parser.add_argument("--no-pitch", action="store_true", help="Omitir análisis de pitch global (más rápido)")
    args = parser.parse_args()

    print("\n🎵 Music Coach\n")

    # Configuración
    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg.get("report", {}).get("output_dir", "docs/reports")

    # 1. Ingesta
    t = _step("Cargando audio")
    try:
        y, sr, meta = ingest.load(args.audio_file)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n[error] {e}")
        sys.exit(1)
    _done(t)
    print(f"     Archivo: {meta['filename']}  |  Duración: {meta['duration_s']}s  |  SR: {meta['original_sr']} Hz")

    # 2. Análisis de pitch global
    if args.no_pitch:
        pitch_results = {
            "times": [], "frequencies": [], "confidence": [],
            "cents_deviation": [], "stability_std": 0.0,
            "in_tune_ratio": 0.0, "problematic_segments": [], "status": "skipped"
        }
    else:
        t = _step("Análisis de afinación global (CREPE / pYIN)")
        pitch_results = pitch_mod.analyze(y, sr, cfg.get("pitch", {}))
        _done(t)
        if pitch_results.get("status") == "fallback_pyin":
            print("     [info] CREPE no instalado — usando pYIN como fallback")

    # 3. Análisis rítmico global
    t = _step("Análisis rítmico global")
    rhythm_results = rhythm_mod.analyze(y, sr, cfg.get("rhythm", {}))
    _done(t)
    print(f"     Tempo: {rhythm_results['tempo_global']:.1f} BPM  |  Desv. media: {rhythm_results['mean_deviation_ms']:.1f} ms")

    # 4. Análisis de dinámica global
    t = _step("Análisis de dinámica global")
    dynamics_results = dynamics_mod.analyze(y, sr, cfg.get("dynamics", {}))
    _done(t)
    lufs_str = f"  |  LUFS: {dynamics_results['lufs']}" if dynamics_results.get("lufs") else ""
    print(f"     Rango dinámico: {dynamics_results['dynamic_range_db']} dB{lufs_str}")

    # 5. Análisis de calidad
    t = _step("Análisis de calidad")
    quality_results = quality_mod.analyze(y, sr, cfg.get("quality", {}))
    _done(t)
    print(f"     SNR estimado: {quality_results['snr_db']} dB")

    # 6. Separación de fuentes + análisis por instrumento (opcional)
    stem_results = None
    stem_summaries = None
    if args.separate:
        print()
        print("  ── Análisis por instrumento ──────────────────────────────")
        try:
            from analysis.separation import separate
            from analysis.per_stem import analyze_all, build_stem_summaries

            t = _step("Separando fuentes con Demucs (htdemucs_6s)")
            stems = separate(args.audio_file)
            _done(t)
            print(f"     Stems separados: {', '.join(stems.keys())}")

            t = _step("Analizando cada instrumento")
            stem_results = analyze_all(stems, cfg)
            _done(t)

            stem_summaries = build_stem_summaries(stem_results, cfg)
            print()
            for stem_name, summary in stem_summaries.items():
                if stem_name.startswith("_"):
                    continue
                icon = {"green": "✅", "yellow": "⚠️", "red": "❌"}[summary["color"]]
                print(f"     {icon}  {summary['instrument']}: {summary['label']}")

            # Timing bajo vs batería
            tc = stem_summaries.get("_timing_comparison")
            if tc and tc.get("verdict"):
                print(f"\n     🎸 Bajo: {tc['verdict']}  (σ={tc['std_offset_ms']:.0f} ms)")

        except ImportError:
            print("\n  [warn] Demucs no instalado. Instala con: pip install demucs")
            print("  Continuando sin separación de fuentes...\n")
        except RuntimeError as e:
            print(f"\n  [error en separación] {e}\n")
        print()

    # 7. Generación del informe
    t = _step("Generando informe HTML")
    report_path = report_gen.generate(
        meta=meta,
        pitch=pitch_results,
        rhythm=rhythm_results,
        dynamics=dynamics_results,
        quality=quality_results,
        config=cfg,
        output_dir=output_dir,
        stem_results=stem_results,
        stem_summaries=stem_summaries,
    )
    _done(t)
    print(f"\n✅ Informe generado: {report_path}\n")

    # 8. Publicación (opcional)
    if args.publish:
        from deploy import github_pages
        print("  → Publicando en GitHub Pages...")
        try:
            url = github_pages.deploy(report_path)
            print(f"\n🌐 Informe publicado: {url}\n")
        except RuntimeError as e:
            print(f"\n[error al publicar] {e}\n")
            sys.exit(1)
    else:
        print("  Abre el informe en tu navegador:")
        print(f"  file://{report_path}\n")
        print("  Para publicar en GitHub Pages, añade --publish\n")


if __name__ == "__main__":
    main()
