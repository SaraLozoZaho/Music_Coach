#!/usr/bin/env python3
"""
Music Coach — CLI entry point.

Usage:
    python analyze.py recording.mp3
    python analyze.py recording.mp3 --publish
    python analyze.py recording.mp3 --separate          # per-instrument analysis
    python analyze.py recording.mp3 --separate --publish
    python analyze.py recording.mp3 --output-dir my_folder/
    python analyze.py recording.mp3 --config my_config.yaml
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
        print(f"[warn] config.yaml not found at '{path}', using default values.")
        return {}


def _step(label: str):
    print(f"  → {label}...", end=" ", flush=True)
    return time.time()


def _done(t0: float):
    print(f"done ({time.time() - t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Music Coach — Analyses a rehearsal recording and generates an HTML report."
    )
    parser.add_argument("audio_file", help="Path to audio file (.mp3, .wav, .m4a, .ogg, .flac, .aac)")
    parser.add_argument("--publish", action="store_true", help="Publish to GitHub Pages after analysis")
    parser.add_argument("--separate", action="store_true", help="Separate instruments with Demucs and analyse per track")
    parser.add_argument("--output-dir", default=None, help="Output folder for the report (default: docs/reports/)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config file")
    parser.add_argument("--no-pitch", action="store_true", help="Skip global pitch analysis (faster)")
    args = parser.parse_args()

    print("\n🎵 Music Coach\n")

    # Config
    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg.get("report", {}).get("output_dir", "docs/reports")

    # 1. Load audio
    t = _step("Loading audio")
    try:
        y, sr, meta = ingest.load(args.audio_file)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n[error] {e}")
        sys.exit(1)
    _done(t)
    print(f"     File: {meta['filename']}  |  Duration: {meta['duration_s']}s  |  SR: {meta['original_sr']} Hz")

    # 2. Global pitch analysis
    if args.no_pitch:
        pitch_results = {
            "times": [], "frequencies": [], "confidence": [],
            "cents_deviation": [], "stability_std": 0.0,
            "in_tune_ratio": 0.0, "problematic_segments": [], "status": "skipped"
        }
    else:
        t = _step("Global pitch analysis (CREPE / pYIN)")
        pitch_results = pitch_mod.analyze(y, sr, cfg.get("pitch", {}))
        _done(t)
        if pitch_results.get("status") == "fallback_pyin":
            print("     [info] CREPE not installed — using pYIN as fallback")

    # 3. Global rhythm analysis
    t = _step("Global rhythm analysis")
    rhythm_results = rhythm_mod.analyze(y, sr, cfg.get("rhythm", {}))
    _done(t)
    print(f"     Tempo: {rhythm_results['tempo_global']:.1f} BPM  |  Mean deviation: {rhythm_results['mean_deviation_ms']:.1f} ms")

    # 4. Global dynamics analysis
    t = _step("Global dynamics analysis")
    dynamics_results = dynamics_mod.analyze(y, sr, cfg.get("dynamics", {}))
    _done(t)
    lufs_str = f"  |  LUFS: {dynamics_results['lufs']}" if dynamics_results.get("lufs") else ""
    print(f"     Dynamic range: {dynamics_results['dynamic_range_db']} dB{lufs_str}")

    # 5. Quality analysis
    t = _step("Quality analysis")
    quality_results = quality_mod.analyze(y, sr, cfg.get("quality", {}))
    _done(t)
    print(f"     Estimated SNR: {quality_results['snr_db']} dB")

    # 6. Source separation + per-instrument analysis (optional)
    stem_results = None
    stem_summaries = None
    if args.separate:
        print()
        print("  ── Per-instrument analysis ───────────────────────────────")
        try:
            from analysis.separation import separate
            from analysis.per_stem import analyze_all, build_stem_summaries

            t = _step("Separating sources with Demucs (htdemucs_6s)")
            stems = separate(args.audio_file)
            _done(t)
            print(f"     Separated stems: {', '.join(stems.keys())}")

            t = _step("Analysing each instrument")
            stem_results = analyze_all(stems, cfg)
            _done(t)

            stem_summaries = build_stem_summaries(stem_results, cfg)
            print()
            for stem_name, summary in stem_summaries.items():
                if stem_name.startswith("_"):
                    continue
                icon = {"green": "✅", "yellow": "⚠️", "red": "❌"}[summary["color"]]
                print(f"     {icon}  {summary['instrument']}: {summary['label']}")

            # Bass vs drums timing
            tc = stem_summaries.get("_timing_comparison")
            if tc and tc.get("verdict"):
                print(f"\n     🎸 Bass: {tc['verdict']}  (σ={tc['std_offset_ms']:.0f} ms)")

        except ImportError:
            print("\n  [warn] Demucs not installed. Install with: pip install demucs")
            print("  Continuing without source separation...\n")
        except RuntimeError as e:
            print(f"\n  [error in separation] {e}\n")
        print()

    # 7. Generate report
    t = _step("Generating HTML report")
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
    print(f"\n✅ Report generated: {report_path}\n")

    # 8. Publish (optional)
    if args.publish:
        from deploy import github_pages
        print("  → Publishing to GitHub Pages...")
        try:
            url = github_pages.deploy(report_path)
            print(f"\n🌐 Report published: {url}\n")
        except RuntimeError as e:
            print(f"\n[publish error] {e}\n")
            sys.exit(1)
    else:
        print("  Open the report in your browser:")
        print(f"  file://{report_path}\n")
        print("  To publish to GitHub Pages, add --publish\n")


if __name__ == "__main__":
    main()
