"""
Tests básicos para los módulos de análisis.
Usa señales sintéticas para no depender de archivos de audio reales.
"""

import numpy as np
import pytest


SR = 22050
DURATION = 5  # segundos


def _sine_wave(freq_hz: float = 440.0, sr: int = SR, duration: float = DURATION) -> np.ndarray:
    """Genera un sinusoide puro normalizado."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    return y


def _noise(sr: int = SR, duration: float = DURATION) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (0.01 * rng.standard_normal(int(sr * duration))).astype(np.float32)


CONFIG = {
    "pitch": {"tolerance_cents": 20, "stability_threshold": 15},
    "rhythm": {"onset_deviation_ms": 30, "tempo_variance_bpm": 5},
    "dynamics": {"min_dynamic_range_db": 10, "clipping_threshold_db": -1},
    "quality": {"min_snr_db": 20},
}


# ---------------------------------------------------------------------------
# Rhythm
# ---------------------------------------------------------------------------

class TestRhythm:
    def test_returns_required_keys(self):
        from analysis.rhythm import analyze
        y = _sine_wave(440)
        result = analyze(y, SR, CONFIG["rhythm"])
        for key in ("tempo_global", "tempo_curve", "onset_times", "onset_deviations_ms",
                    "mean_deviation_ms", "ioi_stats", "onset_strength"):
            assert key in result, f"Clave faltante: {key}"

    def test_tempo_positive(self):
        from analysis.rhythm import analyze
        y = _sine_wave(440)
        result = analyze(y, SR, CONFIG["rhythm"])
        assert result["tempo_global"] > 0

    def test_ioi_stats_structure(self):
        from analysis.rhythm import analyze
        y = _sine_wave(440)
        result = analyze(y, SR, CONFIG["rhythm"])
        ioi = result["ioi_stats"]
        assert "mean_ms" in ioi and "std_ms" in ioi and "cv" in ioi


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

class TestDynamics:
    def test_returns_required_keys(self):
        from analysis.dynamics import analyze
        y = _sine_wave(440)
        result = analyze(y, SR, CONFIG["dynamics"])
        for key in ("rms_times", "rms_db", "dynamic_range_db", "clipping_ratio",
                    "spectrogram"):
            assert key in result, f"Clave faltante: {key}"

    def test_no_clipping_on_clean_sine(self):
        from analysis.dynamics import analyze
        y = _sine_wave(440)
        result = analyze(y, SR, CONFIG["dynamics"])
        assert result["clipping_ratio"] == 0.0

    def test_dynamic_range_positive(self):
        from analysis.dynamics import analyze
        y = _sine_wave(440)
        result = analyze(y, SR, CONFIG["dynamics"])
        assert result["dynamic_range_db"] >= 0


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

class TestQuality:
    def test_returns_required_keys(self):
        from analysis.quality import analyze
        y = _sine_wave(440)
        result = analyze(y, SR, CONFIG["quality"])
        for key in ("snr_db", "spectral_centroid_mean", "zcr_mean",
                    "silence_ratio", "mel_spectrogram", "mfcc_mean"):
            assert key in result, f"Clave faltante: {key}"

    def test_snr_higher_for_clean_signal(self):
        from analysis.quality import analyze
        y_clean = _sine_wave(440)
        y_noise = _noise()
        r_clean = analyze(y_clean, SR, CONFIG["quality"])
        r_noise = analyze(y_noise, SR, CONFIG["quality"])
        assert r_clean["snr_db"] > r_noise["snr_db"]

    def test_mfcc_has_13_coefficients(self):
        from analysis.quality import analyze
        y = _sine_wave(440)
        result = analyze(y, SR, CONFIG["quality"])
        assert len(result["mfcc_mean"]) == 13


# ---------------------------------------------------------------------------
# Pitch (pYIN fallback — no requiere CREPE)
# ---------------------------------------------------------------------------

class TestPitchFallback:
    def test_fallback_returns_required_keys(self):
        from analysis.pitch import _fallback_pitch
        y = _sine_wave(440)
        result = _fallback_pitch(y, SR, CONFIG["pitch"])
        for key in ("times", "frequencies", "cents_deviation",
                    "stability_std", "in_tune_ratio", "problematic_segments"):
            assert key in result, f"Clave faltante: {key}"

    def test_in_tune_ratio_is_fraction(self):
        from analysis.pitch import _fallback_pitch
        y = _sine_wave(440)
        result = _fallback_pitch(y, SR, CONFIG["pitch"])
        assert 0.0 <= result["in_tune_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class TestIngest:
    def test_missing_file_raises(self):
        from ingest import load
        with pytest.raises(FileNotFoundError):
            load("/tmp/nonexistent_audio_file.wav")

    def test_unsupported_format_raises(self):
        import tempfile, os
        from ingest import load
        tmp = tempfile.NamedTemporaryFile(suffix=".xyz", delete=False)
        tmp.close()
        try:
            with pytest.raises(ValueError, match="Formato no soportado"):
                load(tmp.name)
        finally:
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Report generator (smoke test — sin análisis real)
# ---------------------------------------------------------------------------

class TestReportGenerator:
    def test_generates_html_file(self, tmp_path):
        from report.generator import generate
        dummy_pitch = {
            "times": [0.0, 1.0], "frequencies": [440.0, 440.0],
            "confidence": [0.9, 0.9], "cents_deviation": [0.0, 2.0],
            "stability_std": 3.0, "in_tune_ratio": 0.95,
            "problematic_segments": [], "status": "ok",
        }
        dummy_rhythm = {
            "tempo_global": 120.0,
            "tempo_curve": {"times": [0.0, 1.0], "bpm": [120.0, 121.0]},
            "beat_times": [0.5, 1.0],
            "onset_times": [0.5, 1.0],
            "onset_deviations_ms": [2.0, -1.5],
            "mean_deviation_ms": 1.75,
            "ioi_stats": {"mean_ms": 500.0, "std_ms": 5.0, "cv": 0.01},
            "onset_strength": {"times": [0.0, 1.0], "strength": [0.5, 0.8]},
            "onset_dev_threshold_ms": 30,
        }
        dummy_dynamics = {
            "rms_times": [0.0, 1.0], "rms_db": [-10.0, -12.0],
            "dynamic_range_db": 15.0, "clipping_ratio": 0.0,
            "lufs": -18.0,
            "spectrogram": {"times": [0.0], "freqs": [100.0], "magnitude_db": [[-20.0]]},
            "min_dynamic_range_db": 10,
        }
        dummy_quality = {
            "snr_db": 25.0, "spectral_centroid_mean": 1200.0,
            "spectral_rolloff_mean": 3000.0, "zcr_mean": 0.05,
            "silence_ratio": 0.05,
            "mel_spectrogram": {"times": [0.0], "freqs_mel": [100.0], "magnitude_db": [[-20.0]]},
            "mfcc_mean": [float(i) for i in range(13)],
            "min_snr_db": 20,
        }
        dummy_meta = {
            "filename": "test.wav", "original_sr": 22050,
            "original_channels": 1, "duration_s": 5.0, "format": ".wav",
        }
        dummy_config = {
            "pitch": {"tolerance_cents": 20, "stability_threshold": 15},
            "rhythm": {"onset_deviation_ms": 30},
            "dynamics": {"min_dynamic_range_db": 10},
            "quality": {"min_snr_db": 20},
        }
        out = generate(
            meta=dummy_meta,
            pitch=dummy_pitch,
            rhythm=dummy_rhythm,
            dynamics=dummy_dynamics,
            quality=dummy_quality,
            config=dummy_config,
            output_dir=str(tmp_path),
        )
        assert out.endswith(".html")
        content = open(out).read()
        assert "Music Coach" in content
        assert "Plotly" in content
