"""Tests for vibration feature extraction."""

import numpy as np
import pytest

from src.feature_extraction import (
    compute_rms,
    compute_spectral_energy,
    extract_window_features,
    extract_features,
    BearingOEMParams,
)


@pytest.fixture
def skf_6205_params():
    return BearingOEMParams(
        model="SKF 6205-2RS JEM",
        bore_mm=25.0,
        dynamic_load_rating_kn=14.8,
        static_load_rating_kn=7.8,
        life_exponent=3.0,
        bpfi=5.4152,
        bpfo=3.5848,
        ftf=0.39828,
        bsf=4.7135,
        max_speed_rpm=18000.0,
    )


class TestRMS:
    def test_known_value(self):
        """RMS of [1, -1, 1, -1] = 1.0."""
        x = np.array([1.0, -1.0, 1.0, -1.0])
        assert compute_rms(x) == pytest.approx(1.0)

    def test_sine_wave(self):
        """RMS of a sine wave is amplitude / sqrt(2)."""
        t = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
        amplitude = 3.0
        x = amplitude * np.sin(t)
        expected = amplitude / np.sqrt(2)
        assert compute_rms(x) == pytest.approx(expected, rel=1e-3)

    def test_zero_signal(self):
        x = np.zeros(100)
        assert compute_rms(x) == pytest.approx(0.0)


class TestKurtosis:
    def test_gaussian_kurtosis(self):
        """Kurtosis of a Gaussian signal should be approximately 3.0."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50000)
        features = extract_window_features(x, sr=12000.0)
        # Raw kurtosis (not excess) for Gaussian is 3.0
        assert features["kurtosis"] == pytest.approx(3.0, abs=0.1)

    def test_impulsive_signal_high_kurtosis(self):
        """A signal with rare spikes should have kurtosis >> 3.0."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 0.1, 10000)
        # Add sparse impulses
        spike_indices = rng.choice(10000, size=20, replace=False)
        x[spike_indices] = 10.0
        features = extract_window_features(x, sr=12000.0)
        assert features["kurtosis"] > 5.0


class TestSpectralEnergy:
    def test_known_frequency(self):
        """Energy at a known tone frequency should be high."""
        sr = 12000.0
        n = 2400
        freq = 100.0  # Hz
        t = np.arange(n) / sr
        x = np.sin(2 * np.pi * freq * t)

        energy = compute_spectral_energy(x, sr, center_freq=freq, bandwidth=5.0)
        noise_energy = compute_spectral_energy(x, sr, center_freq=500.0, bandwidth=5.0)

        assert energy > noise_energy * 10

    def test_defect_frequency_extraction(self, skf_6205_params):
        """Features should include defect frequency energies when params provided."""
        sr = 12000.0
        n = 2400
        rpm = 1797.0
        shaft_freq = rpm / 60.0

        # Create signal with energy at BPFO frequency
        bpfo_hz = skf_6205_params.bpfo * shaft_freq
        t = np.arange(n) / sr
        x = 0.1 * np.random.randn(n) + np.sin(2 * np.pi * bpfo_hz * t)

        features = extract_window_features(x, sr, skf_6205_params, rpm)

        assert "bpfo_energy" in features
        assert "bpfi_energy" in features
        assert "bsf_energy" in features
        assert "ftf_energy" in features
        assert features["bpfo_energy"] > features["bpfi_energy"]


class TestExtractFeatures:
    def test_output_shape(self):
        """Should produce correct number of windows."""
        sr = 12000.0
        signal = np.random.randn(24000)  # 2 seconds
        window_size = 2400
        hop_size = 1200

        df = extract_features(signal, sr, window_size, hop_size)

        expected_windows = (len(signal) - window_size) // hop_size + 1
        assert len(df) == expected_windows
        assert "rms" in df.columns
        assert "kurtosis" in df.columns
        assert "time_seconds" in df.columns

    def test_time_increases(self):
        """Time column should be monotonically increasing."""
        signal = np.random.randn(24000)
        df = extract_features(signal, sr=12000.0)
        assert all(df["time_seconds"].diff().dropna() > 0)
