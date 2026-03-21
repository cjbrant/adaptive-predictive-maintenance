"""Extract degradation features from raw vibration signals.

Computes time-domain and frequency-domain features from windowed vibration
data. When bearing defect frequencies are provided (via BearingOEMParams),
also computes spectral energy at each characteristic fault frequency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BearingOEMParams:
    """OEM bearing specifications extracted from RAG pipeline."""

    model: str
    bore_mm: float
    dynamic_load_rating_kn: float
    static_load_rating_kn: float
    life_exponent: float
    bpfi: float
    bpfo: float
    ftf: float
    bsf: float
    max_speed_rpm: float


def compute_rms(x: np.ndarray) -> float:
    """Root mean square of signal."""
    return float(np.sqrt(np.mean(x**2)))


def compute_spectral_energy(
    x: np.ndarray,
    sr: float,
    center_freq: float,
    bandwidth: float = 10.0,
) -> float:
    """
    Compute spectral energy in a frequency band centered on center_freq.

    Parameters
    ----------
    x : signal window
    sr : sampling rate in Hz
    center_freq : center of the band in Hz
    bandwidth : half-width of the band in Hz
    """
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    fft_mag = np.abs(np.fft.rfft(x)) / n

    mask = (freqs >= center_freq - bandwidth) & (freqs <= center_freq + bandwidth)
    if not np.any(mask):
        return 0.0
    return float(np.sum(fft_mag[mask] ** 2))


def extract_window_features(
    window: np.ndarray,
    sr: float,
    bearing_params: BearingOEMParams | None = None,
    rpm: float = 1797.0,
) -> dict:
    """Extract all features from a single signal window."""
    rms = compute_rms(window)
    peak = float(np.max(np.abs(window)))
    crest_factor = peak / rms if rms > 0 else 0.0
    kurt = float(kurtosis(window, fisher=False))  # Excess=False -> raw kurtosis
    skewness = float(skew(window))
    peak_to_peak = float(np.max(window) - np.min(window))

    features = {
        "rms": rms,
        "peak": peak,
        "crest_factor": crest_factor,
        "kurtosis": kurt,
        "skewness": skewness,
        "peak_to_peak": peak_to_peak,
    }

    # Frequency-domain features using defect frequencies
    if bearing_params is not None:
        shaft_freq = rpm / 60.0  # Hz
        bw = shaft_freq * 0.5  # bandwidth around each defect freq

        for name, mult in [
            ("bpfi_energy", bearing_params.bpfi),
            ("bpfo_energy", bearing_params.bpfo),
            ("bsf_energy", bearing_params.bsf),
            ("ftf_energy", bearing_params.ftf),
        ]:
            center = mult * shaft_freq
            features[name] = compute_spectral_energy(window, sr, center, bw)

        # Broadband energy: total spectral energy minus defect bands
        n = len(window)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        fft_mag = np.abs(np.fft.rfft(window)) / n
        total_energy = float(np.sum(fft_mag**2))

        defect_energy = sum(
            features[k] for k in ["bpfi_energy", "bpfo_energy", "bsf_energy", "ftf_energy"]
        )
        features["broadband_energy"] = total_energy - defect_energy

    return features


def extract_features(
    signal_data: np.ndarray,
    sr: float,
    window_size: int = 2400,
    hop_size: int = 1200,
    bearing_params: BearingOEMParams | None = None,
    rpm: float = 1797.0,
) -> pd.DataFrame:
    """
    Extract degradation-relevant features from raw vibration signal.

    Slides a window across the signal and computes time-domain and
    frequency-domain features for each window.

    Parameters
    ----------
    signal_data : raw vibration signal (1D array)
    sr : sampling rate in Hz
    window_size : samples per window (default 2400 = 0.2s at 12kHz)
    hop_size : step between windows (default 1200 = 50% overlap)
    bearing_params : OEM specs for defect frequency computation
    rpm : shaft speed for converting defect frequency multiples to Hz

    Returns
    -------
    DataFrame with one row per window and a time_seconds column.
    """
    signal_data = np.asarray(signal_data).flatten()
    n_samples = len(signal_data)
    rows = []

    start = 0
    while start + window_size <= n_samples:
        window = signal_data[start : start + window_size]
        feats = extract_window_features(window, sr, bearing_params, rpm)
        feats["time_seconds"] = (start + window_size / 2) / sr
        rows.append(feats)
        start += hop_size

    return pd.DataFrame(rows)


def load_cwru_signal(mat_path: str | Path, key: str | None = None) -> np.ndarray:
    """
    Load a vibration signal from a CWRU .mat file.

    If key is not provided, looks for the drive-end time series
    (variable name containing 'DE_time').
    """
    mat = loadmat(str(mat_path))
    if key is not None:
        return mat[key].flatten()

    # Auto-detect drive end key
    de_keys = [k for k in mat.keys() if "DE_time" in k]
    if not de_keys:
        raise KeyError(f"No DE_time variable found in {mat_path}. Keys: {list(mat.keys())}")
    return mat[de_keys[0]].flatten()


def build_degradation_trajectory(
    data_dir: str | Path = "data/raw",
    fault_type: str = "inner_race",
    load_hp: int = 0,
    sr: float = 12000.0,
    bearing_params: BearingOEMParams | None = None,
    window_size: int = 2400,
    hop_size: int = 1200,
) -> pd.DataFrame:
    """
    Build a synthetic degradation trajectory by concatenating CWRU data files
    in order of increasing fault severity.

    The CWRU dataset has pre-seeded faults, not run-to-failure data. We arrange
    files by severity (normal -> 0.007" -> 0.014" -> 0.021") to simulate the
    progression from healthy to failed.

    Parameters
    ----------
    data_dir : directory containing downloaded .mat files
    fault_type : "inner_race", "outer_race", or "ball"
    load_hp : motor load (0, 1, 2, or 3 HP)
    sr : sampling rate
    bearing_params : OEM bearing specs for defect frequency features
    window_size : feature extraction window size
    hop_size : feature extraction hop size

    Returns
    -------
    DataFrame with features, plus 'phase' and 'time_index' columns.
    """
    data_dir = Path(data_dir)
    load_suffix = f"{load_hp}hp"

    # Map fault type to file prefix
    prefix_map = {
        "inner_race": "ir",
        "outer_race": "or",
        "ball": "ball",
    }
    prefix = prefix_map[fault_type]

    # RPM for this load
    rpm_map = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}
    rpm = rpm_map[load_hp]

    # Ordered severity levels
    phases = [
        (f"normal_{load_suffix}", "healthy"),
        (f"{prefix}_007_{load_suffix}", "mild"),
        (f"{prefix}_014_{load_suffix}", "moderate"),
        (f"{prefix}_021_{load_suffix}", "severe"),
    ]

    all_features = []
    cumulative_time = 0.0

    for file_name, phase_label in phases:
        mat_path = data_dir / f"{file_name}.mat"
        if not mat_path.exists():
            print(f"Warning: {mat_path} not found, skipping {phase_label} phase")
            continue

        signal_data = load_cwru_signal(mat_path)
        features = extract_features(signal_data, sr, window_size, hop_size, bearing_params, rpm)
        features["phase"] = phase_label
        features["time_seconds"] = features["time_seconds"] + cumulative_time
        all_features.append(features)

        # Advance cumulative time by the full length of this file
        cumulative_time += len(signal_data) / sr

    if not all_features:
        raise FileNotFoundError(f"No CWRU files found in {data_dir}")

    trajectory = pd.concat(all_features, ignore_index=True)
    trajectory["time_index"] = range(len(trajectory))
    return trajectory
