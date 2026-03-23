"""XJTU-SY vibration feature extraction for LDK UER204 bearings."""
import numpy as np
from scipy import stats


def compute_defect_frequencies(rpm: float = 2100) -> dict:
    """Compute defect frequencies for LDK UER204.
    n = 8 balls, d = 7.92 mm, D = 34.55 mm, alpha = 0 deg.
    cos(alpha) = 1, simplifying the formulas.
    """
    n_balls = 8
    d_ball = 7.92     # mm
    d_pitch = 34.55   # mm
    shaft_freq = rpm / 60.0
    ratio = d_ball / d_pitch

    bpfo = (n_balls / 2) * (1 - ratio) * shaft_freq
    bpfi = (n_balls / 2) * (1 + ratio) * shaft_freq
    bsf = (d_pitch / (2 * d_ball)) * (1 - ratio**2) * shaft_freq
    ftf = 0.5 * (1 - ratio) * shaft_freq

    return {"bpfo": bpfo, "bpfi": bpfi, "bsf": bsf, "ftf": ftf}


def compute_spectral_energy(signal: np.ndarray, sr: float,
                            center_freq: float, bandwidth: float = 5.0,
                            n_harmonics: int = 3) -> float:
    """Compute spectral energy around a defect frequency and its harmonics."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    fft_mag = np.abs(np.fft.rfft(signal)) ** 2
    total_energy = 0.0
    for h in range(1, n_harmonics + 1):
        f_center = center_freq * h
        mask = (freqs >= f_center - bandwidth) & (freqs <= f_center + bandwidth)
        total_energy += float(np.sum(fft_mag[mask]))
    return total_energy


def extract_xjtu_features(snapshot: np.ndarray, sr: float = 25600,
                           defect_freqs: dict | None = None) -> dict:
    """Extract features from one XJTU-SY snapshot (32768 samples).
    Uses horizontal channel (column 0) as primary if 2D array passed.
    """
    if snapshot.ndim == 2:
        snapshot = snapshot[:, 0]
    rms_val = float(np.sqrt(np.mean(snapshot ** 2)))
    features = {
        "rms": rms_val,
        "peak": float(np.max(np.abs(snapshot))),
        "kurtosis": float(stats.kurtosis(snapshot, fisher=True) + 3),
        "skewness": float(stats.skew(snapshot)),
        "crest_factor": float(np.max(np.abs(snapshot)) / rms_val) if rms_val > 0 else 0.0,
        "peak_to_peak": float(np.max(snapshot) - np.min(snapshot)),
    }
    if defect_freqs:
        for name, freq in defect_freqs.items():
            features[f"{name}_energy"] = compute_spectral_energy(snapshot, sr, freq)
    return features


def process_xjtu_bearing(bearing_dir: str, sr: float = 25600,
                          defect_freqs: dict | None = None) -> list[dict]:
    """Process all CSV snapshots from one bearing directory.
    Each CSV: 32768 rows x 2 columns (horizontal, vertical acceleration).
    Sort CSV files numerically by stem. Use numpy loadtxt with comma delimiter.
    """
    from pathlib import Path
    bearing_path = Path(bearing_dir)
    csv_files = sorted(
        bearing_path.glob("*.csv"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )
    all_features = []
    for csv_file in csv_files:
        try:
            data = np.loadtxt(csv_file, delimiter=",")
            if data.ndim == 1:
                continue
            snapshot = data[:, 0]
            features = extract_xjtu_features(snapshot, sr, defect_freqs)
            all_features.append(features)
        except Exception as e:
            print(f"  Warning: skipped {csv_file.name}: {e}")
    return all_features
