"""IMS-specific vibration feature extraction."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def compute_defect_frequencies(rpm: float = 2000) -> dict:
    """Compute defect frequencies for Rexnord ZA-2115.

    BPFO = (n/2) * (1 - (d/D) * cos(alpha)) * shaft_freq
    BPFI = (n/2) * (1 + (d/D) * cos(alpha)) * shaft_freq
    BSF  = (D/(2d)) * (1 - ((d/D) * cos(alpha))^2) * shaft_freq
    FTF  = (1/2) * (1 - (d/D) * cos(alpha)) * shaft_freq
    """
    n_rollers = 16
    d_roller = 0.331   # inches
    d_pitch = 2.815    # inches
    alpha = np.radians(15.17)
    shaft_freq = rpm / 60.0

    ratio = d_roller / d_pitch
    cos_alpha = np.cos(alpha)

    bpfo = (n_rollers / 2) * (1 - ratio * cos_alpha) * shaft_freq
    bpfi = (n_rollers / 2) * (1 + ratio * cos_alpha) * shaft_freq
    bsf = (d_pitch / (2 * d_roller)) * (1 - (ratio * cos_alpha)**2) * shaft_freq
    ftf = 0.5 * (1 - ratio * cos_alpha) * shaft_freq

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


def extract_ims_features(snapshot: np.ndarray, sr: float = 20000,
                          defect_freqs: dict | None = None) -> dict:
    """Extract features from a single 1-second IMS vibration snapshot."""
    rms_val = float(np.sqrt(np.mean(snapshot ** 2)))
    features = {
        "rms": rms_val,
        "peak": float(np.max(np.abs(snapshot))),
        "kurtosis": float(stats.kurtosis(snapshot, fisher=True) + 3),  # excess -> regular kurtosis
        "skewness": float(stats.skew(snapshot)),
        "crest_factor": float(np.max(np.abs(snapshot)) / rms_val) if rms_val > 0 else 0.0,
        "peak_to_peak": float(np.max(snapshot) - np.min(snapshot)),
    }

    if defect_freqs:
        for name, freq in defect_freqs.items():
            features[f"{name}_energy"] = compute_spectral_energy(snapshot, sr, freq)

    return features


def process_ims_experiment(data_dir: str, experiment: str,
                            defect_freqs: dict | None = None,
                            sr: float = 20000) -> dict[str, pd.DataFrame]:
    """Process all snapshots from one IMS experiment into feature DataFrames.

    Returns dict mapping bearing name -> DataFrame with time_hours index.
    """
    exp_dir = Path(data_dir) / experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # IMS files are named with timestamps, sort alphabetically = chronologically
    files = sorted([f for f in exp_dir.iterdir() if f.is_file() and not f.name.startswith('.')])

    if defect_freqs is None:
        defect_freqs = compute_defect_frequencies()

    n_channels = 8 if experiment == "1st_test" else 4
    n_bearings = 4

    all_features = {b: [] for b in range(n_bearings)}
    skipped = 0

    for i, fpath in enumerate(files):
        if i % 500 == 0:
            print(f"  Processing {experiment}: file {i+1}/{len(files)}")
        try:
            # Try tab-delimited first, then space
            try:
                data = np.loadtxt(fpath, delimiter='\t')
            except ValueError:
                data = np.loadtxt(fpath)

            if data.ndim == 1:
                skipped += 1
                continue
            if data.shape[1] < n_channels:
                # Some files may have fewer columns
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        for bearing_idx in range(n_bearings):
            if experiment == "1st_test":
                # 2 channels per bearing, use first channel
                ch = bearing_idx * 2
                if ch >= data.shape[1]:
                    continue
                channel_data = data[:, ch]
            else:
                if bearing_idx >= data.shape[1]:
                    continue
                channel_data = data[:, bearing_idx]

            feats = extract_ims_features(channel_data, sr, defect_freqs)
            all_features[bearing_idx].append(feats)

    if skipped > 0:
        print(f"  Skipped {skipped} corrupt/unreadable files in {experiment}")

    # Build DataFrames per bearing
    result_dfs = {}
    for bearing_idx in range(n_bearings):
        if not all_features[bearing_idx]:
            continue
        df = pd.DataFrame(all_features[bearing_idx])
        # 10-minute intervals between snapshots
        df["time_hours"] = np.arange(len(df)) * (10.0 / 60.0)
        df = df.set_index("time_hours")
        result_dfs[f"bearing{bearing_idx + 1}"] = df

    return result_dfs
