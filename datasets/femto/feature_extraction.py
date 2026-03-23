"""FEMTO-specific feature extraction (2-axis vibration)."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def extract_femto_features(horizontal: np.ndarray, vertical: np.ndarray,
                            sr: float = 25600) -> dict:
    """Extract features from a single FEMTO recording (0.1s, 2560 samples)."""
    rms_h = float(np.sqrt(np.mean(horizontal ** 2)))
    rms_v = float(np.sqrt(np.mean(vertical ** 2)))

    features = {
        "rms": rms_h,
        "kurtosis": float(stats.kurtosis(horizontal, fisher=True) + 3),
        "crest_factor": float(np.max(np.abs(horizontal)) / rms_h) if rms_h > 0 else 0.0,
        "peak_to_peak": float(np.max(horizontal) - np.min(horizontal)),
        "skewness": float(stats.skew(horizontal)),
        "rms_vertical": rms_v,
        "kurtosis_vertical": float(stats.kurtosis(vertical, fisher=True) + 3),
        "rms_combined": float(np.sqrt(rms_h**2 + rms_v**2)),
    }
    return features


def process_femto_bearing(bearing_dir: str) -> pd.DataFrame:
    """Process all recordings from one FEMTO bearing into feature DataFrame.

    Each acc_XXXXX.csv has 2560 rows x 2 columns (horizontal, vertical accel).
    Records taken every 10 seconds.
    """
    bearing_path = Path(bearing_dir)
    if not bearing_path.exists():
        raise FileNotFoundError(f"Bearing directory not found: {bearing_path}")

    acc_files = sorted(bearing_path.glob("acc_*.csv"))
    if not acc_files:
        raise FileNotFoundError(f"No acc_*.csv files in {bearing_path}")

    all_features = []
    for fpath in acc_files:
        try:
            data = pd.read_csv(fpath, header=None)
            if data.shape[1] >= 2:
                horizontal = data.iloc[:, 0].values
                vertical = data.iloc[:, 1].values
            else:
                horizontal = data.iloc[:, 0].values
                vertical = np.zeros_like(horizontal)

            feats = extract_femto_features(horizontal, vertical)
            all_features.append(feats)
        except Exception:
            continue

    if not all_features:
        raise ValueError(f"No valid recordings in {bearing_path}")

    df = pd.DataFrame(all_features)
    # 10-second intervals
    df["time_hours"] = np.arange(len(df)) * (10.0 / 3600.0)
    df = df.set_index("time_hours")
    return df
