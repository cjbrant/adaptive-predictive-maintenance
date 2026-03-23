"""C-MAPSS health index construction."""
import numpy as np
import pandas as pd
from pathlib import Path


def load_cmapss_data(data_dir: str, sub_dataset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load train, test, and RUL ground truth."""
    col_names = (["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"] +
                 [f"s_{i}" for i in range(1, 22)])

    data_path = Path(data_dir)
    train = pd.read_csv(data_path / f"train_{sub_dataset}.txt",
                        sep=r"\s+", header=None, names=col_names)
    test = pd.read_csv(data_path / f"test_{sub_dataset}.txt",
                       sep=r"\s+", header=None, names=col_names)
    rul = np.loadtxt(data_path / f"RUL_{sub_dataset}.txt")
    return train, test, rul


def compute_health_index(train: pd.DataFrame, test: pd.DataFrame,
                          informative_sensors: list[int],
                          rul_cap: int = 125) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct health index from sensor readings.

    Normalizes using training set min/max (no data leakage).
    """
    sensor_cols = [f"s_{i}" for i in informative_sensors]

    # Fit scaler on training data only
    train_min = train[sensor_cols].min()
    train_max = train[sensor_cols].max()
    denom = train_max - train_min
    denom[denom == 0] = 1e-8

    for df in [train, test]:
        df[sensor_cols] = (df[sensor_cols] - train_min) / denom
        # Smooth per engine
        for col in sensor_cols:
            df[col] = df.groupby("unit_nr")[col].transform(
                lambda x: x.rolling(5, min_periods=1).mean())
        df["health_index"] = df[sensor_cols].mean(axis=1)

    # RUL labels for training set
    for unit in train["unit_nr"].unique():
        mask = train["unit_nr"] == unit
        max_cycle = train.loc[mask, "time_cycles"].max()
        train.loc[mask, "rul"] = max_cycle - train.loc[mask, "time_cycles"]
    train["rul"] = train["rul"].clip(upper=rul_cap)

    return train, test


def compute_fleet_prior(train_df: pd.DataFrame) -> np.ndarray:
    """Compute fleet-average health_index trajectory from training engines.

    1. For each engine, reverse its health_index series (align from end-of-life)
    2. Compute pointwise mean across all engines
    3. Reverse back to get forward-time curve
    """
    reversed_series = []
    for unit in train_df["unit_nr"].unique():
        hi = train_df[train_df["unit_nr"] == unit]["health_index"].values
        reversed_series.append(hi[::-1])

    max_len = max(len(s) for s in reversed_series)
    padded = np.array([
        np.pad(s, (0, max_len - len(s)), mode="edge")
        for s in reversed_series
    ])
    fleet_avg_reversed = np.nanmean(padded, axis=0)
    return fleet_avg_reversed[::-1]
