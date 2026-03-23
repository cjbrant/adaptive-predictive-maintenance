"""XJTU-SY dataset loader."""
import numpy as np
import pandas as pd
from pathlib import Path
from framework.dataset_loader import DatasetLoader, DegradationTrajectory, OEMPrior
from datasets.xjtu_sy.config import XJTU_SY_CONFIG
from datasets.xjtu_sy.download import download_xjtu_sy_data
from datasets.xjtu_sy.feature_extraction import process_xjtu_bearing, compute_defect_frequencies
from core.oem_prior import compute_l10_hours, compute_degradation_baseline


class XJTUSYLoader(DatasetLoader):
    def __init__(self, data_dir="data/raw/xjtu_sy", processed_dir="data/processed"):
        self.data_dir = data_dir
        self.processed_dir = Path(processed_dir)
        self.config = XJTU_SY_CONFIG

    def download(self):
        download_xjtu_sy_data(self.data_dir)

    def load_trajectories(self) -> list[DegradationTrajectory]:
        """Load all 15 XJTU-SY run-to-failure trajectories.

        For each condition (1-3), for each bearing (1-5):
        1. Load or extract features from CSV files
        2. Build time index (minutes from start)
        3. Compute true RUL in hours
        4. Compute OEM prior using L10 formula
        5. Return DegradationTrajectory
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        trajectories = []

        for cond_num, cond_info in self.config["conditions"].items():
            rpm = cond_info["rpm"]
            defect_freqs = compute_defect_frequencies(rpm)
            cond_dir = Path(self.data_dir) / cond_info["dir_name"]

            for bearing_idx in range(1, cond_info["n_bearings"] + 1):
                bearing_name = f"Bearing{cond_num}_{bearing_idx}"
                bearing_dir = cond_dir / bearing_name
                unit_id = f"xjtu_sy_{bearing_name}"

                df = self._load_or_extract(bearing_name, bearing_dir, defect_freqs, rpm)
                if df is None or len(df) == 0:
                    print(f"  Skipping {bearing_name}: no data")
                    continue

                n = len(df)
                failure_info = self.config["bearing_failures"].get(bearing_name, {})

                failure_index = n - 1
                time_hours = np.arange(n) / 60.0
                total_hours = time_hours[-1]
                true_rul = np.array([total_hours - t for t in time_hours])

                prior = self._compute_oem_prior(df, condition=cond_num)

                traj = DegradationTrajectory(
                    unit_id=unit_id,
                    dataset="xjtu_sy",
                    features=df.reset_index(drop=True),
                    primary_feature=self.config["feature_settings"]["primary_feature"],
                    true_rul=true_rul,
                    failure_index=failure_index,
                    oem_prior=prior,
                    operating_conditions={
                        "rpm": rpm,
                        "radial_load_kn": cond_info["radial_load_kn"],
                        "condition": cond_num,
                    },
                    metadata={
                        "equipment_type": self.config["equipment_type"],
                        "failure_mode": failure_info.get("failure", "unknown"),
                        "life_min": failure_info.get("life_min"),
                        "condition": cond_num,
                        "bearing_name": bearing_name,
                    },
                    is_run_to_failure=True,
                )
                trajectories.append(traj)

        return trajectories

    def _load_or_extract(self, bearing_name, bearing_dir, defect_freqs, rpm):
        cache_path = self.processed_dir / f"xjtu_sy_{bearing_name}_features.csv"
        if cache_path.exists():
            print(f"  Loading cached features for {bearing_name}")
            return pd.read_csv(cache_path)
        if not bearing_dir.exists():
            return None
        print(f"  Extracting features for {bearing_name}...")
        feature_list = process_xjtu_bearing(str(bearing_dir), sr=25600, defect_freqs=defect_freqs)
        if not feature_list:
            return None
        df = pd.DataFrame(feature_list)
        df["time_min"] = np.arange(len(df))
        df.to_csv(cache_path, index=False)
        print(f"  Cached {cache_path} ({len(df)} snapshots)")
        return df

    def _compute_oem_prior(self, features_df, condition):
        """Compute OEM prior for LDK UER204 under a specific condition.

        Uses same pattern as IMSLoader._compute_oem_prior():
        - Compute L10 hours from ISO 281
        - Generate exponential degradation baseline
        - Scale to feature space using healthy-phase statistics
        """
        specs = self.config["oem_specs"]
        cond = self.config["conditions"][condition]

        l10h = compute_l10_hours(
            C_kn=specs["C_kn"], P_kn=cond["radial_load_kn"],
            rpm=cond["rpm"], p=specs["life_exponent"],
        )

        n = len(features_df)
        baseline = compute_degradation_baseline(l10h, n)

        primary = self.config["feature_settings"]["primary_feature"]
        feat_vals = features_df[primary].values
        healthy_n = max(1, int(n * 0.1))
        healthy_mean = float(np.mean(feat_vals[:healthy_n]))
        healthy_std = float(np.std(feat_vals[:healthy_n]))
        threshold = healthy_mean + 5 * healthy_std
        if threshold <= healthy_mean:
            threshold = healthy_mean + 1.0

        scaled_baseline = healthy_mean + baseline * (threshold - healthy_mean)

        return OEMPrior(
            expected_life=l10h, baseline_curve=scaled_baseline,
            threshold=threshold, life_unit="hours",
            source=f"LDK {specs['designation']} catalog",
            confidence="exact_oem", parameters=dict(specs),
        )

    def get_dataset_info(self):
        return {
            "name": "xjtu_sy",
            "equipment": self.config["equipment"],
            "equipment_type": self.config["equipment_type"],
            "prior_quality": self.config["prior_quality"],
            "n_trajectories": 15,
        }
