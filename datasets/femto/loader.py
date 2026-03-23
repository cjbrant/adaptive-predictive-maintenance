"""FEMTO dataset loader."""
import numpy as np
import pandas as pd
from pathlib import Path
from framework.dataset_loader import DatasetLoader, DegradationTrajectory, OEMPrior
from datasets.femto.config import FEMTO_CONFIG
from datasets.femto.download import download_femto_data
from datasets.femto.feature_extraction import process_femto_bearing
from core.oem_prior import compute_l10_hours, compute_degradation_baseline


# Mapping of bearing IDs to directories
BEARING_MAP = {
    # Learning_set (full run-to-failure)
    "Bearing1_1": {"condition": 1, "split": "train"},
    "Bearing1_2": {"condition": 1, "split": "train"},
    "Bearing2_1": {"condition": 2, "split": "train"},
    "Bearing2_2": {"condition": 2, "split": "train"},
    # Test_set (may be truncated)
    "Bearing1_3": {"condition": 1, "split": "test"},
    "Bearing1_4": {"condition": 1, "split": "test"},
    "Bearing1_5": {"condition": 1, "split": "test"},
    "Bearing1_6": {"condition": 1, "split": "test"},
    "Bearing1_7": {"condition": 1, "split": "test"},
    "Bearing2_3": {"condition": 2, "split": "test"},
    "Bearing2_4": {"condition": 2, "split": "test"},
    "Bearing2_5": {"condition": 2, "split": "test"},
    "Bearing2_6": {"condition": 2, "split": "test"},
    "Bearing2_7": {"condition": 2, "split": "test"},
    "Bearing3_3": {"condition": 3, "split": "test"},
}


class FEMTOLoader(DatasetLoader):
    def __init__(self, data_dir: str = "data/raw/femto",
                 processed_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.config = FEMTO_CONFIG

    def download(self) -> None:
        download_femto_data(str(self.data_dir))

    def load_trajectories(self) -> list[DegradationTrajectory]:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        trajectories = []

        for bearing_id, info in BEARING_MAP.items():
            condition = info["condition"]
            split = info["split"]

            # Determine directory
            if split == "train":
                bearing_dir = self.data_dir / "Learning_set" / bearing_id
            else:
                bearing_dir = self.data_dir / "Test_set" / bearing_id

            if not bearing_dir.exists():
                # Try lowercase
                if split == "train":
                    bearing_dir = self.data_dir / "learning_set" / bearing_id
                else:
                    bearing_dir = self.data_dir / "test_set" / bearing_id
                if not bearing_dir.exists():
                    print(f"  Skipping {bearing_id}: directory not found")
                    continue

            # Load or extract features
            cache_path = self.processed_dir / f"femto_{bearing_id}.csv"
            if cache_path.exists():
                df = pd.read_csv(cache_path, index_col=0)
            else:
                try:
                    print(f"  Extracting features for {bearing_id}...")
                    df = process_femto_bearing(str(bearing_dir))
                    df.to_csv(cache_path)
                except Exception as e:
                    print(f"  Error processing {bearing_id}: {e}")
                    continue

            n = len(df)
            primary = self.config["feature_settings"]["primary_feature"]

            # Training bearings are run-to-failure
            is_rtf = (split == "train")
            failure_index = n - 1 if is_rtf else None

            true_rul = None
            if is_rtf:
                time_vals = df.index.values
                total_time = time_vals[-1]
                true_rul = np.array([total_time - t for t in time_vals])

            # Compute approximate OEM prior
            prior = self._compute_oem_prior(df, condition)

            traj = DegradationTrajectory(
                unit_id=f"femto_{bearing_id}",
                dataset="femto",
                features=df.reset_index(),
                primary_feature=primary,
                true_rul=true_rul,
                failure_index=failure_index,
                oem_prior=prior,
                operating_conditions=dict(self.config["conditions"][condition]),
                metadata={
                    "equipment_type": self.config["equipment_type"],
                    "condition": condition,
                    "split": split,
                    "bearing_id": bearing_id,
                },
                is_run_to_failure=is_rtf,
            )
            trajectories.append(traj)

        return trajectories

    def _compute_oem_prior(self, features_df: pd.DataFrame, condition: int) -> OEMPrior:
        """Compute approximate OEM prior using SKF 6204 specs."""
        specs = self.config["approximate_oem_specs"]
        cond = self.config["conditions"][condition]

        C_kn = specs["estimated_C_kn"]
        P_kn = cond["radial_load_N"] / 1000.0  # N -> kN
        rpm = cond["rpm"]
        p = specs["life_exponent"]

        l10h = compute_l10_hours(C_kn, P_kn, rpm, p)

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
            expected_life=l10h,
            baseline_curve=scaled_baseline,
            threshold=threshold,
            life_unit="hours",
            source=specs["comparable_model"],
            confidence="approximate_oem",
            parameters={"condition": condition, **specs},
        )

    def get_dataset_info(self) -> dict:
        return {
            "name": "femto",
            "equipment": self.config["equipment"],
            "equipment_type": self.config["equipment_type"],
            "prior_quality": self.config["prior_quality"],
            "n_trajectories": len(BEARING_MAP),
        }
