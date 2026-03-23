"""IMS dataset loader."""
import numpy as np
import pandas as pd
from pathlib import Path
from framework.dataset_loader import DatasetLoader, DegradationTrajectory, OEMPrior
from datasets.ims.config import IMS_CONFIG
from datasets.ims.download import download_ims_data
from datasets.ims.feature_extraction import process_ims_experiment, compute_defect_frequencies
from core.oem_prior import compute_l10_hours, compute_degradation_baseline


class IMSLoader(DatasetLoader):
    def __init__(self, data_dir: str = "data/raw/ims",
                 processed_dir: str = "data/processed"):
        self.data_dir = data_dir
        self.processed_dir = Path(processed_dir)
        self.config = IMS_CONFIG

    def download(self) -> None:
        download_ims_data(self.data_dir)

    def load_trajectories(self) -> list[DegradationTrajectory]:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        defect_freqs = compute_defect_frequencies()
        trajectories = []

        for set_key, exp_info in self.config["experiments"].items():
            set_num = set_key.replace("set", "")
            exp_name = exp_info["dir_name"]

            # Load or extract features
            bearing_dfs = self._load_or_extract(set_num, exp_name, defect_freqs)

            for bearing_name, df in bearing_dfs.items():
                unit_id = f"ims_set{set_num}_{bearing_name}"
                failure_mode = exp_info["failures"].get(bearing_name)
                is_failed = failure_mode is not None
                n = len(df)

                # Compute RUL for failed bearings
                true_rul = None
                failure_index = None
                if is_failed:
                    failure_index = n - 1
                    time_vals = df.index.values if isinstance(df.index[0], (float, np.floating)) else np.arange(n) * (10/60)
                    total_time = time_vals[-1]
                    true_rul = np.array([total_time - t for t in time_vals])

                # Compute OEM prior
                prior = self._compute_oem_prior(df)

                traj = DegradationTrajectory(
                    unit_id=unit_id,
                    dataset="ims",
                    features=df.reset_index(),
                    primary_feature=self.config["feature_settings"]["primary_feature"],
                    true_rul=true_rul,
                    failure_index=failure_index,
                    oem_prior=prior,
                    operating_conditions=dict(self.config["operating_conditions"]),
                    metadata={
                        "equipment_type": self.config["equipment_type"],
                        "experiment": exp_name,
                        "failure_mode": failure_mode,
                        "set": set_num,
                    },
                    is_run_to_failure=is_failed,
                )
                trajectories.append(traj)

        return trajectories

    def _load_or_extract(self, set_num: str, experiment: str,
                          defect_freqs: dict) -> dict[str, pd.DataFrame]:
        """Load cached features or extract from raw data."""
        cached = {}
        all_cached = True
        for b in range(1, 5):
            bp = self.processed_dir / f"ims_set{set_num}_bearing{b}.csv"
            if bp.exists():
                cached[f"bearing{b}"] = pd.read_csv(bp, index_col=0)
            else:
                all_cached = False

        if all_cached and cached:
            print(f"Loading cached features for set {set_num}")
            return cached

        print(f"Extracting features for {experiment}...")
        bearing_dfs = process_ims_experiment(self.data_dir, experiment, defect_freqs)

        for name, df in bearing_dfs.items():
            bp = self.processed_dir / f"ims_set{set_num}_{name}.csv"
            df.to_csv(bp)
            print(f"  Cached {bp}")

        return bearing_dfs

    def _compute_oem_prior(self, features_df: pd.DataFrame) -> OEMPrior:
        """Compute OEM prior for Rexnord ZA-2115."""
        specs = self.config["oem_specs"]
        ops = self.config["operating_conditions"]

        l10h = compute_l10_hours(
            C_kn=specs["C_kn"],
            P_kn=ops["radial_load_kn"],
            rpm=ops["rpm"],
            p=specs["life_exponent"],
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
            expected_life=l10h,
            baseline_curve=scaled_baseline,
            threshold=threshold,
            life_unit="hours",
            source=f"Rexnord {specs['designation']} catalog",
            confidence="exact_oem",
            parameters=dict(specs),
        )

    def get_dataset_info(self) -> dict:
        return {
            "name": "ims",
            "equipment": self.config["equipment"],
            "equipment_type": self.config["equipment_type"],
            "prior_quality": self.config["prior_quality"],
            "n_trajectories": 12,
        }
