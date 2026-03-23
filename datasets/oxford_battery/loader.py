"""Oxford battery dataset loader."""
import numpy as np
import pandas as pd
from pathlib import Path
from framework.dataset_loader import DatasetLoader, DegradationTrajectory, OEMPrior
from datasets.oxford_battery.config import OXFORD_BATTERY_CONFIG
from datasets.oxford_battery.download import download_oxford_battery_data


class OxfordBatteryLoader(DatasetLoader):
    def __init__(self, data_dir: str = "data/raw/oxford_battery",
                 processed_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.config = OXFORD_BATTERY_CONFIG

    def download(self) -> None:
        download_oxford_battery_data(str(self.data_dir))

    def load_trajectories(self) -> list[DegradationTrajectory]:
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Check for cached data
        cache_files = list(self.processed_dir.glob("battery_cell_*.csv"))

        if cache_files:
            cell_dfs = {}
            for f in cache_files:
                cell_id = f.stem.replace("battery_cell_", "")
                cell_dfs[cell_id] = pd.read_csv(f, index_col=0)
        else:
            # Extract from mat file
            mat_files = list(self.data_dir.glob("*.mat"))
            if not mat_files:
                raise FileNotFoundError(
                    f"No .mat files found in {self.data_dir}. Run download() first."
                )

            from datasets.oxford_battery.feature_extraction import extract_all_cells
            nominal_cap = self.config["cell_specs"]["nominal_capacity_mah"]
            cell_dfs = extract_all_cells(str(mat_files[0]), nominal_cap)

            # Cache
            for cell_id, df in cell_dfs.items():
                cache_path = self.processed_dir / f"battery_cell_{cell_id}.csv"
                df.to_csv(cache_path)

        trajectories = []
        for cell_id, df in cell_dfs.items():
            n = len(df)
            primary = self.config["feature_settings"]["primary_feature"]

            # SOH degrades over time; "failure" = SOH drops below 80%
            soh_vals = df["soh"].values
            failure_idx = None
            for i, soh in enumerate(soh_vals):
                if soh < 0.80:
                    failure_idx = i
                    break

            # True RUL: cycles until SOH < 80%
            true_rul = None
            if failure_idx is not None:
                true_rul = np.array([max(0, failure_idx - i) for i in range(n)])

            # OEM prior: linear fade
            prior = self._compute_linear_prior(df)

            traj = DegradationTrajectory(
                unit_id=f"oxford_{cell_id}",
                dataset="oxford_battery",
                features=df.reset_index(),
                primary_feature=primary,
                true_rul=true_rul,
                failure_index=failure_idx,
                oem_prior=prior,
                operating_conditions={
                    "temperature_C": self.config["cell_specs"]["temperature_C"],
                    "charge_protocol": self.config["cell_specs"]["charge_protocol"],
                    "discharge_protocol": self.config["cell_specs"]["discharge_protocol"],
                },
                metadata={
                    "equipment_type": self.config["equipment_type"],
                    "cell_id": cell_id,
                },
                is_run_to_failure=(failure_idx is not None),
            )
            trajectories.append(traj)

        return trajectories

    def _compute_linear_prior(self, features_df: pd.DataFrame) -> OEMPrior:
        """Compute linear fade baseline for battery SOH."""
        rated_life = self.config["cell_specs"]["estimated_cycle_life_80pct"]
        n = len(features_df)

        # Linear baseline: SOH from 1.0 to 0.8 over rated_life cycles
        # Extended to full length of data
        t_norm = np.arange(n) / rated_life
        baseline_soh = 1.0 - 0.2 * t_norm  # Linear from 1.0 toward 0.8
        baseline_soh = np.clip(baseline_soh, 0.0, 1.0)

        return OEMPrior(
            expected_life=float(rated_life),
            baseline_curve=baseline_soh,
            threshold=0.80,
            life_unit="cycles",
            source="Estimated linear fade model (approximate)",
            confidence="approximate_oem",
            parameters=dict(self.config["cell_specs"]),
        )

    def get_dataset_info(self) -> dict:
        return {
            "name": "oxford_battery",
            "equipment": self.config["equipment"],
            "equipment_type": self.config["equipment_type"],
            "prior_quality": self.config["prior_quality"],
            "n_trajectories": self.config["cell_specs"]["n_cells"],
        }
