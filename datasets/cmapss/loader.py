"""C-MAPSS dataset loader."""
import numpy as np
import pandas as pd
from pathlib import Path
from framework.dataset_loader import DatasetLoader, DegradationTrajectory, OEMPrior
from datasets.cmapss.config import CMAPSS_CONFIG
from datasets.cmapss.download import download_cmapss_data
from datasets.cmapss.feature_extraction import load_cmapss_data, compute_health_index, compute_fleet_prior


class CMAPSSLoader(DatasetLoader):
    def __init__(self, sub_dataset: str = "FD001",
                 data_dir: str = "data/raw/cmapss",
                 processed_dir: str = "data/processed"):
        self.sub_dataset = sub_dataset
        self.data_dir = data_dir
        self.processed_dir = Path(processed_dir)
        self.config = CMAPSS_CONFIG

    def download(self) -> None:
        download_cmapss_data(self.data_dir)

    def load_trajectories(self) -> list[DegradationTrajectory]:
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        cache_path = self.processed_dir / f"cmapss_{self.sub_dataset}_processed.csv"
        fleet_cache = self.processed_dir / f"cmapss_{self.sub_dataset}_fleet_prior.npy"

        train, test, test_rul = load_cmapss_data(self.data_dir, self.sub_dataset)
        train, test = compute_health_index(
            train, test,
            self.config["informative_sensors"],
            self.config["rul_cap"]
        )

        fleet_prior = compute_fleet_prior(train)

        # Save fleet prior
        np.save(fleet_cache, fleet_prior)

        trajectories = []

        # Training engines (full run-to-failure, known RUL)
        for unit in train["unit_nr"].unique():
            unit_data = train[train["unit_nr"] == unit].copy()
            n = len(unit_data)

            features_df = unit_data[["time_cycles", "health_index"]].copy()
            features_df = features_df.reset_index(drop=True)

            rul_values = unit_data["rul"].values

            # Align fleet prior to this engine's length
            if n <= len(fleet_prior):
                baseline = fleet_prior[:n]
            else:
                baseline = np.pad(fleet_prior, (0, n - len(fleet_prior)), mode="edge")

            prior = OEMPrior(
                expected_life=float(len(fleet_prior)),
                baseline_curve=baseline,
                threshold=0.0,  # Health index degrades toward 0
                life_unit="cycles",
                source=f"Fleet average from {self.sub_dataset} training set",
                confidence="fleet_derived",
                parameters={"sub_dataset": self.sub_dataset, "n_train_engines": len(train["unit_nr"].unique())},
            )

            traj = DegradationTrajectory(
                unit_id=f"cmapss_{self.sub_dataset}_train_{int(unit)}",
                dataset="cmapss",
                features=features_df,
                primary_feature="health_index",
                true_rul=rul_values,
                failure_index=n - 1,
                oem_prior=prior,
                operating_conditions={"sub_dataset": self.sub_dataset},
                metadata={
                    "equipment_type": self.config["equipment_type"],
                    "split": "train",
                    "unit_nr": int(unit),
                },
                is_run_to_failure=True,
            )
            trajectories.append(traj)

        # Test engines (truncated, RUL known only at last step)
        for i, unit in enumerate(sorted(test["unit_nr"].unique())):
            unit_data = test[test["unit_nr"] == unit].copy()
            n = len(unit_data)

            features_df = unit_data[["time_cycles", "health_index"]].copy()
            features_df = features_df.reset_index(drop=True)

            # Ground truth RUL at last step only
            last_rul = float(test_rul[i])
            # Build full RUL array: at each step, RUL = last_rul + (n - 1 - step_idx)
            rul_values = np.array([last_rul + (n - 1 - j) for j in range(n)])
            rul_values = np.clip(rul_values, 0, self.config["rul_cap"])

            if n <= len(fleet_prior):
                baseline = fleet_prior[:n]
            else:
                baseline = np.pad(fleet_prior, (0, n - len(fleet_prior)), mode="edge")

            prior = OEMPrior(
                expected_life=float(len(fleet_prior)),
                baseline_curve=baseline,
                threshold=0.0,
                life_unit="cycles",
                source=f"Fleet average from {self.sub_dataset} training set",
                confidence="fleet_derived",
                parameters={"sub_dataset": self.sub_dataset},
            )

            traj = DegradationTrajectory(
                unit_id=f"cmapss_{self.sub_dataset}_test_{int(unit)}",
                dataset="cmapss",
                features=features_df,
                primary_feature="health_index",
                true_rul=rul_values,
                failure_index=None,  # Test engines are truncated, not run to failure
                oem_prior=prior,
                operating_conditions={"sub_dataset": self.sub_dataset},
                metadata={
                    "equipment_type": self.config["equipment_type"],
                    "split": "test",
                    "unit_nr": int(unit),
                    "ground_truth_rul_at_end": last_rul,
                },
                is_run_to_failure=False,
            )
            trajectories.append(traj)

        return trajectories

    def get_dataset_info(self) -> dict:
        sub = self.config["sub_datasets"][self.sub_dataset]
        return {
            "name": "cmapss",
            "equipment": self.config["equipment"],
            "equipment_type": self.config["equipment_type"],
            "prior_quality": self.config["prior_quality"],
            "n_trajectories": sub["n_train"] + sub["n_test"],
            "sub_dataset": self.sub_dataset,
        }
