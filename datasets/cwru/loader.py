from framework.dataset_loader import DatasetLoader, DegradationTrajectory, OEMPrior
from datasets.cwru.config import CWRU_CONFIG
from datasets.cwru.download import download_cwru_data
from datasets.cwru.feature_extraction import build_degradation_trajectory, BearingOEMParams
from core.oem_prior import compute_l10_hours, compute_degradation_baseline, estimate_bearing_load
import numpy as np


class CWRULoader(DatasetLoader):
    def __init__(self, data_dir: str = "data/raw/cwru"):
        self.data_dir = data_dir
        self.config = CWRU_CONFIG

    def download(self) -> None:
        download_cwru_data(output_dir=self.data_dir, subset="minimal")

    def load_trajectories(self) -> list[DegradationTrajectory]:
        specs = self.config["oem_specs"]
        ops = self.config["operating_conditions"]

        bearing_params = BearingOEMParams(
            model=specs["designation"],
            bore_mm=specs["bore_mm"],
            dynamic_load_rating_kn=specs["C_kn"],
            static_load_rating_kn=specs["C0_kn"],
            life_exponent=specs["life_exponent"],
            bpfi=specs["bpfi_mult"],
            bpfo=specs["bpfo_mult"],
            bsf=specs["bsf_mult"],
            ftf=specs["ftf_mult"],
            max_speed_rpm=10000,
        )

        features = build_degradation_trajectory(
            data_dir=self.data_dir,
            fault_type=self.config["trajectory_construction"]["fault_type"],
            load_hp=0,
            sr=ops["sampling_rate_hz"],
            bearing_params=bearing_params,
        )

        n = len(features)
        primary = self.config["trajectory_construction"]["primary_feature"]

        # Compute OEM prior
        P_kn = estimate_bearing_load(motor_hp=2, rpm=ops["rpm"])
        l10h = compute_l10_hours(specs["C_kn"], P_kn, ops["rpm"], specs["life_exponent"])
        baseline = compute_degradation_baseline(l10h, n)

        # Scale baseline to feature range
        feat_vals = features[primary].values
        healthy_n = max(1, int(n * 0.1))
        healthy_mean = float(np.mean(feat_vals[:healthy_n]))
        healthy_std = float(np.std(feat_vals[:healthy_n]))
        threshold = healthy_mean + 5 * healthy_std
        if threshold <= healthy_mean:
            threshold = healthy_mean + 1.0
        scaled_baseline = healthy_mean + baseline * (threshold - healthy_mean)

        prior = OEMPrior(
            expected_life=l10h,
            baseline_curve=scaled_baseline,
            threshold=threshold,
            life_unit="hours",
            source=f"SKF {specs['designation']} catalog",
            confidence="exact_oem",
            parameters=dict(specs),
        )

        traj = DegradationTrajectory(
            unit_id="cwru_inner_race_0hp",
            dataset="cwru",
            features=features,
            primary_feature=primary,
            true_rul=np.linspace(n, 0, n),
            failure_index=n - 1,
            oem_prior=prior,
            operating_conditions=dict(ops),
            metadata={"equipment_type": self.config["equipment_type"]},
            is_run_to_failure=False,
        )
        return [traj]

    def get_dataset_info(self) -> dict:
        return {
            "name": "cwru",
            "equipment": self.config["equipment"],
            "equipment_type": self.config["equipment_type"],
            "prior_quality": self.config["prior_quality"],
            "n_trajectories": 1,
        }
