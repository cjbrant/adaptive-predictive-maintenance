# framework/benchmark_runner.py
"""Run all models on all datasets, collect results."""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from framework.dataset_loader import DegradationTrajectory, DatasetLoader
from core.adaptive_drift import adaptive_drift_pid, adaptive_drift_with_regime, PIDParams
from core.baselines import static_degradation_curve, rolling_refit_curve, threshold_alarm
from core.evaluation import compute_rul_metrics, compute_detection_metrics, compute_actual_rul

DEFAULT_MODELS = ["threshold_alarm", "static_curve", "rolling_refit", "pid_adaptive", "pid_regime"]


def run_single_trajectory(trajectory: DegradationTrajectory,
                          models: list[str] | None = None) -> pd.DataFrame:
    """Run all models on one trajectory. Return one-row-per-model DataFrame.

    Steps for each model:
    1. If model needs an OEM prior (pid_adaptive, pid_regime) and trajectory.oem_prior is None, skip with warning.
    2. Run the model on trajectory.features[trajectory.primary_feature] using trajectory.oem_prior.baseline_curve as the baseline.
    3. Compute metrics against trajectory.true_rul (if available).
    4. Compute detection metrics against trajectory.failure_index.
    5. Return results row with: dataset, unit_id, model, all metrics, prior_quality, equipment_type, is_run_to_failure.

    Handle edge cases:
    - true_rul is None → RUL metrics are NaN, detection metrics still computed
    - failure_index is None (non-failed unit) → only false alarm rate computed
    - trajectory shorter than 50 steps → skip rolling_refit, flag in metadata
    """
    if models is None:
        models = DEFAULT_MODELS.copy()

    features = trajectory.features
    primary = trajectory.primary_feature
    n = len(features)
    has_rul = trajectory.true_rul is not None
    has_prior = trajectory.oem_prior is not None
    has_failure = trajectory.failure_index is not None

    if has_rul:
        actual_rul = trajectory.true_rul
    elif has_failure:
        actual_rul = compute_actual_rul(n, trajectory.failure_index)
    else:
        actual_rul = None

    rows = []
    for model_name in models:
        if model_name in ("pid_adaptive", "pid_regime") and not has_prior:
            warnings.warn(f"Skipping {model_name} for {trajectory.unit_id}: no OEM prior")
            continue
        if model_name == "rolling_refit" and n < 50:
            warnings.warn(f"Skipping rolling_refit for {trajectory.unit_id}: only {n} steps")
            continue

        predicted_rul = None
        alarm_series = None
        threshold_val = trajectory.oem_prior.threshold if has_prior else None

        try:
            if model_name == "threshold_alarm":
                result = threshold_alarm(features, primary)
                alarm_series = result.alarm_series
            elif model_name == "static_curve":
                result = static_degradation_curve(features, primary, threshold=threshold_val or 1.0)
                predicted_rul = result.predicted_rul
            elif model_name == "rolling_refit":
                result = rolling_refit_curve(features, primary, threshold=threshold_val or 1.0)
                predicted_rul = result.predicted_rul
            elif model_name == "pid_adaptive":
                observed = features[primary].values
                baseline = trajectory.oem_prior.baseline_curve[:n]
                result = adaptive_drift_pid(observed, baseline, threshold=threshold_val or 1.0)
                predicted_rul = result.predicted_rul
            elif model_name == "pid_regime":
                observed = features[primary].values
                baseline = trajectory.oem_prior.baseline_curve[:n]
                result = adaptive_drift_with_regime(observed, baseline, threshold=threshold_val or 1.0)
                predicted_rul = result.predicted_rul
        except Exception as e:
            warnings.warn(f"Error running {model_name} on {trajectory.unit_id}: {e}")
            continue

        row = {
            "dataset": trajectory.dataset,
            "unit_id": trajectory.unit_id,
            "model": model_name,
            "prior_quality": trajectory.oem_prior.confidence if has_prior else "none",
            "equipment_type": trajectory.metadata.get("equipment_type", "unknown"),
            "is_run_to_failure": trajectory.is_run_to_failure,
            "n_valid": n,
        }

        # RUL metrics
        if predicted_rul is not None and actual_rul is not None:
            rul_metrics = compute_rul_metrics(predicted_rul, actual_rul)
            row.update({"rmse": rul_metrics.rmse, "mae": rul_metrics.mae,
                       "nasa_score": rul_metrics.score, "mean_bias": rul_metrics.mean_bias})
        else:
            row.update({"rmse": np.nan, "mae": np.nan, "nasa_score": np.nan, "mean_bias": np.nan})

        # Detection metrics
        if has_failure:
            if alarm_series is not None:
                det = compute_detection_metrics(alarm_series, trajectory.failure_index)
            elif predicted_rul is not None:
                alarm = (predicted_rul < 50).astype(float)
                det = compute_detection_metrics(alarm, trajectory.failure_index)
            else:
                det = None
            if det is not None:
                row.update({"detection_lead_time": det.detection_lead_time,
                           "false_alarm_rate": det.false_alarm_rate,
                           "detection_success": det.detection_success})
            else:
                row.update({"detection_lead_time": np.nan, "false_alarm_rate": np.nan,
                           "detection_success": np.nan})
        else:
            row.update({"detection_lead_time": np.nan, "false_alarm_rate": np.nan,
                       "detection_success": np.nan})

        rows.append(row)

    return pd.DataFrame(rows)


def run_dataset(loader: DatasetLoader, models: list[str] | None = None) -> pd.DataFrame:
    """Run all models on all trajectories from a dataset. Save CSV."""
    info = loader.get_dataset_info()
    trajectories = loader.load_trajectories()
    all_results = []
    for traj in trajectories:
        result = run_single_trajectory(traj, models)
        all_results.append(result)
    combined = pd.concat(all_results, ignore_index=True)
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{info['name']}_metrics.csv"
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} rows to {output_path}")
    return combined


def _ensure_rag_extraction():
    """Run RAG ingestion and extraction if needed."""
    json_path = Path("analysis/extracted_oem_params.json")
    oem_dir = Path("data/oem")

    if json_path.exists():
        return  # Already extracted

    if not oem_dir.exists() or not list(oem_dir.glob("*.pdf")):
        return  # No PDFs to process

    try:
        from rag.extract_params import run_full_extraction
        print("\nRunning RAG extraction from OEM PDFs...")
        run_full_extraction()
    except Exception as e:
        warnings.warn(f"RAG extraction failed: {e}. Using hardcoded configs.")


def run_full_benchmark(datasets: list[str] | None = None) -> pd.DataFrame:
    """Run the complete benchmark across all datasets."""
    if datasets is None:
        datasets = ["cwru", "ims", "femto", "cmapss", "xjtu_sy"]

    loader_map = {}
    try:
        from datasets.cwru.loader import CWRULoader
        loader_map["cwru"] = CWRULoader
    except ImportError:
        pass
    try:
        from datasets.ims.loader import IMSLoader
        loader_map["ims"] = IMSLoader
    except ImportError:
        pass
    try:
        from datasets.femto.loader import FEMTOLoader
        loader_map["femto"] = FEMTOLoader
    except ImportError:
        pass
    try:
        from datasets.cmapss.loader import CMAPSSLoader
        loader_map["cmapss"] = CMAPSSLoader
    except ImportError:
        pass
    try:
        from datasets.xjtu_sy.loader import XJTUSYLoader
        loader_map["xjtu_sy"] = XJTUSYLoader
    except ImportError:
        pass

    # Run RAG extraction if OEM PDFs exist and results don't
    _ensure_rag_extraction()

    all_results = []
    for name in datasets:
        if name not in loader_map:
            warnings.warn(f"Unknown or unavailable dataset: {name}")
            continue
        print(f"\n{'='*60}\nRunning benchmark on {name.upper()}\n{'='*60}")
        try:
            loader = loader_map[name]()
            result = run_dataset(loader)
            all_results.append(result)
        except Exception as e:
            warnings.warn(f"Failed to run {name}: {e}")

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    Path("analysis").mkdir(exist_ok=True)
    combined.to_csv("analysis/cross_dataset_summary.csv", index=False)
    summary = combined.groupby(["dataset", "model"]).agg(
        mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"), mean_nasa=("nasa_score", "mean"),
    ).round(2)
    print("\n" + summary.to_string())
    return combined


if __name__ == "__main__":
    run_full_benchmark()
