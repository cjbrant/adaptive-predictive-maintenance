"""Evaluation metrics for predictive maintenance models.

Computes RUL prediction accuracy, detection lead time, false alarm rates,
and model comparison tables. Uses the NASA asymmetric scoring function
that penalizes late predictions more heavily than early ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class RULMetrics:
    """RUL prediction quality metrics."""

    rmse: float
    mae: float
    score: float  # NASA asymmetric scoring
    mean_bias: float  # positive = optimistic (dangerous)
    n_valid: int


@dataclass
class DetectionMetrics:
    """Early warning detection metrics."""

    detection_lead_time: int | None  # steps before failure
    false_alarm_rate: float
    detection_success: bool


def compute_rul_metrics(
    predicted_rul: np.ndarray,
    actual_rul: np.ndarray,
) -> RULMetrics:
    """
    Compare predicted remaining useful life against actual.

    Parameters
    ----------
    predicted_rul : model's RUL prediction at each time step
    actual_rul : true remaining life at each time step

    Returns
    -------
    RULMetrics with RMSE, MAE, asymmetric score, and mean bias
    """
    valid = np.isfinite(predicted_rul) & np.isfinite(actual_rul)
    if not np.any(valid):
        return RULMetrics(rmse=np.nan, mae=np.nan, score=np.nan, mean_bias=np.nan, n_valid=0)

    pred = predicted_rul[valid]
    actual = actual_rul[valid]

    diff = pred - actual  # positive = predicted more life than actual (optimistic/dangerous)

    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    mean_bias = float(np.mean(diff))

    # NASA asymmetric scoring function
    # Late predictions (d >= 0, optimistic) penalized more heavily
    score = 0.0
    for d in diff:
        if d < 0:
            # Early prediction (conservative): lighter penalty
            # Clamp exponent to prevent overflow
            exp_arg = min(-d / 13.0, 20.0)
            penalty = np.exp(exp_arg) - 1.0
        else:
            # Late prediction (optimistic): heavier penalty
            exp_arg = min(d / 10.0, 20.0)
            penalty = np.exp(exp_arg) - 1.0
        score += min(penalty, 1e6)  # Cap at 1e6

    return RULMetrics(
        rmse=rmse,
        mae=mae,
        score=float(score),
        mean_bias=mean_bias,
        n_valid=int(np.sum(valid)),
    )


def compute_detection_metrics(
    alarm_series: np.ndarray,
    true_failure_index: int,
    healthy_end_index: int | None = None,
) -> DetectionMetrics:
    """
    Evaluate early warning detection performance.

    Parameters
    ----------
    alarm_series : binary array (1 = alarm raised)
    true_failure_index : index at which true failure occurs
    healthy_end_index : index marking end of healthy phase (for false alarm calc).
                        If None, uses first 25% of data.
    """
    alarm_indices = np.where(alarm_series > 0)[0]

    if healthy_end_index is None:
        healthy_end_index = len(alarm_series) // 4

    # Detection lead time
    pre_failure_alarms = alarm_indices[alarm_indices < true_failure_index]
    if len(pre_failure_alarms) > 0:
        first_detection = int(pre_failure_alarms[0])
        lead_time = true_failure_index - first_detection
        detection_success = True
    else:
        lead_time = None
        detection_success = False

    # False alarm rate during healthy phase
    healthy_alarms = alarm_series[:healthy_end_index]
    false_alarm_rate = float(np.mean(healthy_alarms)) if healthy_end_index > 0 else 0.0

    return DetectionMetrics(
        detection_lead_time=lead_time,
        false_alarm_rate=false_alarm_rate,
        detection_success=detection_success,
    )


def compute_actual_rul(n_points: int, failure_index: int) -> np.ndarray:
    """
    Compute actual RUL at each time step.

    RUL decreases linearly from failure_index to 0 at the failure point,
    and is 0 for all points after failure.
    """
    rul = np.maximum(0, failure_index - np.arange(n_points)).astype(float)
    return rul


def compare_models(
    results: dict[str, dict],
    actual_rul: np.ndarray,
    true_failure_index: int,
    healthy_end_index: int | None = None,
) -> pd.DataFrame:
    """
    Compare all models side by side.

    Parameters
    ----------
    results : dict mapping model name to dict with keys:
              'predicted_rul' (array) and optionally 'alarm_series' (array)
    actual_rul : true RUL at each step
    true_failure_index : index of true failure
    healthy_end_index : end of healthy phase for false alarm calculation

    Returns
    -------
    DataFrame with one row per model, sorted by RMSE.
    """
    rows = []
    for name, result in results.items():
        rul_metrics = compute_rul_metrics(result["predicted_rul"], actual_rul)

        row = {
            "model": name,
            "rmse": rul_metrics.rmse,
            "mae": rul_metrics.mae,
            "nasa_score": rul_metrics.score,
            "mean_bias": rul_metrics.mean_bias,
            "n_valid": rul_metrics.n_valid,
        }

        # Detection metrics if alarm series is available
        if "alarm_series" in result:
            det = compute_detection_metrics(
                result["alarm_series"],
                true_failure_index,
                healthy_end_index,
            )
            row["detection_lead_time"] = det.detection_lead_time
            row["false_alarm_rate"] = det.false_alarm_rate
            row["detection_success"] = det.detection_success

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("rmse", ascending=True).reset_index(drop=True)
    return df
