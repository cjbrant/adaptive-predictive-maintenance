"""Baseline models for degradation prediction.

Provides non-adaptive approaches to compare against the PID adaptive drift model:
static curve fitting, rolling refit, and simple threshold alarms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import curve_fit


@dataclass
class StaticCurveResult:
    """Results from static degradation curve fit."""

    fitted_curve: np.ndarray
    predicted_rul: np.ndarray
    params: dict


@dataclass
class RollingRefitResult:
    """Results from rolling refit model."""

    fitted_curve: np.ndarray
    predicted_rul: np.ndarray
    refit_indices: list[int]


@dataclass
class ThresholdAlarmResult:
    """Results from threshold alarm model."""

    alarm_series: np.ndarray
    first_alarm_index: int | None
    threshold_crossings: dict[str, int | None]


def _exp_curve(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """Exponential degradation: a * exp(b * t) - a."""
    return a * (np.exp(b * t) - 1.0)


def _fit_exp_curve(
    t: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    """Fit exponential curve, returning (a, b) parameters."""
    try:
        popt, _ = curve_fit(
            _exp_curve,
            t,
            y,
            p0=[0.01, 0.5],
            bounds=([1e-6, 1e-4], [10.0, 10.0]),
            maxfev=5000,
        )
        return float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        # Fallback to linear fit
        if len(t) > 1 and t[-1] > t[0]:
            slope = (y[-1] - y[0]) / (t[-1] - t[0])
            return slope, 0.0
        return 0.01, 0.1


def static_degradation_curve(
    features: pd.DataFrame,
    feature_col: str,
    fit_window: int | None = None,
    threshold: float = 1.0,
) -> StaticCurveResult:
    """
    Fit a static exponential degradation curve to observed features.

    No online adaptation. Optionally fits only on the first fit_window
    observations, then extrapolates.

    Parameters
    ----------
    features : DataFrame with a time_index and the feature column
    feature_col : column name to fit
    fit_window : if set, only use first N observations for fitting
    threshold : replacement threshold value
    """
    y = features[feature_col].values.astype(float)
    n = len(y)
    t = np.arange(n, dtype=float)

    # Normalize to [0, 1] range for fitting
    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-8:
        y_max = y_min + 1.0
    y_norm = (y - y_min) / (y_max - y_min)

    # Fit on subset or all data
    fit_n = fit_window if fit_window is not None else n
    fit_n = min(fit_n, n)
    t_fit = t[:fit_n] / n  # normalize time to [0, 1]
    y_fit = y_norm[:fit_n]

    a, b = _fit_exp_curve(t_fit, y_fit)

    # Generate fitted curve for all time points
    t_all = t / n
    fitted_norm = _exp_curve(t_all, a, b)
    fitted = fitted_norm * (y_max - y_min) + y_min

    # Predict RUL at each step
    predicted_rul = np.full(n, np.nan)
    threshold_val = threshold * (y_max - y_min) + y_min
    for i in range(n):
        remaining_t = t_all[i:]
        remaining_fitted = _exp_curve(remaining_t, a, b) * (y_max - y_min) + y_min
        crossings = np.where(remaining_fitted >= threshold_val)[0]
        if len(crossings) > 0:
            predicted_rul[i] = crossings[0]
        else:
            predicted_rul[i] = n - i

    return StaticCurveResult(
        fitted_curve=fitted,
        predicted_rul=predicted_rul,
        params={"a": a, "b": b, "y_min": y_min, "y_max": y_max},
    )


def rolling_refit_curve(
    features: pd.DataFrame,
    feature_col: str,
    window: int = 50,
    refit_every: int = 10,
    threshold: float = 1.0,
) -> RollingRefitResult:
    """
    Refit degradation curve on a trailing window periodically.

    Parameters
    ----------
    features : DataFrame with feature values
    feature_col : column to model
    window : trailing window size for fitting
    refit_every : refit every N observations
    threshold : replacement threshold
    """
    y = features[feature_col].values.astype(float)
    n = len(y)
    t = np.arange(n, dtype=float)

    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-8:
        y_max = y_min + 1.0
    y_norm = (y - y_min) / (y_max - y_min)

    fitted = np.full(n, np.nan)
    predicted_rul = np.full(n, np.nan)
    refit_indices = []
    current_a, current_b = 0.01, 0.1

    for i in range(n):
        if i >= window and (i % refit_every == 0 or i == window):
            # Refit on trailing window
            t_win = t[max(0, i - window) : i + 1]
            y_win = y_norm[max(0, i - window) : i + 1]
            t_win_norm = (t_win - t_win[0]) / max(t_win[-1] - t_win[0], 1.0)
            current_a, current_b = _fit_exp_curve(t_win_norm, y_win)
            refit_indices.append(i)

        # Current prediction
        if i < window:
            fitted[i] = y_norm[i] * (y_max - y_min) + y_min
            predicted_rul[i] = n - i
        else:
            t_pred = (t[i] - t[max(0, i - window)]) / max(window, 1.0)
            fitted_val = _exp_curve(np.array([t_pred]), current_a, current_b)[0]
            fitted[i] = fitted_val * (y_max - y_min) + y_min

            # Project RUL
            threshold_norm = (threshold * (y_max - y_min) + y_min - y_min) / (y_max - y_min)
            future_t = np.linspace(t_pred, t_pred + 2.0, 200)
            future_vals = _exp_curve(future_t, current_a, current_b)
            crossings = np.where(future_vals >= threshold_norm)[0]
            if len(crossings) > 0:
                # Convert from normalized time to steps
                steps_to_threshold = int(crossings[0] * window / 2.0)
                predicted_rul[i] = max(0, steps_to_threshold)
            else:
                predicted_rul[i] = n - i

    return RollingRefitResult(
        fitted_curve=fitted,
        predicted_rul=predicted_rul,
        refit_indices=refit_indices,
    )


def threshold_alarm(
    features: pd.DataFrame,
    feature_col: str,
    thresholds: dict[str, float] | None = None,
) -> ThresholdAlarmResult:
    """
    Simple threshold-based alarm. No prediction; purely reactive.

    Parameters
    ----------
    features : DataFrame with feature values
    feature_col : column to monitor
    thresholds : dict with keys "warning", "alert", "danger" mapping to values.
                 Defaults to ISO 10816 acceleration thresholds.
    """
    if thresholds is None:
        thresholds = {
            "warning": 0.5,
            "alert": 1.0,
            "danger": 2.0,
        }

    y = features[feature_col].values.astype(float)
    n = len(y)

    # Alarm when any threshold is exceeded
    alarm_threshold = thresholds.get("alert", thresholds.get("warning", 1.0))
    alarm_series = (y >= alarm_threshold).astype(int)

    # Find first crossing for each threshold level
    crossings = {}
    for level, value in thresholds.items():
        above = np.where(y >= value)[0]
        crossings[level] = int(above[0]) if len(above) > 0 else None

    first_alarm = np.where(alarm_series > 0)[0]
    first_alarm_idx = int(first_alarm[0]) if len(first_alarm) > 0 else None

    return ThresholdAlarmResult(
        alarm_series=alarm_series,
        first_alarm_index=first_alarm_idx,
        threshold_crossings=crossings,
    )
