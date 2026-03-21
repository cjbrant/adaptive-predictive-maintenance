"""PID adaptive drift model for bearing degradation tracking.

Ported from the R implementation in the adaptive-drift-forecasting project.
The model tracks deviation between observed degradation and the OEM baseline,
using PID feedback control to adapt the baseline to the individual bearing's
actual behavior.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class PIDParams:
    """PID controller parameters for adaptive drift tracking."""

    Kp: float = 0.08  # Proportional gain
    Ki: float = 0.02  # Integral gain
    Kd: float = 0.03  # Derivative gain
    integral_window: int = 10  # Lookback window for integral term
    clip: float = 0.05  # Max absolute correction per step


@dataclass
class AdaptiveDriftResult:
    """Results from adaptive drift model run."""

    adjusted_baseline: np.ndarray
    corrections: np.ndarray
    cumulative_correction: np.ndarray
    errors: np.ndarray
    p_terms: np.ndarray
    i_terms: np.ndarray
    d_terms: np.ndarray
    predicted_rul: np.ndarray
    observed: np.ndarray
    original_baseline: np.ndarray


def _clip_value(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def adaptive_drift_pid(
    observed: np.ndarray,
    baseline: np.ndarray,
    params: PIDParams | None = None,
    threshold: float = 1.0,
) -> AdaptiveDriftResult:
    """
    Track deviation between observed degradation and OEM baseline using PID
    feedback control.

    At each time step t:
      error_t = observed_t - adjusted_baseline_t
      P_t = Kp * error_t
      I_t = Ki * mean(error_history[max(0, t-window):t+1])
      D_t = Kd * (error_t - error_{t-1})
      correction_t = clip(P_t + I_t + D_t, -clip, +clip)
      adjusted_baseline_{t+1} = baseline_{t+1} + cumulative_corrections

    The adjusted baseline is the model's real-time estimate of this specific
    bearing's degradation, adapted from the OEM prediction.

    Parameters
    ----------
    observed : observed degradation feature values (normalized to [0, 1+])
    baseline : OEM expected degradation curve (same length as observed)
    params : PID controller parameters
    threshold : health index value at which bearing needs replacement

    Returns
    -------
    AdaptiveDriftResult with all tracking arrays
    """
    if params is None:
        params = PIDParams()

    n = len(observed)
    if len(baseline) != n:
        raise ValueError(f"observed ({n}) and baseline ({len(baseline)}) must have same length")

    adjusted = np.copy(baseline).astype(float)
    corrections = np.zeros(n)
    cumulative = np.zeros(n)
    errors = np.zeros(n)
    p_terms = np.zeros(n)
    i_terms = np.zeros(n)
    d_terms = np.zeros(n)
    predicted_rul = np.full(n, np.nan)

    error_history = []
    cum_correction = 0.0

    for t in range(n):
        # Current adjusted baseline includes all prior corrections
        adjusted[t] = baseline[t] + cum_correction

        # Error: positive means degrading faster than expected
        e_t = observed[t] - adjusted[t]
        errors[t] = e_t
        error_history.append(e_t)

        # PID terms
        P_t = params.Kp * e_t

        int_start = max(0, len(error_history) - params.integral_window)
        I_t = params.Ki * np.mean(error_history[int_start:])

        D_t = params.Kd * (e_t - error_history[-2]) if len(error_history) >= 2 else 0.0

        p_terms[t] = P_t
        i_terms[t] = I_t
        d_terms[t] = D_t

        raw_correction = P_t + I_t + D_t
        clipped = _clip_value(raw_correction, -params.clip, params.clip)
        corrections[t] = clipped
        cum_correction += clipped
        cumulative[t] = cum_correction

        # Predict RUL: extrapolate adjusted baseline to find when it hits threshold
        if t >= 2:
            # Use recent drift rate to project forward
            recent_rate = (adjusted[t] + cum_correction - adjusted[max(0, t - 5)]) / min(t, 5)
            if recent_rate > 1e-8:
                remaining = (threshold - (adjusted[t] + clipped)) / recent_rate
                predicted_rul[t] = max(0, remaining)
            else:
                predicted_rul[t] = n - t  # Default: remaining time steps

    return AdaptiveDriftResult(
        adjusted_baseline=adjusted,
        corrections=corrections,
        cumulative_correction=cumulative,
        errors=errors,
        p_terms=p_terms,
        i_terms=i_terms,
        d_terms=d_terms,
        predicted_rul=predicted_rul,
        observed=observed,
        original_baseline=baseline,
    )


@dataclass
class RegimeState:
    """Tracks the current degradation regime."""

    regime: str = "normal"  # "normal" or "accelerated"
    volatility: float = 0.0
    switch_count: int = 0


def adaptive_drift_with_regime(
    observed: np.ndarray,
    baseline: np.ndarray,
    pid_params: PIDParams | None = None,
    threshold: float = 1.0,
    vol_window: int = 15,
    vol_threshold: float = 2.0,
    accel_gain_multiplier: float = 2.5,
) -> AdaptiveDriftResult:
    """
    Enhanced PID with volatility-based regime detection.

    Monitors the trailing standard deviation of the PID error signal.
    When error volatility exceeds a threshold (relative to its historical
    average), switches to "accelerated degradation" regime with increased
    PID gains for faster response.

    This adapts the model's responsiveness: smooth tracking during normal
    wear, aggressive tracking during damage propagation.

    Parameters
    ----------
    observed : observed degradation values
    baseline : OEM expected degradation curve
    pid_params : base PID parameters (gains are scaled by regime)
    threshold : replacement threshold
    vol_window : trailing window for volatility computation
    vol_threshold : multiple of baseline volatility to trigger regime switch
    accel_gain_multiplier : factor to multiply PID gains in accelerated regime
    """
    if pid_params is None:
        pid_params = PIDParams()

    n = len(observed)
    if len(baseline) != n:
        raise ValueError("observed and baseline must have same length")

    adjusted = np.copy(baseline).astype(float)
    corrections = np.zeros(n)
    cumulative = np.zeros(n)
    errors = np.zeros(n)
    p_terms = np.zeros(n)
    i_terms = np.zeros(n)
    d_terms = np.zeros(n)
    predicted_rul = np.full(n, np.nan)
    regimes = []

    error_history = []
    cum_correction = 0.0
    regime = "normal"
    baseline_vol = None

    for t in range(n):
        adjusted[t] = baseline[t] + cum_correction
        e_t = observed[t] - adjusted[t]
        errors[t] = e_t
        error_history.append(e_t)

        # Regime detection via error volatility
        if len(error_history) >= vol_window:
            current_vol = float(np.std(error_history[-vol_window:]))
            if baseline_vol is None:
                baseline_vol = current_vol if current_vol > 1e-8 else 1e-8
            else:
                # Slowly update baseline volatility during normal regime
                if regime == "normal":
                    baseline_vol = 0.95 * baseline_vol + 0.05 * current_vol

            if current_vol > vol_threshold * baseline_vol:
                regime = "accelerated"
            else:
                regime = "normal"

        regimes.append(regime)

        # Scale gains by regime
        gain_mult = accel_gain_multiplier if regime == "accelerated" else 1.0
        Kp = pid_params.Kp * gain_mult
        Ki = pid_params.Ki * gain_mult
        Kd = pid_params.Kd * gain_mult
        clip_val = pid_params.clip * gain_mult

        P_t = Kp * e_t

        int_start = max(0, len(error_history) - pid_params.integral_window)
        I_t = Ki * np.mean(error_history[int_start:])

        D_t = Kd * (e_t - error_history[-2]) if len(error_history) >= 2 else 0.0

        p_terms[t] = P_t
        i_terms[t] = I_t
        d_terms[t] = D_t

        raw_correction = P_t + I_t + D_t
        clipped = _clip_value(raw_correction, -clip_val, clip_val)
        corrections[t] = clipped
        cum_correction += clipped
        cumulative[t] = cum_correction

        # RUL prediction
        if t >= 2:
            lookback = min(t, 5)
            recent_rate = (adjusted[t] + cum_correction - adjusted[t - lookback]) / lookback
            if recent_rate > 1e-8:
                remaining = (threshold - (adjusted[t] + clipped)) / recent_rate
                predicted_rul[t] = max(0, remaining)
            else:
                predicted_rul[t] = n - t

    result = AdaptiveDriftResult(
        adjusted_baseline=adjusted,
        corrections=corrections,
        cumulative_correction=cumulative,
        errors=errors,
        p_terms=p_terms,
        i_terms=i_terms,
        d_terms=d_terms,
        predicted_rul=predicted_rul,
        observed=observed,
        original_baseline=baseline,
    )
    # Attach regime info as extra attribute
    result.regimes = np.array(regimes)  # type: ignore[attr-defined]
    return result
