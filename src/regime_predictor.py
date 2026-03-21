"""Volatility regime predictor for degradation phases.

Detects transitions between normal wear and accelerated degradation
using the volatility of the error signal from the adaptive drift model.
Adapted from the financial volatility regime predictor for the
predictive maintenance domain.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class RegimeResult:
    """Results from regime prediction."""

    regimes: np.ndarray  # "normal" or "accelerated" at each step
    volatilities: np.ndarray  # trailing error volatility
    regime_changes: list[int]  # indices where regime changed
    regime_durations: list[tuple[str, int, int]]  # (regime, start, end)


def detect_regimes(
    errors: np.ndarray,
    vol_window: int = 15,
    threshold_multiplier: float = 2.0,
    min_regime_length: int = 5,
) -> RegimeResult:
    """
    Detect degradation regimes from the PID error signal.

    Computes trailing standard deviation of errors. When current volatility
    exceeds `threshold_multiplier` times the baseline volatility, labels
    the regime as "accelerated".

    Parameters
    ----------
    errors : PID error signal (observed - adjusted baseline)
    vol_window : window for trailing volatility computation
    threshold_multiplier : multiple of baseline vol to trigger regime switch
    min_regime_length : minimum steps before allowing a regime switch (debounce)
    """
    n = len(errors)
    volatilities = np.full(n, np.nan)
    regimes = np.array(["normal"] * n, dtype="U12")
    regime_changes = []

    baseline_vol = None
    current_regime = "normal"
    steps_in_regime = 0

    for t in range(n):
        if t >= vol_window - 1:
            vol = float(np.std(errors[t - vol_window + 1 : t + 1]))
            volatilities[t] = vol

            if baseline_vol is None:
                baseline_vol = vol if vol > 1e-8 else 1e-8
            elif current_regime == "normal":
                # Slowly update baseline during normal regime
                baseline_vol = 0.95 * baseline_vol + 0.05 * vol

            # Regime detection with debounce
            steps_in_regime += 1
            if steps_in_regime >= min_regime_length:
                if vol > threshold_multiplier * baseline_vol and current_regime == "normal":
                    current_regime = "accelerated"
                    regime_changes.append(t)
                    steps_in_regime = 0
                elif vol <= threshold_multiplier * baseline_vol and current_regime == "accelerated":
                    current_regime = "normal"
                    regime_changes.append(t)
                    steps_in_regime = 0

        regimes[t] = current_regime

    # Compute regime durations
    durations = []
    if n > 0:
        start = 0
        for i in range(1, n):
            if regimes[i] != regimes[i - 1]:
                durations.append((regimes[i - 1], start, i - 1))
                start = i
        durations.append((regimes[-1], start, n - 1))

    return RegimeResult(
        regimes=regimes,
        volatilities=volatilities,
        regime_changes=regime_changes,
        regime_durations=durations,
    )
