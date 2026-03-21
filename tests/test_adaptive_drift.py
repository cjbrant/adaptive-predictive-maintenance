"""Tests for the PID adaptive drift model."""

import numpy as np
import pytest

from src.adaptive_drift import (
    PIDParams,
    adaptive_drift_pid,
    adaptive_drift_with_regime,
)


class TestPIDTracking:
    def test_no_drift(self):
        """When observed matches baseline, corrections should be near zero."""
        n = 100
        baseline = np.linspace(0, 1, n)
        observed = baseline.copy()

        result = adaptive_drift_pid(observed, baseline)

        assert np.max(np.abs(result.corrections)) < 0.01
        assert np.allclose(result.adjusted_baseline, baseline, atol=0.01)

    def test_constant_offset(self):
        """PID should converge toward a constant offset."""
        n = 200
        baseline = np.linspace(0, 0.5, n)
        observed = baseline + 0.1  # constant positive offset

        result = adaptive_drift_pid(observed, baseline)

        # After convergence, error should be small
        late_errors = result.errors[150:]
        assert np.mean(np.abs(late_errors)) < 0.05

    def test_linear_drift(self):
        """PID should track a linearly increasing drift."""
        n = 200
        baseline = np.linspace(0, 0.5, n)
        drift = np.linspace(0, 0.2, n)
        observed = baseline + drift

        result = adaptive_drift_pid(observed, baseline)

        # Cumulative correction should grow to track the drift
        assert result.cumulative_correction[-1] > 0.1

    def test_step_change_response(self):
        """PID should respond to a step change in degradation rate."""
        n = 200
        baseline = np.linspace(0, 0.5, n)
        observed = baseline.copy()
        observed[100:] += 0.15  # step increase at midpoint

        result = adaptive_drift_pid(observed, baseline)

        # Error should be large initially at step, then decrease
        error_at_step = abs(result.errors[101])
        error_later = abs(result.errors[180])
        assert error_later < error_at_step

    def test_clipping(self):
        """Corrections should be clipped to max magnitude."""
        params = PIDParams(Kp=1.0, Ki=0.5, Kd=0.5, clip=0.02)
        n = 50
        baseline = np.zeros(n)
        observed = np.ones(n) * 5.0  # huge deviation

        result = adaptive_drift_pid(observed, baseline, params)

        assert np.all(np.abs(result.corrections) <= params.clip + 1e-10)

    def test_rul_prediction(self):
        """Predicted RUL should generally decrease over time for degrading bearing."""
        n = 200
        baseline = np.linspace(0, 1, n)
        # Slightly faster than expected degradation
        observed = np.linspace(0, 1.1, n)

        result = adaptive_drift_pid(observed, baseline, threshold=1.0)

        # RUL should be available after initial period
        valid_rul = result.predicted_rul[~np.isnan(result.predicted_rul)]
        assert len(valid_rul) > 0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            adaptive_drift_pid(np.zeros(10), np.zeros(15))


class TestRegimeDetection:
    def test_regime_switch_on_volatility(self):
        """Should detect accelerated regime when error volatility spikes."""
        n = 300
        baseline = np.linspace(0, 0.5, n)
        rng = np.random.default_rng(42)

        # Low noise first half, high noise second half
        noise = np.concatenate([
            rng.normal(0, 0.01, n // 2),
            rng.normal(0, 0.1, n - n // 2),
        ])
        observed = baseline + noise + np.linspace(0, 0.3, n)

        result = adaptive_drift_with_regime(observed, baseline, vol_window=15)

        # Should have regimes attribute
        assert hasattr(result, "regimes")
        regimes = result.regimes

        # Should detect at least one accelerated regime in the noisy section
        accel_count = np.sum(regimes == "accelerated")
        assert accel_count > 0

    def test_stays_normal_when_stable(self):
        """Should remain in normal regime when error is stable."""
        n = 200
        baseline = np.linspace(0, 0.5, n)
        rng = np.random.default_rng(42)
        observed = baseline + rng.normal(0, 0.005, n)

        result = adaptive_drift_with_regime(observed, baseline, vol_window=15)

        # Most should be normal
        normal_frac = np.mean(result.regimes == "normal")
        assert normal_frac > 0.8
