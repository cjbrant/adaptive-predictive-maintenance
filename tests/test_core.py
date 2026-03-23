import numpy as np
import pytest
import pandas as pd
from core.adaptive_drift import PIDParams, adaptive_drift_pid, adaptive_drift_with_regime, PIDState, pid_step
from core.baselines import static_degradation_curve, rolling_refit_curve, threshold_alarm
from core.evaluation import compute_rul_metrics, compute_detection_metrics, compute_actual_rul
from core.oem_prior import compute_l10_hours, compute_degradation_baseline
from core.regime_predictor import detect_regimes

class TestAdaptiveDrift:
    def test_no_drift_no_correction(self):
        baseline = np.linspace(0, 1, 100)
        result = adaptive_drift_pid(baseline, baseline)
        assert np.allclose(result.corrections, 0, atol=0.01)

    def test_constant_offset_tracked(self):
        baseline = np.linspace(0, 1, 200)
        observed = baseline + 0.1
        result = adaptive_drift_pid(observed, baseline)
        assert abs(result.adjusted_baseline[-1] - observed[-1]) < 0.05

    def test_pid_step_single_step(self):
        state = PIDState()
        adj, rul, state = pid_step(0.5, 0.4, state)
        assert adj != 0.4  # should have some correction
        assert isinstance(state, PIDState)

class TestRegimeDetection:
    def test_detects_volatility_spike(self):
        np.random.seed(42)
        errors = np.concatenate([
            np.random.randn(50) * 0.1,
            np.random.randn(50) * 1.0,
        ])
        result = detect_regimes(errors, vol_window=10, threshold_multiplier=2.0)
        assert len(result.regime_changes) > 0
        assert result.regimes[-1] == "accelerated"

class TestEvaluation:
    def test_nasa_score_capped(self):
        predicted = np.array([0.0])
        actual = np.array([1000.0])
        metrics = compute_rul_metrics(predicted, actual)
        assert metrics.score <= 1e6
        assert np.isfinite(metrics.score)

    def test_perfect_prediction(self):
        rul = np.array([100.0, 50.0, 10.0, 0.0])
        metrics = compute_rul_metrics(rul, rul)
        assert metrics.rmse == 0.0
        assert metrics.mae == 0.0

class TestOEMPrior:
    def test_l10_ball_bearing(self):
        l10h = compute_l10_hours(C_kn=14.8, P_kn=0.5, rpm=1797, p=3.0)
        assert l10h > 100

    def test_l10_roller_bearing(self):
        l10h = compute_l10_hours(C_kn=90.3, P_kn=26.69, rpm=2000, p=10/3)
        assert 50 < l10h < 2000

    def test_baseline_monotonic(self):
        baseline = compute_degradation_baseline(l10_hours=1000, n_points=200)
        assert np.all(np.diff(baseline) >= 0)

class TestBaselines:
    def test_threshold_alarm_fires(self):
        features = pd.DataFrame({"kurtosis": np.concatenate([
            np.ones(50) * 0.1,
            np.ones(50) * 2.0,
        ])})
        result = threshold_alarm(features, "kurtosis")
        assert result.first_alarm_index is not None
        assert result.first_alarm_index >= 50
