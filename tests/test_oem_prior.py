"""Tests for OEM prior calculations."""

import numpy as np
import pytest

from src.oem_prior import (
    compute_l10_hours,
    compute_degradation_baseline,
    estimate_bearing_load,
)


class TestL10:
    def test_known_calculation(self):
        """Verify L10 against hand calculation.

        C = 14.8 kN, P = 1.0 kN, rpm = 1797, p = 3
        L10 = (14.8/1.0)^3 = 3241.792 million revolutions
        L10h = (10^6 / (60 * 1797)) * 3241.792 = 30,068.9 hours
        """
        L10h = compute_l10_hours(C_kn=14.8, P_kn=1.0, rpm=1797.0, p=3.0)

        L10_revs = (14.8 / 1.0) ** 3  # 3241.792
        expected = (1e6 / (60 * 1797)) * L10_revs

        assert L10h == pytest.approx(expected, rel=1e-6)

    def test_higher_load_shorter_life(self):
        """Increasing load should decrease life."""
        L10_light = compute_l10_hours(14.8, 0.5, 1797.0)
        L10_heavy = compute_l10_hours(14.8, 2.0, 1797.0)
        assert L10_light > L10_heavy

    def test_higher_speed_shorter_life(self):
        """Increasing speed should decrease life (in hours)."""
        L10_slow = compute_l10_hours(14.8, 1.0, 1000.0)
        L10_fast = compute_l10_hours(14.8, 1.0, 3000.0)
        assert L10_slow > L10_fast

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            compute_l10_hours(14.8, 0.0, 1797.0)
        with pytest.raises(ValueError):
            compute_l10_hours(14.8, 1.0, 0.0)


class TestDegradationBaseline:
    def test_starts_at_zero(self):
        baseline = compute_degradation_baseline(10000, 100)
        assert baseline[0] == pytest.approx(0.0, abs=1e-6)

    def test_ends_at_one(self):
        for shape in ["linear", "exponential"]:
            baseline = compute_degradation_baseline(10000, 100, shape=shape)
            assert baseline[-1] == pytest.approx(1.0, abs=1e-6)

    def test_monotonically_increasing(self):
        for shape in ["linear", "exponential"]:
            baseline = compute_degradation_baseline(10000, 200, shape=shape)
            assert all(np.diff(baseline) >= 0)

    def test_exponential_slower_early(self):
        """Exponential should degrade slower than linear in the first half."""
        n = 200
        linear = compute_degradation_baseline(10000, n, "linear")
        exponential = compute_degradation_baseline(10000, n, "exponential")

        # At 25% of life, exponential should be below linear
        quarter = n // 4
        assert exponential[quarter] < linear[quarter]

    def test_shapes_differ(self):
        linear = compute_degradation_baseline(10000, 100, "linear")
        exponential = compute_degradation_baseline(10000, 100, "exponential")
        assert not np.allclose(linear, exponential)

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            compute_degradation_baseline(10000, 100, "quadratic")


class TestBearingLoad:
    def test_positive_load(self):
        P = estimate_bearing_load(2.0, 1750.0)
        assert P > 0

    def test_higher_hp_higher_load(self):
        P_low = estimate_bearing_load(0.0, 1797.0)
        P_high = estimate_bearing_load(3.0, 1730.0)
        assert P_high > P_low

    def test_reasonable_range(self):
        """Estimated load should be much less than dynamic rating (14.8 kN)."""
        P = estimate_bearing_load(2.0, 1750.0)
        assert P < 1.0  # Should be well under 1 kN for this light-duty rig
