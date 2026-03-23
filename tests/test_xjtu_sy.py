import numpy as np
import pytest

class TestXJTUSYFeatureExtraction:
    def test_compute_defect_frequencies(self):
        from datasets.xjtu_sy.feature_extraction import compute_defect_frequencies
        freqs = compute_defect_frequencies(rpm=2100)
        assert freqs["bpfo"] > 0
        assert freqs["bpfi"] > freqs["bpfo"]
        assert freqs["ftf"] < freqs["bsf"]

    def test_extract_features_shape(self):
        from datasets.xjtu_sy.feature_extraction import extract_xjtu_features
        np.random.seed(42)
        snapshot = np.random.randn(32768)
        features = extract_xjtu_features(snapshot, sr=25600)
        assert "rms" in features
        assert "kurtosis" in features
        assert "crest_factor" in features
        assert features["rms"] > 0

    def test_extract_features_with_defect_freqs(self):
        from datasets.xjtu_sy.feature_extraction import (
            extract_xjtu_features, compute_defect_frequencies
        )
        np.random.seed(42)
        snapshot = np.random.randn(32768)
        freqs = compute_defect_frequencies(rpm=2100)
        features = extract_xjtu_features(snapshot, sr=25600, defect_freqs=freqs)
        assert "bpfo_energy" in features
        assert "bpfi_energy" in features

class TestXJTUSYConfig:
    def test_config_structure(self):
        from datasets.xjtu_sy.config import XJTU_SY_CONFIG
        assert XJTU_SY_CONFIG["name"] == "xjtu_sy"
        assert len(XJTU_SY_CONFIG["bearing_failures"]) == 15
        assert len(XJTU_SY_CONFIG["conditions"]) == 3


class TestXJTUSYLoader:
    def test_loader_instantiates(self):
        from datasets.xjtu_sy.loader import XJTUSYLoader
        loader = XJTUSYLoader()
        assert loader.config["name"] == "xjtu_sy"

    def test_dataset_info(self):
        from datasets.xjtu_sy.loader import XJTUSYLoader
        loader = XJTUSYLoader()
        info = loader.get_dataset_info()
        assert info["name"] == "xjtu_sy"
        assert info["prior_quality"] == "exact_oem"
        assert info["n_trajectories"] == 15

    def test_compute_oem_prior(self):
        from datasets.xjtu_sy.loader import XJTUSYLoader
        import pandas as pd
        loader = XJTUSYLoader()
        df = pd.DataFrame({"kurtosis": np.random.randn(100) * 0.1 + 3.0})
        prior = loader._compute_oem_prior(df, condition=1)
        assert prior.expected_life > 0
        assert prior.confidence == "exact_oem"
        assert len(prior.baseline_curve) == 100

    def test_l10_values_per_condition(self):
        from core.oem_prior import compute_l10_hours
        l10_1 = compute_l10_hours(C_kn=12.0, P_kn=12.0, rpm=2100, p=3.0)
        assert 5 < l10_1 < 10
        l10_2 = compute_l10_hours(C_kn=12.0, P_kn=11.0, rpm=2250, p=3.0)
        assert 7 < l10_2 < 12
        l10_3 = compute_l10_hours(C_kn=12.0, P_kn=10.0, rpm=2400, p=3.0)
        assert 10 < l10_3 < 15
