# Predictive Maintenance Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-dataset benchmarking framework that validates the adaptive PID drift model against four public degradation datasets (CWRU, IMS, FEMTO, C-MAPSS) spanning three equipment types and three levels of OEM prior quality.

**Architecture:** Core model code (PID adaptive drift, baselines, evaluation) is copied from the existing `adaptive-predictive-maintenance` repo into a `core/` module. A `framework/` layer provides abstract dataset loading, benchmark orchestration, and cross-dataset comparison. Each dataset gets its own module under `datasets/` implementing the `DatasetLoader` ABC. Results chain through CSVs in `analysis/`.

**Tech Stack:** Python 3.11+, numpy, scipy, pandas, matplotlib, seaborn, sentence-transformers, chromadb, PyMuPDF, h5py, pytest. R Markdown for final report.

**Source repo:** `/Users/christopherbrantner/local/projects/adaptive-drift-forecasting/adaptive-predictive-maintenance/src/`
**Target repo:** `/Users/christopherbrantner/local/projects/adaptive-drift-forecasting/predictive-maintenance-application/`

---

## File Structure

```
predictive-maintenance-application/
  README.md
  INTEGRATION.md
  requirements.txt
  pyproject.toml
  .gitignore

  core/
    __init__.py
    adaptive_drift.py          # PID model + new pid_step() single-step API
    baselines.py               # Static curve, rolling refit, threshold alarm
    regime_predictor.py        # Error-volatility regime detection
    evaluation.py              # RUL metrics, detection metrics, NASA scoring
    oem_prior.py               # Generalized L10/baseline computation

  framework/
    __init__.py
    dataset_loader.py          # OEMPrior, DegradationTrajectory, DatasetLoader ABC
    benchmark_runner.py        # Run all models on all datasets
    results_summary.py         # Cross-dataset tables and plots

  datasets/
    __init__.py
    cwru/
      __init__.py
      config.py
      download.py
      loader.py
      feature_extraction.py
    ims/
      __init__.py
      config.py
      download.py
      loader.py
      feature_extraction.py
    femto/
      __init__.py
      config.py
      download.py
      loader.py
      feature_extraction.py
    cmapss/
      __init__.py
      config.py
      download.py
      loader.py
      feature_extraction.py
    oxford_battery/
      __init__.py
      config.py
      download.py
      loader.py
      feature_extraction.py

  data/
    raw/                       # Downloaded datasets (gitignored)
    oem/                       # OEM documentation (gitignored)
    processed/                 # Extracted feature CSVs (gitignored)

  notebooks/
    01_ims_analysis.ipynb
    02_femto_analysis.ipynb
    03_cmapss_analysis.ipynb
    04_oxford_battery.ipynb
    05_cross_dataset_comparison.ipynb

  reports/
    benchmark_report.Rmd
    figures/

  analysis/                   # All numeric results as CSV

  tests/
    __init__.py
    test_core.py
    test_dataset_loaders.py
    test_benchmark_runner.py
    test_oem_priors.py

  scripts/
    download_all.sh
    run_benchmark.sh
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.gitignore`, all `__init__.py` files

- [ ] **Step 1: Create .gitignore**

```gitignore
data/raw/
data/oem/
data/processed/
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/
dist/
build/
.pytest_cache/
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "predictive-maintenance-benchmark"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "pandas>=2.0",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "h5py>=3.8",
    "sentence-transformers>=2.2",
    "chromadb>=0.4",
    "PyMuPDF>=1.22",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "ruff>=0.1"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create requirements.txt**

```
numpy>=1.24
scipy>=1.10
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
h5py>=3.8
sentence-transformers>=2.2
chromadb>=0.4
PyMuPDF>=1.22
pytest>=7.0
```

- [ ] **Step 4: Create all directory structure and __init__.py files**

Create empty `__init__.py` in: `core/`, `framework/`, `datasets/`, `datasets/cwru/`, `datasets/ims/`, `datasets/femto/`, `datasets/cmapss/`, `datasets/oxford_battery/`, `tests/`.

Create empty directories: `data/raw/`, `data/oem/`, `data/processed/`, `notebooks/`, `reports/figures/`, `analysis/`, `scripts/`.

- [ ] **Step 5: Initialize git and commit**

```bash
cd /Users/christopherbrantner/local/projects/adaptive-drift-forecasting/predictive-maintenance-application
git init
git add .
git commit -m "chore: scaffold project structure"
```

---

## Task 2: Port Core Model Code

**Files:**
- Create: `core/adaptive_drift.py`, `core/baselines.py`, `core/regime_predictor.py`, `core/evaluation.py`, `core/oem_prior.py`
- Create: `tests/test_core.py`

**Source:** `/Users/christopherbrantner/local/projects/adaptive-drift-forecasting/adaptive-predictive-maintenance/src/`

- [ ] **Step 1: Copy and adapt adaptive_drift.py**

Copy from source. Then add the new `PIDState` dataclass and `pid_step()` function for single-step streaming:

```python
@dataclass
class PIDState:
    """Mutable state for incremental PID updates."""
    cumulative_correction: float = 0.0
    error_history: list = field(default_factory=list)
    prev_error: float = 0.0
    regime: str = "normal"
    error_volatility_history: list = field(default_factory=list)
    baseline_volatility: float | None = None
    steps_in_regime: int = 0

def pid_step(observed: float, baseline: float, state: PIDState,
             params: PIDParams = PIDParams(),
             threshold: float = 1.0,
             vol_window: int = 15,
             vol_threshold: float = 2.0,
             accel_gain_multiplier: float = 2.5,
             ) -> tuple[float, float, PIDState]:
    """
    Process a single observation. Returns (adjusted_baseline, predicted_rul, updated_state).
    This is what a real-time system calls per sensor reading.
    """
    # Compute adjusted baseline
    adjusted = baseline + state.cumulative_correction

    # Error
    error = observed - adjusted
    state.error_history.append(error)

    # Regime detection (if enough history)
    gain_mult = 1.0
    if len(state.error_history) >= vol_window:
        recent = state.error_history[-vol_window:]
        vol = float(np.std(recent))
        state.error_volatility_history.append(vol)

        if state.baseline_volatility is None:
            state.baseline_volatility = vol
        elif state.regime == "normal":
            state.baseline_volatility = 0.95 * state.baseline_volatility + 0.05 * vol

        state.steps_in_regime += 1
        if state.baseline_volatility > 0:
            if state.regime == "normal" and vol > vol_threshold * state.baseline_volatility and state.steps_in_regime >= 5:
                state.regime = "accelerated"
                state.steps_in_regime = 0
            elif state.regime == "accelerated" and vol <= vol_threshold * state.baseline_volatility and state.steps_in_regime >= 5:
                state.regime = "normal"
                state.steps_in_regime = 0

        if state.regime == "accelerated":
            gain_mult = accel_gain_multiplier

    # PID terms
    window = state.error_history[-params.integral_window:]
    p_term = params.Kp * gain_mult * error
    i_term = params.Ki * gain_mult * float(np.mean(window))
    d_term = params.Kd * gain_mult * (error - state.prev_error)

    correction = max(-params.clip * gain_mult, min(params.clip * gain_mult, p_term + i_term + d_term))
    state.cumulative_correction += correction
    state.prev_error = error

    adjusted = baseline + state.cumulative_correction

    # RUL prediction (use recent drift rate)
    if len(state.error_history) >= 5:
        recent_vals = [baseline + state.cumulative_correction for _ in range(1)]  # simplified
        # Use rate of change of observed values
        rate = (observed - (state.error_history[-5] + baseline)) / 5 if len(state.error_history) >= 5 else 0
        if rate > 1e-8:
            predicted_rul = max(0, (threshold - observed) / rate)
        else:
            predicted_rul = float('inf')
    else:
        predicted_rul = float('inf')

    return adjusted, predicted_rul, state
```

- [ ] **Step 2: Copy baselines.py**

Copy from source. No modifications needed — the static_degradation_curve, rolling_refit_curve, and threshold_alarm functions work as-is.

- [ ] **Step 3: Copy regime_predictor.py**

Copy from source. No modifications needed.

- [ ] **Step 4: Copy and adapt evaluation.py**

Copy from source. **Critical change:** Cap NASA score penalties at 1e6 per observation:

```python
# In compute_rul_metrics, replace the scoring loop with:
for d_i in d:
    if d_i < 0:  # early prediction
        penalty = np.exp(-d_i / 13.0) - 1
    else:  # late prediction
        penalty = np.exp(d_i / 10.0) - 1
    score += min(penalty, 1e6)  # Cap at 1e6
```

- [ ] **Step 5: Copy and adapt oem_prior.py**

Copy from source. Generalize to handle both ball bearings (p=3) and roller bearings (p=10/3). Ensure `compute_l10_hours` accepts the `p` parameter (it already does). Add JSON export support:

```python
def config_to_json(config: dict, output_path: str | Path) -> None:
    """Export a dataset config dict to JSON."""
    import json
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

def config_from_json(json_path: str | Path) -> dict:
    """Load a dataset config from JSON."""
    import json
    with open(json_path, 'r') as f:
        return json.load(f)
```

- [ ] **Step 6: Write core tests**

```python
# tests/test_core.py
import numpy as np
import pytest
from core.adaptive_drift import PIDParams, adaptive_drift_pid, adaptive_drift_with_regime, PIDState, pid_step
from core.baselines import static_degradation_curve, rolling_refit_curve, threshold_alarm
from core.evaluation import compute_rul_metrics, compute_detection_metrics, compute_actual_rul
from core.oem_prior import compute_l10_hours, compute_degradation_baseline
from core.regime_predictor import detect_regimes
import pandas as pd

class TestAdaptiveDrift:
    def test_no_drift_no_correction(self):
        """When observed matches baseline, corrections should be near zero."""
        baseline = np.linspace(0, 1, 100)
        result = adaptive_drift_pid(baseline, baseline)
        assert np.allclose(result.corrections, 0, atol=0.01)

    def test_constant_offset_tracked(self):
        """PID should track a constant offset."""
        baseline = np.linspace(0, 1, 200)
        observed = baseline + 0.1
        result = adaptive_drift_pid(observed, baseline)
        # By end, adjusted baseline should be close to observed
        assert abs(result.adjusted_baseline[-1] - observed[-1]) < 0.05

    def test_pid_step_matches_batch(self):
        """Single-step API should produce similar results to batch."""
        baseline = np.linspace(0, 1, 100)
        observed = baseline + 0.05 * np.random.randn(100)
        batch_result = adaptive_drift_pid(observed, baseline)

        state = PIDState()
        step_adjustments = []
        for i in range(100):
            adj, rul, state = pid_step(observed[i], baseline[i], state)
            step_adjustments.append(adj)

        # Adjusted baselines should be in the same ballpark
        step_arr = np.array(step_adjustments)
        assert np.corrcoef(batch_result.adjusted_baseline, step_arr)[0, 1] > 0.95

class TestRegimeDetection:
    def test_detects_volatility_spike(self):
        """Should detect regime change when error volatility spikes."""
        errors = np.concatenate([
            np.random.randn(50) * 0.1,   # normal
            np.random.randn(50) * 1.0,   # accelerated
        ])
        result = detect_regimes(errors, vol_window=10, threshold_multiplier=2.0)
        # Should have at least one regime change
        assert len(result.regime_changes) > 0
        # Last regime should be accelerated
        assert result.regimes[-1] == "accelerated"

class TestEvaluation:
    def test_nasa_score_capped(self):
        """NASA score penalties must be capped at 1e6."""
        predicted = np.array([0.0])  # very late
        actual = np.array([1000.0])
        metrics = compute_rul_metrics(predicted, actual)
        assert metrics.score <= 1e6
        assert np.isfinite(metrics.score)

    def test_perfect_prediction(self):
        """Perfect predictions should give zero error."""
        rul = np.array([100.0, 50.0, 10.0, 0.0])
        metrics = compute_rul_metrics(rul, rul)
        assert metrics.rmse == 0.0
        assert metrics.mae == 0.0

class TestOEMPrior:
    def test_l10_ball_bearing(self):
        """L10 for SKF 6205 at CWRU conditions."""
        l10h = compute_l10_hours(C_kn=14.8, P_kn=0.5, rpm=1797, p=3.0)
        assert l10h > 100  # Should be a reasonable number of hours

    def test_l10_roller_bearing(self):
        """L10 for Rexnord ZA-2115 at IMS conditions."""
        l10h = compute_l10_hours(C_kn=128.5, P_kn=26.69, rpm=2000, p=10/3)
        assert 500 < l10h < 5000  # Spec says 500-5000 range

    def test_baseline_monotonic(self):
        """Degradation baseline should be monotonically increasing."""
        baseline = compute_degradation_baseline(l10_hours=1000, n_points=200)
        assert np.all(np.diff(baseline) >= 0)

class TestBaselines:
    def test_threshold_alarm_fires(self):
        """Threshold alarm should fire when feature exceeds threshold."""
        features = pd.DataFrame({"kurtosis": np.concatenate([
            np.ones(50) * 0.1,  # healthy
            np.ones(50) * 2.0,  # degraded
        ])})
        result = threshold_alarm(features, "kurtosis")
        assert result.first_alarm_index is not None
        assert result.first_alarm_index >= 50
```

- [ ] **Step 7: Run tests**

```bash
cd /Users/christopherbrantner/local/projects/adaptive-drift-forecasting/predictive-maintenance-application
python -m pytest tests/test_core.py -v
```
Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add core/ tests/test_core.py
git commit -m "feat: port core model code with single-step PID API"
```

---

## Task 3: Framework Layer — Dataset Loader

**Files:**
- Create: `framework/dataset_loader.py`
- Create: `tests/test_dataset_loaders.py`

- [ ] **Step 1: Write tests for dataclasses and ABC**

```python
# tests/test_dataset_loaders.py
import numpy as np
import pandas as pd
import pytest
from framework.dataset_loader import OEMPrior, DegradationTrajectory, DatasetLoader

class TestOEMPrior:
    def test_creation(self):
        prior = OEMPrior(
            expected_life=1500.0,
            baseline_curve=np.linspace(0, 1, 100),
            threshold=1.0,
            life_unit="hours",
            source="test",
            confidence="exact_oem",
        )
        assert prior.expected_life == 1500.0
        assert prior.confidence == "exact_oem"

    def test_json_roundtrip(self):
        prior = OEMPrior(
            expected_life=1500.0,
            baseline_curve=np.linspace(0, 1, 50),
            threshold=1.0,
            life_unit="hours",
            source="test",
            confidence="exact_oem",
        )
        json_str = prior.to_json()
        restored = OEMPrior.from_json(json_str)
        assert restored.expected_life == 1500.0
        assert len(restored.baseline_curve) == 50

class TestDegradationTrajectory:
    def test_creation(self):
        features = pd.DataFrame({"kurtosis": np.random.randn(100)})
        traj = DegradationTrajectory(
            unit_id="test_1",
            dataset="test",
            features=features,
            primary_feature="kurtosis",
            true_rul=np.linspace(100, 0, 100),
            failure_index=99,
            oem_prior=None,
        )
        assert traj.unit_id == "test_1"
        assert len(traj.features) == 100

class TestDatasetLoaderABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            DatasetLoader()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_dataset_loaders.py -v
```
Expected: FAIL — framework.dataset_loader doesn't exist yet.

- [ ] **Step 3: Implement dataset_loader.py**

```python
# framework/dataset_loader.py
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import numpy as np
import pandas as pd


@dataclass
class OEMPrior:
    """OEM-derived degradation baseline for a single unit."""
    expected_life: float
    baseline_curve: np.ndarray
    threshold: float
    life_unit: str
    source: str
    confidence: str  # "exact_oem", "approximate_oem", "fleet_derived"
    parameters: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = {
            "expected_life": self.expected_life,
            "baseline_curve": self.baseline_curve.tolist(),
            "threshold": self.threshold,
            "life_unit": self.life_unit,
            "source": self.source,
            "confidence": self.confidence,
            "parameters": self.parameters,
        }
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "OEMPrior":
        d = json.loads(json_str)
        d["baseline_curve"] = np.array(d["baseline_curve"])
        return cls(**d)


@dataclass
class DegradationTrajectory:
    """A single unit's degradation trajectory with metadata."""
    unit_id: str
    dataset: str
    features: pd.DataFrame
    primary_feature: str
    true_rul: np.ndarray | None
    failure_index: int | None
    oem_prior: OEMPrior | None
    operating_conditions: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    is_run_to_failure: bool = True

    def append_observation(self, features_dict: dict) -> None:
        """Append a single observation for streaming use."""
        new_row = pd.DataFrame([features_dict])
        self.features = pd.concat([self.features, new_row], ignore_index=True)


class DatasetLoader(ABC):
    """Abstract base class for dataset-specific loaders."""

    @abstractmethod
    def download(self) -> None:
        """Download raw data to data/raw/{dataset_name}/."""

    @abstractmethod
    def load_trajectories(self) -> list[DegradationTrajectory]:
        """Load and return all trajectories with features and OEM priors."""

    @abstractmethod
    def get_dataset_info(self) -> dict:
        """Return metadata: name, equipment type, prior quality, etc."""
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_dataset_loaders.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add framework/dataset_loader.py tests/test_dataset_loaders.py
git commit -m "feat: add framework layer with OEMPrior, DegradationTrajectory, DatasetLoader ABC"
```

---

## Task 4: Framework Layer — Benchmark Runner

**Files:**
- Create: `framework/benchmark_runner.py`
- Create: `tests/test_benchmark_runner.py`

- [ ] **Step 1: Write benchmark runner tests**

```python
# tests/test_benchmark_runner.py
import numpy as np
import pandas as pd
import pytest
from framework.dataset_loader import OEMPrior, DegradationTrajectory
from framework.benchmark_runner import run_single_trajectory

def _make_test_trajectory(n=200, with_rul=True, with_prior=True):
    """Helper to create a synthetic trajectory for testing."""
    features = pd.DataFrame({
        "kurtosis": np.concatenate([
            3.0 + 0.1 * np.random.randn(150),  # healthy
            3.0 + np.linspace(0, 5, 50) + 0.1 * np.random.randn(50),  # degrading
        ])
    })
    prior = None
    if with_prior:
        from core.oem_prior import compute_degradation_baseline
        baseline = compute_degradation_baseline(l10_hours=1000, n_points=n)
        # Scale baseline to match feature range
        feat_min = features["kurtosis"].iloc[:20].mean()
        feat_max = feat_min + 5.0
        scaled_baseline = feat_min + baseline * (feat_max - feat_min)
        prior = OEMPrior(
            expected_life=1000,
            baseline_curve=scaled_baseline,
            threshold=feat_max,
            life_unit="hours",
            source="test",
            confidence="exact_oem",
        )
    return DegradationTrajectory(
        unit_id="test_1",
        dataset="test",
        features=features,
        primary_feature="kurtosis",
        true_rul=np.linspace(n, 0, n) if with_rul else None,
        failure_index=n - 1 if with_rul else None,
        oem_prior=prior,
    )

class TestRunSingleTrajectory:
    def test_returns_all_models(self):
        traj = _make_test_trajectory()
        results = run_single_trajectory(traj)
        models = results["model"].unique()
        assert "threshold_alarm" in models
        assert "static_curve" in models
        assert "pid_adaptive" in models
        assert "pid_regime" in models

    def test_handles_missing_rul(self):
        traj = _make_test_trajectory(with_rul=False)
        results = run_single_trajectory(traj)
        # Should still return results (detection metrics only)
        assert len(results) > 0
        # RUL metrics should be NaN
        assert results["rmse"].isna().all()

    def test_handles_missing_prior(self):
        traj = _make_test_trajectory(with_prior=False)
        results = run_single_trajectory(traj)
        # PID models should be skipped
        models = results["model"].unique()
        assert "threshold_alarm" in models
        # PID models may or may not appear depending on implementation
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_benchmark_runner.py -v
```

- [ ] **Step 3: Implement benchmark_runner.py**

```python
# framework/benchmark_runner.py
"""Run all models on all datasets, collect results."""
import warnings
import numpy as np
import pandas as pd
from framework.dataset_loader import DegradationTrajectory, DatasetLoader
from core.adaptive_drift import adaptive_drift_pid, adaptive_drift_with_regime, PIDParams
from core.baselines import static_degradation_curve, rolling_refit_curve, threshold_alarm
from core.evaluation import compute_rul_metrics, compute_detection_metrics, compute_actual_rul

DEFAULT_MODELS = ["threshold_alarm", "static_curve", "rolling_refit", "pid_adaptive", "pid_regime"]


def run_single_trajectory(trajectory: DegradationTrajectory,
                          models: list[str] | None = None) -> pd.DataFrame:
    """Run all models on one trajectory. Return one-row-per-model DataFrame."""
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
        # Skip PID models if no prior
        if model_name in ("pid_adaptive", "pid_regime") and not has_prior:
            warnings.warn(f"Skipping {model_name} for {trajectory.unit_id}: no OEM prior")
            continue

        # Skip rolling refit on short trajectories
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
                predicted_rul = None  # threshold alarm doesn't predict RUL

            elif model_name == "static_curve":
                result = static_degradation_curve(features, primary,
                    threshold=threshold_val or 1.0)
                predicted_rul = result.predicted_rul

            elif model_name == "rolling_refit":
                result = rolling_refit_curve(features, primary,
                    threshold=threshold_val or 1.0)
                predicted_rul = result.predicted_rul

            elif model_name == "pid_adaptive":
                observed = features[primary].values
                baseline = trajectory.oem_prior.baseline_curve[:n]
                result = adaptive_drift_pid(observed, baseline,
                    threshold=threshold_val or 1.0)
                predicted_rul = result.predicted_rul

            elif model_name == "pid_regime":
                observed = features[primary].values
                baseline = trajectory.oem_prior.baseline_curve[:n]
                result = adaptive_drift_with_regime(observed, baseline,
                    threshold=threshold_val or 1.0)
                predicted_rul = result.predicted_rul

        except Exception as e:
            warnings.warn(f"Error running {model_name} on {trajectory.unit_id}: {e}")
            continue

        # Compute metrics
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
            row.update({
                "rmse": rul_metrics.rmse,
                "mae": rul_metrics.mae,
                "nasa_score": rul_metrics.score,
                "mean_bias": rul_metrics.mean_bias,
            })
        else:
            row.update({"rmse": np.nan, "mae": np.nan, "nasa_score": np.nan, "mean_bias": np.nan})

        # Detection metrics
        if has_failure:
            if alarm_series is not None:
                det = compute_detection_metrics(alarm_series, trajectory.failure_index)
            elif predicted_rul is not None:
                # Create alarm from RUL predictions (alarm when RUL < some threshold)
                alarm = (predicted_rul < 50).astype(float)
                det = compute_detection_metrics(alarm, trajectory.failure_index)
            else:
                det = None

            if det is not None:
                row.update({
                    "detection_lead_time": det.detection_lead_time,
                    "false_alarm_rate": det.false_alarm_rate,
                    "detection_success": det.detection_success,
                })
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
    output_path = f"analysis/{info['name']}_metrics.csv"
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} rows to {output_path}")
    return combined


def run_full_benchmark(datasets: list[str] | None = None) -> pd.DataFrame:
    """Run the complete benchmark across all datasets."""
    from datasets.cwru.loader import CWRULoader
    from datasets.ims.loader import IMSLoader
    from datasets.femto.loader import FEMTOLoader
    from datasets.cmapss.loader import CMAPSSLoader

    if datasets is None:
        datasets = ["cwru", "ims", "femto", "cmapss"]

    loader_map = {
        "cwru": CWRULoader,
        "ims": IMSLoader,
        "femto": FEMTOLoader,
        "cmapss": CMAPSSLoader,
    }

    all_results = []
    for name in datasets:
        if name not in loader_map:
            warnings.warn(f"Unknown dataset: {name}")
            continue
        print(f"\n{'='*60}")
        print(f"Running benchmark on {name.upper()}")
        print(f"{'='*60}")
        try:
            loader = loader_map[name]()
            result = run_dataset(loader)
            all_results.append(result)
        except Exception as e:
            warnings.warn(f"Failed to run {name}: {e}")

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("analysis/cross_dataset_summary.csv", index=False)

    # Print summary table
    summary = combined.groupby(["dataset", "model"]).agg(
        mean_rmse=("rmse", "mean"),
        mean_mae=("mae", "mean"),
        mean_nasa=("nasa_score", "mean"),
    ).round(2)
    print("\n" + summary.to_string())

    return combined


if __name__ == "__main__":
    run_full_benchmark()
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_benchmark_runner.py -v
```

- [ ] **Step 5: Commit**

```bash
git add framework/benchmark_runner.py tests/test_benchmark_runner.py
git commit -m "feat: add benchmark runner with single-trajectory and full-benchmark execution"
```

---

## Task 5: Framework Layer — Results Summary

**Files:**
- Create: `framework/results_summary.py`

- [ ] **Step 1: Implement results_summary.py**

```python
# framework/results_summary.py
"""Cross-dataset comparison tables and plots."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Colorblind-friendly palette
CB_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
              "#D55E00", "#F0E442", "#000000"]


def cross_dataset_table(results: pd.DataFrame) -> pd.DataFrame:
    """Main results table: one row per (dataset, model) with mean metrics."""
    grouped = results.groupby(["dataset", "model"]).agg(
        equipment_type=("equipment_type", "first"),
        prior_quality=("prior_quality", "first"),
        n_trajectories=("unit_id", "nunique"),
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        mean_detection_lead_time=("detection_lead_time", "mean"),
        detection_success_rate=("detection_success", "mean"),
    ).reset_index()
    return grouped.sort_values(["dataset", "mean_rmse"])


def prior_quality_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Group PID+regime results by prior_quality."""
    pid_results = results[results["model"].isin(["pid_adaptive", "pid_regime"])]
    grouped = pid_results.groupby("prior_quality").agg(
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_mae=("mae", "mean"),
        mean_detection_lead_time=("detection_lead_time", "mean"),
        n_trajectories=("unit_id", "nunique"),
    ).reset_index()
    return grouped


def regime_benefit_table(results: pd.DataFrame) -> pd.DataFrame:
    """For each dataset: PID MAE vs PID+regime MAE improvement."""
    pid = results[results["model"] == "pid_adaptive"].groupby("dataset")["mae"].mean()
    regime = results[results["model"] == "pid_regime"].groupby("dataset")["mae"].mean()
    combined = pd.DataFrame({"pid_mae": pid, "pid_regime_mae": regime})
    combined["improvement_pct"] = ((combined["pid_mae"] - combined["pid_regime_mae"])
                                    / combined["pid_mae"] * 100)
    return combined.reset_index()


def plot_cross_dataset_comparison(results: pd.DataFrame, output_dir: str = "reports/figures"):
    """Generate publication-quality cross-dataset comparison figures."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # 1. Grouped bar chart: RMSE by model, grouped by dataset
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = results.groupby(["dataset", "model"])["rmse"].mean().reset_index()
    pivot_wide = pivot.pivot(index="dataset", columns="model", values="rmse")
    pivot_wide.plot(kind="bar", ax=ax, color=CB_PALETTE[:len(pivot_wide.columns)])
    ax.set_ylabel("Mean RMSE (life units)")
    ax.set_xlabel("")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(f"{output_dir}/rmse_by_model_dataset.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Prior quality scatter
    pid_regime = results[results["model"] == "pid_regime"].copy()
    if not pid_regime.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (dataset, group) in enumerate(pid_regime.groupby("dataset")):
            ax.scatter(group["prior_quality"], group["rmse"],
                      color=CB_PALETTE[i], label=dataset, alpha=0.7, s=60)
        ax.set_ylabel("PID + Regime RMSE (life units)")
        ax.set_xlabel("Prior Quality Tier")
        ax.legend(title="Dataset")
        plt.tight_layout()
        fig.savefig(f"{output_dir}/prior_quality_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 3. Regime benefit waterfall
    benefit = regime_benefit_table(results)
    if not benefit.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(benefit))
        width = 0.35
        ax.bar(x - width/2, benefit["pid_mae"], width, label="PID", color=CB_PALETTE[0])
        ax.bar(x + width/2, benefit["pid_regime_mae"], width, label="PID + Regime", color=CB_PALETTE[2])
        ax.set_xticks(x)
        ax.set_xticklabels(benefit["dataset"])
        ax.set_ylabel("Mean Absolute Error (life units)")
        ax.legend()
        for i, row in benefit.iterrows():
            if pd.notna(row["improvement_pct"]):
                ax.annotate(f'{row["improvement_pct"]:.0f}% better',
                           xy=(i, min(row["pid_mae"], row["pid_regime_mae"])),
                           xytext=(0, -20), textcoords="offset points",
                           ha="center", fontsize=9, color=CB_PALETTE[2])
        plt.tight_layout()
        fig.savefig(f"{output_dir}/regime_benefit.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 4. Detection lead time comparison
    det = results.dropna(subset=["detection_lead_time"])
    if not det.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        det_summary = det.groupby(["dataset", "model"])["detection_lead_time"].mean().reset_index()
        det_summary["label"] = det_summary["dataset"] + " / " + det_summary["model"]
        det_summary = det_summary.sort_values("detection_lead_time")
        ax.barh(det_summary["label"], det_summary["detection_lead_time"], color=CB_PALETTE[0])
        ax.set_xlabel("Mean Detection Lead Time (steps before failure)")
        plt.tight_layout()
        fig.savefig(f"{output_dir}/detection_lead_time.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved figures to {output_dir}/")
```

- [ ] **Step 2: Commit**

```bash
git add framework/results_summary.py
git commit -m "feat: add cross-dataset results summary tables and plots"
```

---

## Task 6: CWRU Dataset — Config, Download, Feature Extraction, Loader

**Files:**
- Create: `datasets/cwru/config.py`, `datasets/cwru/download.py`, `datasets/cwru/feature_extraction.py`, `datasets/cwru/loader.py`

- [ ] **Step 1: Create CWRU config**

Use the exact config from BUILD_SPEC.md (CWRU_CONFIG dict with OEM specs, operating conditions, trajectory construction settings).

- [ ] **Step 2: Port download.py**

Port from source `download_cwru.py`. Adapt paths to use `data/raw/cwru/` within this project.

- [ ] **Step 3: Port feature_extraction.py**

Port from source `feature_extraction.py`. Includes `extract_window_features()`, `extract_features()`, `load_cwru_signal()`, and `build_degradation_trajectory()`.

- [ ] **Step 4: Implement CWRULoader**

```python
# datasets/cwru/loader.py
from framework.dataset_loader import DatasetLoader, DegradationTrajectory, OEMPrior
from datasets.cwru.config import CWRU_CONFIG
from datasets.cwru.download import download_cwru_data
from datasets.cwru.feature_extraction import build_degradation_trajectory
from core.oem_prior import compute_l10_hours, compute_degradation_baseline
import numpy as np

class CWRULoader(DatasetLoader):
    def __init__(self, data_dir: str = "data/raw/cwru"):
        self.data_dir = data_dir
        self.config = CWRU_CONFIG

    def download(self) -> None:
        download_cwru_data(output_dir=self.data_dir, subset="minimal")

    def load_trajectories(self) -> list[DegradationTrajectory]:
        features = build_degradation_trajectory(data_dir=self.data_dir)
        n = len(features)
        primary = self.config["trajectory_construction"]["primary_feature"]

        # Compute OEM prior
        specs = self.config["oem_specs"]
        ops = self.config["operating_conditions"]
        # Estimate load for CWRU test rig
        from core.oem_prior import estimate_bearing_load
        P_kn = estimate_bearing_load(motor_hp=2, rpm=ops["rpm"])
        l10h = compute_l10_hours(specs["C_kn"], P_kn, ops["rpm"], specs["life_exponent"])
        baseline = compute_degradation_baseline(l10h, n)

        # Scale baseline to feature range
        feat_vals = features[primary].values
        healthy_mean = np.mean(feat_vals[:int(n * 0.1)])
        feat_max = healthy_mean + 5 * np.std(feat_vals[:int(n * 0.1)])
        if feat_max <= healthy_mean:
            feat_max = healthy_mean + 1.0
        scaled_baseline = healthy_mean + baseline * (feat_max - healthy_mean)

        prior = OEMPrior(
            expected_life=l10h,
            baseline_curve=scaled_baseline,
            threshold=feat_max,
            life_unit="hours",
            source=f"SKF {specs['designation']} catalog",
            confidence="exact_oem",
            parameters=dict(specs),
        )

        # CWRU is synthetic: failure_index = last step
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
```

- [ ] **Step 5: Test CWRU loader (requires data download)**

```bash
python -c "
from datasets.cwru.loader import CWRULoader
loader = CWRULoader()
loader.download()
trajs = loader.load_trajectories()
print(f'Loaded {len(trajs)} trajectories')
print(f'First trajectory: {trajs[0].unit_id}, {len(trajs[0].features)} steps')
print(f'OEM prior L10: {trajs[0].oem_prior.expected_life:.0f} hours')
"
```

- [ ] **Step 6: Run CWRU through benchmark runner**

```bash
python -c "
from datasets.cwru.loader import CWRULoader
from framework.benchmark_runner import run_dataset
results = run_dataset(CWRULoader())
print(results[['model', 'rmse', 'mae', 'nasa_score']].to_string())
"
```

Verify `analysis/cwru_metrics.csv` is created and numbers are reasonable.

- [ ] **Step 7: Commit**

```bash
git add datasets/cwru/ analysis/cwru_metrics.csv
git commit -m "feat: add CWRU dataset loader and run initial benchmark"
```

---

## Task 7: IMS Dataset — Config, Download, Feature Extraction

**Files:**
- Create: `datasets/ims/config.py`, `datasets/ims/download.py`, `datasets/ims/feature_extraction.py`

- [ ] **Step 1: Create IMS config**

Use the exact IMS_CONFIG from BUILD_SPEC.md (Rexnord ZA-2115 specs, experiment details, feature settings).

- [ ] **Step 2: Create IMS download**

```python
# datasets/ims/download.py
"""Download IMS bearing dataset from NASA data portal."""
import urllib.request
import zipfile
import os
from pathlib import Path

DOWNLOAD_URLS = [
    "https://data.nasa.gov/download/brfb-gzcv/application%2Fx-zip-compressed",
]

def download_ims_data(output_dir: str = "data/raw/ims") -> None:
    """Download and extract IMS dataset."""
    output_path = Path(output_dir)
    if (output_path / "1st_test").exists():
        print("IMS data already downloaded.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    zip_path = output_path / "ims_bearing.zip"

    for url in DOWNLOAD_URLS:
        try:
            print(f"Downloading IMS dataset from {url}...")
            urllib.request.urlretrieve(url, zip_path)
            break
        except Exception as e:
            print(f"Failed: {e}")
            continue
    else:
        raise RuntimeError("Could not download IMS dataset from any source.")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(output_path)
    zip_path.unlink()
    print(f"IMS data extracted to {output_path}")
```

- [ ] **Step 3: Create IMS feature extraction**

```python
# datasets/ims/feature_extraction.py
"""IMS-specific vibration feature extraction."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

def compute_defect_frequencies(rpm: float = 2000) -> dict:
    """Compute defect frequencies for Rexnord ZA-2115."""
    n_rollers = 16
    d_roller = 0.331   # inches
    d_pitch = 2.815    # inches
    alpha = np.radians(15.17)
    shaft_freq = rpm / 60.0

    bpfo = (n_rollers / 2) * (1 - (d_roller / d_pitch) * np.cos(alpha)) * shaft_freq
    bpfi = (n_rollers / 2) * (1 + (d_roller / d_pitch) * np.cos(alpha)) * shaft_freq
    bsf = (d_pitch / (2 * d_roller)) * (1 - ((d_roller / d_pitch) * np.cos(alpha))**2) * shaft_freq
    ftf = 0.5 * (1 - (d_roller / d_pitch) * np.cos(alpha)) * shaft_freq

    return {"bpfo": bpfo, "bpfi": bpfi, "bsf": bsf, "ftf": ftf}


def compute_spectral_energy(signal: np.ndarray, sr: float,
                             center_freq: float, bandwidth: float = 5.0,
                             n_harmonics: int = 3) -> float:
    """Compute spectral energy around a defect frequency and its harmonics."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    fft_mag = np.abs(np.fft.rfft(signal)) ** 2

    total_energy = 0.0
    for h in range(1, n_harmonics + 1):
        f_center = center_freq * h
        mask = (freqs >= f_center - bandwidth) & (freqs <= f_center + bandwidth)
        total_energy += np.sum(fft_mag[mask])
    return total_energy


def extract_ims_features(snapshot: np.ndarray, sr: float = 20000,
                          defect_freqs: dict | None = None) -> dict:
    """Extract features from a single 1-second IMS vibration snapshot."""
    features = {
        "rms": float(np.sqrt(np.mean(snapshot ** 2))),
        "peak": float(np.max(np.abs(snapshot))),
        "kurtosis": float(stats.kurtosis(snapshot, fisher=True) + 3),  # excess -> regular
        "skewness": float(stats.skew(snapshot)),
        "crest_factor": float(np.max(np.abs(snapshot)) / np.sqrt(np.mean(snapshot ** 2))) if np.mean(snapshot ** 2) > 0 else 0,
        "peak_to_peak": float(np.max(snapshot) - np.min(snapshot)),
    }

    if defect_freqs:
        for name, freq in defect_freqs.items():
            features[f"{name}_energy"] = compute_spectral_energy(snapshot, sr, freq)

    return features


def process_ims_experiment(data_dir: str, experiment: str,
                            defect_freqs: dict | None = None,
                            sr: float = 20000) -> pd.DataFrame:
    """Process all snapshots from one IMS experiment into a feature DataFrame."""
    exp_dir = Path(data_dir) / experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # IMS files are named with timestamps
    files = sorted(exp_dir.iterdir())
    files = [f for f in files if f.is_file() and not f.name.startswith('.')]

    if defect_freqs is None:
        defect_freqs = compute_defect_frequencies()

    # Determine number of channels
    n_channels = 8 if experiment == "1st_test" else 4
    n_bearings = 4

    all_features = {b: [] for b in range(n_bearings)}
    timestamps = []
    skipped = 0

    for i, fpath in enumerate(files):
        if i % 500 == 0:
            print(f"  Processing {experiment}: file {i+1}/{len(files)}")
        try:
            data = np.loadtxt(fpath, delimiter='\t')
            if data.ndim == 1:
                data = np.loadtxt(fpath)  # try space delimiter
            if data.ndim == 1:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        timestamps.append(fpath.name)

        for bearing_idx in range(n_bearings):
            if experiment == "1st_test":
                # 2 channels per bearing
                ch1 = bearing_idx * 2
                channel_data = data[:, ch1]
            else:
                channel_data = data[:, bearing_idx]

            feats = extract_ims_features(channel_data, sr, defect_freqs)
            all_features[bearing_idx].append(feats)

    if skipped > 0:
        print(f"  Skipped {skipped} corrupt/unreadable files")

    # Build DataFrames per bearing
    result_dfs = {}
    for bearing_idx in range(n_bearings):
        df = pd.DataFrame(all_features[bearing_idx])
        # Compute time in hours (10-minute intervals)
        df["time_hours"] = np.arange(len(df)) * (10.0 / 60.0)
        df = df.set_index("time_hours")
        result_dfs[f"bearing{bearing_idx + 1}"] = df

    return result_dfs
```

- [ ] **Step 4: Commit**

```bash
git add datasets/ims/config.py datasets/ims/download.py datasets/ims/feature_extraction.py
git commit -m "feat: add IMS config, download, and feature extraction"
```

---

## Task 8: IMS Dataset — Loader

**Files:**
- Create: `datasets/ims/loader.py`
- Create: `tests/test_oem_priors.py`

- [ ] **Step 1: Write OEM prior tests**

```python
# tests/test_oem_priors.py
import pytest
import numpy as np
from core.oem_prior import compute_l10_hours

class TestIMSOEMPrior:
    def test_ims_l10_range(self):
        """IMS L10 life should be 500-5000 hours at 26.69 kN load."""
        l10h = compute_l10_hours(C_kn=128.5, P_kn=26.69, rpm=2000, p=10/3)
        assert 500 < l10h < 5000

    def test_ims_l10_value(self):
        """Verify approximate IMS L10 computation."""
        # L10 = (128.5/26.69)^(10/3) million rev
        # = 4.816^3.333 ≈ 145 million rev
        # L10h = 145e6 / (60*2000) ≈ 1208 hours
        l10h = compute_l10_hours(C_kn=128.5, P_kn=26.69, rpm=2000, p=10/3)
        assert 800 < l10h < 2000

class TestFEMTOOEMPrior:
    def test_femto_c1_l10(self):
        """FEMTO condition 1 L10 should be reasonable for accelerated test."""
        l10h = compute_l10_hours(C_kn=12.7, P_kn=4.0, rpm=1800, p=3.0)
        assert l10h > 0
```

- [ ] **Step 2: Run OEM tests**

```bash
python -m pytest tests/test_oem_priors.py -v
```

- [ ] **Step 3: Implement IMSLoader**

```python
# datasets/ims/loader.py
"""IMS dataset loader."""
import numpy as np
import pandas as pd
from pathlib import Path
from framework.dataset_loader import DatasetLoader, DegradationTrajectory, OEMPrior
from datasets.ims.config import IMS_CONFIG
from datasets.ims.download import download_ims_data
from datasets.ims.feature_extraction import process_ims_experiment, compute_defect_frequencies
from core.oem_prior import compute_l10_hours, compute_degradation_baseline


class IMSLoader(DatasetLoader):
    def __init__(self, data_dir: str = "data/raw/ims",
                 processed_dir: str = "data/processed"):
        self.data_dir = data_dir
        self.processed_dir = Path(processed_dir)
        self.config = IMS_CONFIG

    def download(self) -> None:
        download_ims_data(self.data_dir)

    def load_trajectories(self) -> list[DegradationTrajectory]:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        defect_freqs = compute_defect_frequencies()
        trajectories = []

        experiments = {
            "1st_test": self.config["experiments"]["set1"],
            "2nd_test": self.config["experiments"]["set2"],
            "3rd_test": self.config["experiments"]["set3"],
        }

        for exp_name, exp_info in experiments.items():
            set_num = exp_name[0]  # "1", "2", "3"

            # Check for cached features
            cache_path = self.processed_dir / f"ims_set{set_num}_features.csv"
            bearing_dfs = self._load_or_extract(exp_name, cache_path, defect_freqs)

            for bearing_name, df in bearing_dfs.items():
                bearing_num = bearing_name.replace("bearing", "")
                unit_id = f"ims_set{set_num}_{bearing_name}"

                # Determine if this bearing failed
                failure_mode = exp_info["failures"].get(bearing_name)
                is_failed = failure_mode is not None
                failure_index = len(df) - 1 if is_failed else None

                # Compute RUL for failed bearings
                true_rul = None
                if is_failed:
                    n = len(df)
                    total_hours = df.index[-1] if isinstance(df.index[0], float) else n * (10/60)
                    true_rul = np.array([total_hours - t for t in df.index])

                # Compute OEM prior
                prior = self._compute_oem_prior(df)

                traj = DegradationTrajectory(
                    unit_id=unit_id,
                    dataset="ims",
                    features=df.reset_index(),
                    primary_feature=self.config["feature_settings"]["primary_feature"],
                    true_rul=true_rul,
                    failure_index=failure_index,
                    oem_prior=prior,
                    operating_conditions=dict(self.config["operating_conditions"]),
                    metadata={
                        "equipment_type": self.config["equipment_type"],
                        "experiment": exp_name,
                        "failure_mode": failure_mode,
                    },
                    is_run_to_failure=is_failed,
                )
                trajectories.append(traj)

        return trajectories

    def _load_or_extract(self, experiment: str, cache_path: Path,
                          defect_freqs: dict) -> dict[str, pd.DataFrame]:
        """Load cached features or extract from raw data."""
        set_num = experiment[0]

        # Check for per-bearing cache files
        cached_bearings = {}
        all_cached = True
        for b in range(1, 5):
            bp = self.processed_dir / f"ims_set{set_num}_bearing{b}.csv"
            if bp.exists():
                cached_bearings[f"bearing{b}"] = pd.read_csv(bp, index_col=0)
            else:
                all_cached = False

        if all_cached and cached_bearings:
            print(f"Loading cached features for {experiment}")
            return cached_bearings

        # Extract features from raw data
        print(f"Extracting features for {experiment}...")
        bearing_dfs = process_ims_experiment(self.data_dir, experiment, defect_freqs)

        # Cache
        for name, df in bearing_dfs.items():
            bp = self.processed_dir / f"ims_set{set_num}_{name}.csv"
            df.to_csv(bp)
            print(f"  Cached {bp}")

        return bearing_dfs

    def _compute_oem_prior(self, features_df: pd.DataFrame) -> OEMPrior:
        """Compute OEM prior for Rexnord ZA-2115 at IMS conditions."""
        specs = self.config["oem_specs"]
        ops = self.config["operating_conditions"]

        l10h = compute_l10_hours(
            C_kn=specs["C_kn"],
            P_kn=ops["radial_load_kn"],
            rpm=ops["rpm"],
            p=specs["life_exponent"],
        )

        n = len(features_df)
        baseline = compute_degradation_baseline(l10h, n)

        primary = self.config["feature_settings"]["primary_feature"]
        feat_vals = features_df[primary].values
        healthy_n = max(1, int(n * 0.1))
        healthy_mean = np.mean(feat_vals[:healthy_n])
        healthy_std = np.std(feat_vals[:healthy_n])
        threshold = healthy_mean + 5 * healthy_std
        if threshold <= healthy_mean:
            threshold = healthy_mean + 1.0

        scaled_baseline = healthy_mean + baseline * (threshold - healthy_mean)

        return OEMPrior(
            expected_life=l10h,
            baseline_curve=scaled_baseline,
            threshold=threshold,
            life_unit="hours",
            source=f"Rexnord {specs['designation']} catalog",
            confidence="exact_oem",
            parameters=dict(specs),
        )

    def get_dataset_info(self) -> dict:
        return {
            "name": "ims",
            "equipment": self.config["equipment"],
            "equipment_type": self.config["equipment_type"],
            "prior_quality": self.config["prior_quality"],
            "n_trajectories": 12,  # 4 bearings x 3 experiments
        }
```

- [ ] **Step 4: Commit**

```bash
git add datasets/ims/loader.py tests/test_oem_priors.py
git commit -m "feat: add IMS dataset loader with OEM prior computation"
```

---

## Task 9: IMS Notebook (01_ims_analysis.ipynb)

**Files:**
- Create: `notebooks/01_ims_analysis.ipynb`

This is the flagship notebook — most detail.

- [ ] **Step 1: Create notebook with all analysis sections**

Sections:
1. **Dataset Overview**: 3 experiments, durations, which bearings failed, failure modes
2. **Raw Feature Trajectories**: Plot kurtosis over time for all bearings in each experiment. Failed bearings show degradation trend; healthy bearings stay flat.
3. **OEM Prior Computation**: L10 calculation step-by-step for Rexnord ZA-2115, plot expected baseline
4. **Baseline Overlay**: Plot OEM baseline vs actual kurtosis for each failed bearing
5. **Model Comparison**: Run all 5 models per failed bearing. Plot predicted RUL vs actual RUL. Show PID tracking with regime activation shading.
6. **Results Table**: Per-bearing metrics, then aggregate
7. **Regime Analysis**: When did regime predictor activate? Hours of warning before failure.
8. **CWRU Comparison**: Side-by-side table of synthetic vs real results.

Use colorblind-friendly palette, clean axis labels with units, no default titles.

- [ ] **Step 2: Run notebook and verify outputs**

```bash
cd /Users/christopherbrantner/local/projects/adaptive-drift-forecasting/predictive-maintenance-application
python -m jupyter nbconvert --execute notebooks/01_ims_analysis.ipynb --to notebook
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_ims_analysis.ipynb analysis/ims_metrics.csv
git commit -m "feat: add IMS flagship analysis notebook"
```

---

## Task 10: FEMTO Dataset

**Files:**
- Create: `datasets/femto/config.py`, `datasets/femto/download.py`, `datasets/femto/feature_extraction.py`, `datasets/femto/loader.py`
- Create: `notebooks/02_femto_analysis.ipynb`

- [ ] **Step 1: Create FEMTO config**

Use FEMTO_CONFIG from BUILD_SPEC.md (approximate SKF 6204 specs, 3 operating conditions, feature settings).

- [ ] **Step 2: Create FEMTO download**

```python
# datasets/femto/download.py
import subprocess
from pathlib import Path

def download_femto_data(output_dir: str = "data/raw/femto") -> None:
    """Download FEMTO/PRONOSTIA dataset from GitHub."""
    output_path = Path(output_dir)
    if (output_path / "Learning_set").exists():
        print("FEMTO data already downloaded.")
        return
    output_path.mkdir(parents=True, exist_ok=True)
    repo_url = "https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset.git"
    print(f"Cloning FEMTO dataset from {repo_url}...")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(output_path)], check=True)
    print(f"FEMTO data saved to {output_path}")
```

- [ ] **Step 3: Create FEMTO feature extraction**

Extract per-recording features from 2-axis vibration CSVs. Primary feature: kurtosis (horizontal).

- [ ] **Step 4: Create FEMTOLoader**

Implements DatasetLoader. One trajectory per bearing. Approximate OEM prior using SKF 6204 specs per operating condition. Confidence = "approximate_oem".

- [ ] **Step 5: Run FEMTO benchmark**

```bash
python -c "
from datasets.femto.loader import FEMTOLoader
from framework.benchmark_runner import run_dataset
results = run_dataset(FEMTOLoader())
print(results.groupby('model')[['rmse','mae']].mean().round(2))
"
```

Verify `analysis/femto_metrics.csv` is created.

- [ ] **Step 6: Create FEMTO notebook**

Focus on: aggregate results across 17 trajectories, per-condition comparison, approximate vs exact prior comparison (vs IMS).

- [ ] **Step 7: Commit**

```bash
git add datasets/femto/ notebooks/02_femto_analysis.ipynb analysis/femto_metrics.csv
git commit -m "feat: add FEMTO dataset with approximate OEM priors"
```

---

## Task 11: C-MAPSS Dataset

**Files:**
- Create: `datasets/cmapss/config.py`, `datasets/cmapss/download.py`, `datasets/cmapss/feature_extraction.py`, `datasets/cmapss/loader.py`
- Create: `notebooks/03_cmapss_analysis.ipynb`

- [ ] **Step 1: Create C-MAPSS config**

Use CMAPSS_CONFIG from BUILD_SPEC.md (sub-datasets, sensor columns, informative sensors, RUL cap).

- [ ] **Step 2: Create C-MAPSS download**

Download from NASA or Kaggle. Four text files per sub-dataset.

- [ ] **Step 3: Create C-MAPSS feature extraction**

```python
# datasets/cmapss/feature_extraction.py
"""C-MAPSS health index construction."""
import numpy as np
import pandas as pd

def load_cmapss_data(data_dir: str, sub_dataset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load train, test, and RUL ground truth."""
    col_names = ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"] + \
                [f"s_{i}" for i in range(1, 22)]

    train = pd.read_csv(f"{data_dir}/train_{sub_dataset}.txt",
                        sep=r"\s+", header=None, names=col_names)
    test = pd.read_csv(f"{data_dir}/test_{sub_dataset}.txt",
                       sep=r"\s+", header=None, names=col_names)
    rul = np.loadtxt(f"{data_dir}/RUL_{sub_dataset}.txt")
    return train, test, rul


def compute_health_index(train: pd.DataFrame, test: pd.DataFrame,
                          informative_sensors: list[int],
                          rul_cap: int = 125) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct health index from sensor readings."""
    sensor_cols = [f"s_{i}" for i in informative_sensors]

    # Fit scaler on training data only
    train_min = train[sensor_cols].min()
    train_max = train[sensor_cols].max()
    denom = train_max - train_min
    denom[denom == 0] = 1e-8

    for df in [train, test]:
        df[sensor_cols] = (df[sensor_cols] - train_min) / denom
        # Smooth per engine
        for col in sensor_cols:
            df[col] = df.groupby("unit_nr")[col].transform(
                lambda x: x.rolling(5, min_periods=1).mean())
        df["health_index"] = df[sensor_cols].mean(axis=1)

    # RUL labels for training
    for unit in train["unit_nr"].unique():
        mask = train["unit_nr"] == unit
        max_cycle = train.loc[mask, "time_cycles"].max()
        train.loc[mask, "rul"] = max_cycle - train.loc[mask, "time_cycles"]
    train["rul"] = train["rul"].clip(upper=rul_cap)

    return train, test


def compute_fleet_prior(train_df: pd.DataFrame) -> np.ndarray:
    """Compute fleet-average health_index trajectory from training engines."""
    reversed_series = []
    for unit in train_df["unit_nr"].unique():
        hi = train_df[train_df["unit_nr"] == unit]["health_index"].values
        reversed_series.append(hi[::-1])

    max_len = max(len(s) for s in reversed_series)
    padded = np.array([
        np.pad(s, (0, max_len - len(s)), mode="edge")
        for s in reversed_series
    ])
    fleet_avg_reversed = np.nanmean(padded, axis=0)
    return fleet_avg_reversed[::-1]
```

- [ ] **Step 4: Create CMAPSSLoader**

Implements DatasetLoader. One trajectory per engine. Fleet-derived prior from training set. Confidence = "fleet_derived".

- [ ] **Step 5: Run C-MAPSS benchmark (FD001)**

```bash
python -c "
from datasets.cmapss.loader import CMAPSSLoader
from framework.benchmark_runner import run_dataset
results = run_dataset(CMAPSSLoader('FD001'))
print(results.groupby('model')[['rmse','mae','nasa_score']].mean().round(2))
"
```

- [ ] **Step 6: Create C-MAPSS notebook**

Include: dataset overview, fleet prior visualization, example engine tracking, test set RMSE/S-Score, comparison to published methods (LSTM, CNN-LSTM, Transformer from BUILD_SPEC table).

- [ ] **Step 7: Commit**

```bash
git add datasets/cmapss/ notebooks/03_cmapss_analysis.ipynb analysis/cmapss_metrics.csv
git commit -m "feat: add C-MAPSS FD001 with fleet-derived priors"
```

---

## Task 12: Cross-Dataset Comparison

**Files:**
- Create: `notebooks/05_cross_dataset_comparison.ipynb`

- [ ] **Step 1: Create cross-dataset comparison notebook**

Sections:
1. **Main table**: Load all `analysis/*_metrics.csv`. Build cross_dataset_table().
2. **Prior quality analysis**: Box plot of RMSE by prior quality tier. Statistical test.
3. **Regime benefit by dataset**: Bar chart of PID vs PID+regime improvement.
4. **Real vs synthetic**: CWRU vs IMS comparison.
5. **Equipment type comparison**: Bearings vs turbofan.
6. **"No training" argument**: PID uses zero training data. Where does it compete?

- [ ] **Step 2: Generate all figures**

Run `plot_cross_dataset_comparison()` from results_summary.py. Verify all 4 figures saved to `reports/figures/`.

- [ ] **Step 3: Save summary CSVs**

Generate and save:
- `analysis/cross_dataset_summary.csv`
- `analysis/prior_quality_comparison.csv`

- [ ] **Step 4: Commit**

```bash
git add notebooks/05_cross_dataset_comparison.ipynb analysis/ reports/figures/
git commit -m "feat: add cross-dataset comparison notebook and figures"
```

---

## Task 13: Oxford Battery (Stretch Goal)

**Files:**
- Create: `datasets/oxford_battery/config.py`, `datasets/oxford_battery/download.py`, `datasets/oxford_battery/feature_extraction.py`, `datasets/oxford_battery/loader.py`
- Create: `notebooks/04_oxford_battery.ipynb`

Only implement if Tasks 1-12 are working cleanly.

- [ ] **Step 1: Create config, download, feature extraction, loader**

Use OXFORD_BATTERY_CONFIG from BUILD_SPEC.md. Primary feature: SOH (capacity/initial_capacity). Linear fade prior. The regime predictor should catch the capacity "knee".

- [ ] **Step 2: Create notebook**

Focus on: knee detection, regime predictor activation vs actual knee onset.

- [ ] **Step 3: Run and commit**

```bash
git add datasets/oxford_battery/ notebooks/04_oxford_battery.ipynb
git commit -m "feat: add Oxford battery dataset (stretch goal)"
```

---

## Task 14: Benchmark Report (R Markdown)

**Files:**
- Create: `reports/benchmark_report.Rmd`

- [ ] **Step 1: Write R Markdown report**

```yaml
---
title: "Multi-Dataset Benchmark: Adaptive PID Drift Forecasting"
author: ""
date: "`r Sys.Date()`"
output:
  prettydoc::html_pretty:
    theme: tactile
    highlight: github
---
```

Sections per BUILD_SPEC.md spec:
1. Introduction (why multi-dataset, three prior tiers)
2. Datasets (one paragraph each)
3. IMS Results (headline — most detail)
4. FEMTO Results
5. C-MAPSS Results
6. Battery Results (if completed)
7. Cross-Dataset Comparison (main tables and figures)
8. Discussion (what generalizes, practical implications)
9. Future Work (real-time, fleet priors, desktop app)

Use `=` for assignment. Read CSVs from `../analysis/`. Narrative style, not bullet points.

- [ ] **Step 2: Render and verify**

```bash
Rscript -e "rmarkdown::render('reports/benchmark_report.Rmd')"
```

- [ ] **Step 3: Commit**

```bash
git add reports/benchmark_report.Rmd
git commit -m "feat: add benchmark report in R Markdown"
```

---

## Task 15: Final Deliverables

**Files:**
- Create: `INTEGRATION.md`, `README.md`
- Create: `scripts/download_all.sh`, `scripts/run_benchmark.sh`

- [ ] **Step 1: Write INTEGRATION.md**

Document app-facing interfaces per BUILD_SPEC.md: OEMPrior (JSON serializable), PIDState + pid_step() (O(1) per observation), DegradationTrajectory.append_observation(), config-as-JSON.

- [ ] **Step 2: Write README.md**

Lead with finding, not methodology. Include cross-dataset results table. Link to original repo.

- [ ] **Step 3: Create shell scripts**

```bash
# scripts/download_all.sh
#!/usr/bin/env bash
set -e
echo "Downloading all datasets..."
python -c "from datasets.cwru.download import download_cwru_data; download_cwru_data()"
python -c "from datasets.ims.download import download_ims_data; download_ims_data()"
python -c "from datasets.femto.download import download_femto_data; download_femto_data()"
python -c "from datasets.cmapss.download import download_cmapss_data; download_cmapss_data()"
echo "Done."

# scripts/run_benchmark.sh
#!/usr/bin/env bash
set -e
echo "Running full benchmark..."
python -m framework.benchmark_runner
echo "Done. Results in analysis/"
```

- [ ] **Step 4: Commit**

```bash
git add INTEGRATION.md README.md scripts/
git commit -m "docs: add integration guide, README, and convenience scripts"
```

---

## Task 16: Full Integration Test

- [ ] **Step 1: Run full benchmark end-to-end**

```bash
bash scripts/download_all.sh
bash scripts/run_benchmark.sh
```

- [ ] **Step 2: Verify all outputs exist**

```bash
ls analysis/*.csv
ls reports/figures/*.png
```

Expected files:
- `analysis/cwru_metrics.csv`
- `analysis/ims_metrics.csv`
- `analysis/femto_metrics.csv`
- `analysis/cmapss_metrics.csv`
- `analysis/cross_dataset_summary.csv`
- `analysis/prior_quality_comparison.csv`
- `reports/figures/rmse_by_model_dataset.png`
- `reports/figures/prior_quality_scatter.png`
- `reports/figures/regime_benefit.png`
- `reports/figures/detection_lead_time.png`

- [ ] **Step 3: Run all tests**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: complete multi-dataset benchmark framework"
```
