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
