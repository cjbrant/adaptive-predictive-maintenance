"""OEM prior calculations for bearing life prediction.

Implements ISO 281 L10 bearing life calculations and generates expected
degradation baseline curves from OEM specifications. These priors serve
as the starting point that the adaptive drift model adjusts from.

Can load parameters from the RAG extraction output
(analysis/extracted_oem_params.json) or accept them directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def compute_l10_hours(C_kn: float, P_kn: float, rpm: float, p: float = 3.0) -> float:
    """
    Compute L10 bearing life in hours using ISO 281 formula.

    L10 = (C/P)^p million revolutions
    L10h = (10^6 / (60 * rpm)) * L10

    Parameters
    ----------
    C_kn : basic dynamic load rating in kN
    P_kn : equivalent dynamic bearing load in kN
    rpm : rotational speed in RPM
    p : life exponent (3.0 for ball bearings, 10/3 for roller bearings)

    Returns
    -------
    L10 life in hours (90% reliability)
    """
    if P_kn <= 0 or rpm <= 0:
        raise ValueError("P and rpm must be positive")
    L10_revs = (C_kn / P_kn) ** p  # million revolutions
    L10h = (1e6 / (60.0 * rpm)) * L10_revs
    return L10h


def compute_degradation_baseline(
    l10_hours: float,
    n_points: int,
    shape: str = "exponential",
) -> np.ndarray:
    """
    Generate the OEM expected degradation curve.

    The bearing starts at health_index = 0 (healthy) and reaches
    health_index = 1.0 (replacement threshold) at L10 hours.

    Parameters
    ----------
    l10_hours : predicted L10 bearing life in hours
    n_points : number of time points to generate
    shape : "linear" or "exponential"

    Returns
    -------
    Array of expected health index values in [0, 1] at each time point.
    """
    t = np.linspace(0, 1, n_points)  # normalized time [0, 1] where 1 = L10

    if shape == "linear":
        baseline = t.copy()
    elif shape == "exponential":
        # Slow degradation early, accelerating toward end of life.
        # Calibrated so baseline(0) = 0 and baseline(1) = 1.
        alpha = 3.0  # controls curvature
        baseline = (np.exp(alpha * t) - 1) / (np.exp(alpha) - 1)
    else:
        raise ValueError(f"Unknown shape: {shape}. Use 'linear' or 'exponential'.")

    return baseline


def estimate_bearing_load(
    motor_hp: float,
    rpm: float,
    shaft_weight_kg: float = 2.0,
) -> float:
    """
    Estimate the equivalent dynamic bearing load P for the CWRU test rig.

    The CWRU test rig applies motor load via a dynamometer that primarily
    affects torque, not radial force. The dominant radial load on the
    drive-end bearing comes from shaft weight and any residual misalignment.

    This is a rough estimate for plugging into L10; precise values would
    require the actual rig's mechanical drawings.

    Parameters
    ----------
    motor_hp : motor load in horsepower
    rpm : shaft speed in RPM
    shaft_weight_kg : estimated shaft/rotor weight supported by this bearing

    Returns
    -------
    Estimated equivalent dynamic bearing load in kN.
    """
    # Radial load from shaft weight (gravity)
    g = 9.81  # m/s^2
    # Assume roughly half the shaft weight on the drive-end bearing
    radial_load_n = shaft_weight_kg * g * 0.5

    # Small additional dynamic load from motor torque coupling
    # Torque = Power / angular_velocity
    power_w = motor_hp * 745.7  # HP to watts
    omega = 2.0 * np.pi * rpm / 60.0
    torque_nm = power_w / omega if omega > 0 else 0.0

    # A fraction of torque reaction contributes to radial load
    # This is a rough engineering estimate
    radial_from_torque = torque_nm * 0.05  # small fraction

    total_radial_n = radial_load_n + radial_from_torque
    P_kn = total_radial_n / 1000.0

    return P_kn


def compute_time_axis_hours(n_points: int, l10_hours: float) -> np.ndarray:
    """Generate a time axis in hours from 0 to L10."""
    return np.linspace(0, l10_hours, n_points)


def load_extracted_params(
    designation: str = "6205",
    json_path: str | Path = "analysis/extracted_oem_params.json",
) -> dict:
    """
    Load bearing parameters from the RAG extraction output.

    If the JSON doesn't exist, runs the full extraction pipeline first.
    Validates that the extracted dynamic load rating C is within 10% of
    the known ground truth.

    Returns the bearing parameter dict from the JSON file.
    """
    json_path = Path(json_path)

    if not json_path.exists():
        print(f"{json_path} not found — running extraction pipeline...")
        from rag.extract_params import run_full_extraction
        run_full_extraction()

    with open(json_path) as f:
        data = json.load(f)

    key = f"bearing_{designation}"
    if key not in data:
        raise KeyError(f"No parameters for {designation} in {json_path}")

    params = data[key]

    # Validate dynamic load rating
    ground_truth_C = {"6205": 14.8, "6203": 9.95, "6204": 13.5, "ZA-2115": 90.3, "UER204": 12.0}
    if designation in ground_truth_C:
        C_extracted = params.get("dynamic_load_rating_kn", 0)
        C_expected = ground_truth_C[designation]
        if abs(C_extracted - C_expected) / C_expected > 0.10:
            print(
                f"WARNING: Extracted C={C_extracted} kN for {designation} deviates "
                f">10% from known value {C_expected} kN. Check extraction pipeline."
            )

    return params


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
