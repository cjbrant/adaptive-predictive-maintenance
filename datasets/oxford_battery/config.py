OXFORD_BATTERY_CONFIG = {
    "name": "oxford_battery",
    "equipment": "740 mAh Li-ion pouch cell",
    "equipment_type": "lithium_ion_battery",
    "prior_quality": "approximate_oem",
    "is_run_to_failure": True,

    "cell_specs": {
        "nominal_capacity_mah": 740,
        "estimated_cycle_life_80pct": 500,
        "charge_protocol": "CC-CV, 1.48A to 4.2V, cutoff 100mA",
        "discharge_protocol": "Artemis urban drive cycle",
        "temperature_C": 40,
        "n_cells": 8,
        "characterization_interval": 100,
    },

    "feature_settings": {
        "primary_feature": "soh",
        "features": ["capacity_mah", "soh", "internal_resistance"],
    },

    "prior_settings": {
        "baseline_model": "linear_fade",
        # Linear capacity fade from 100% to 80% SOH over rated_cycle_life
    },

    "phase_thresholds": {
        "linear_fade": 0.85,
        "knee_region": 0.75,
        # linear_fade: SOH > 85%, knee_region: 85% > SOH > 75%, rapid_fade: SOH < 75%
    }
}
