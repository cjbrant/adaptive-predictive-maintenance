FEMTO_CONFIG = {
    "name": "femto",
    "equipment": "Small bearing (model not documented in public literature)",
    "equipment_type": "ball_bearing",
    "prior_quality": "approximate_oem",
    "is_run_to_failure": True,

    "approximate_oem_specs": {
        "comparable_model": "SKF 6204-2RS (approximate match based on load capacity)",
        "estimated_C_kn": 13.5,
        "estimated_C0_kn": 6.55,
        "life_exponent": 3.0,
        "note": "Bearing model not documented. Using SKF 6204 open (C=13.5 kN per SKF catalog) as approximate match."
    },

    "conditions": {
        1: {"rpm": 1800, "radial_load_N": 4000, "n_training": 2, "n_test": 5},
        2: {"rpm": 1650, "radial_load_N": 4200, "n_training": 2, "n_test": 5},
        3: {"rpm": 1500, "radial_load_N": 5000, "n_training": 0, "n_test": 1},
    },

    "data_settings": {
        "vibration_sr_hz": 25600,
        "vibration_samples_per_record": 2560,
        "record_interval_sec": 10,
        "failure_threshold_g": 20,
        "channels": ["horizontal_accel", "vertical_accel"],
    },

    "feature_settings": {
        "primary_feature": "kurtosis",
        "features": ["rms", "kurtosis", "crest_factor", "peak_to_peak",
                     "rms_vertical", "kurtosis_vertical"],
    }
}
