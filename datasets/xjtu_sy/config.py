XJTU_SY_CONFIG = {
    "name": "xjtu_sy",
    "equipment": "LDK UER204 deep groove ball bearing",
    "equipment_type": "ball_bearing",
    "prior_quality": "exact_oem",
    "is_run_to_failure": True,

    "oem_specs": {
        "designation": "LDK UER204",
        "bore_mm": 20.0,
        "C_kn": 12.0,
        "C0_kn": 3.0,
        "life_exponent": 3.0,
        "n_balls": 8,
        "pitch_diameter_mm": 34.55,
        "ball_diameter_mm": 7.92,
        "contact_angle_deg": 0.0,
        "outer_race_diameter_mm": 39.80,
        "inner_race_diameter_mm": 29.30,
    },

    "conditions": {
        1: {"rpm": 2100, "radial_load_kn": 12.0, "n_bearings": 5, "dir_name": "35Hz12kN"},
        2: {"rpm": 2250, "radial_load_kn": 11.0, "n_bearings": 5, "dir_name": "37.5Hz11kN"},
        3: {"rpm": 2400, "radial_load_kn": 10.0, "n_bearings": 5, "dir_name": "40Hz10kN"},
    },

    "data_settings": {
        "sampling_rate_hz": 25600,
        "samples_per_snapshot": 32768,
        "snapshot_interval_sec": 60,
        "channels": ["horizontal", "vertical"],
    },

    "bearing_failures": {
        "Bearing1_1": {"failure": "outer_race", "life_min": 123},
        "Bearing1_2": {"failure": "outer_race", "life_min": 161},
        "Bearing1_3": {"failure": "outer_race", "life_min": 158},
        "Bearing1_4": {"failure": "cage", "life_min": 122},
        "Bearing1_5": {"failure": "inner_race+roller", "life_min": 52},
        "Bearing2_1": {"failure": "inner_race", "life_min": 491},
        "Bearing2_2": {"failure": "outer_race", "life_min": 161},
        "Bearing2_3": {"failure": "cage", "life_min": 533},
        "Bearing2_4": {"failure": "outer_race", "life_min": 42},
        "Bearing2_5": {"failure": "outer_race", "life_min": 339},
        "Bearing3_1": {"failure": "outer_race", "life_min": 2538},
        "Bearing3_2": {"failure": "inner_race+outer_race+cage", "life_min": 2496},
        "Bearing3_3": {"failure": "inner_race", "life_min": 371},
        "Bearing3_4": {"failure": "inner_race", "life_min": 1515},
        "Bearing3_5": {"failure": "outer_race", "life_min": 114},
    },

    "feature_settings": {
        "primary_feature": "kurtosis",
        "features": ["rms", "kurtosis", "crest_factor", "peak_to_peak", "skewness",
                     "bpfi_energy", "bpfo_energy", "bsf_energy", "ftf_energy"],
    },
}
