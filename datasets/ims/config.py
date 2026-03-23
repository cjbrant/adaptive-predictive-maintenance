import numpy as np

IMS_CONFIG = {
    "name": "ims",
    "equipment": "Rexnord ZA-2115 double-row spherical roller bearing",
    "equipment_type": "roller_bearing",
    "prior_quality": "exact_oem",
    "is_run_to_failure": True,

    "oem_specs": {
        "designation": "Rexnord ZA-2115",
        "bore_in": 1.9375,
        "bore_mm": 49.2125,
        "C_lbf": 20300,
        "C_kn": 90.3,
        "C0_lbf": 26200,
        "C0_kn": 116.54,
        "life_exponent": 10/3,
        "n_rollers": 16,
        "pitch_diameter_in": 2.815,
        "roller_diameter_in": 0.331,
        "contact_angle_deg": 15.17,
    },

    "operating_conditions": {
        "rpm": 2000,
        "radial_load_lbf": 6000,
        "radial_load_kn": 26.69,
        "sampling_rate_hz": 20000,
        "snapshot_duration_sec": 1.0,
        "snapshot_interval_min": 10,
        "lubrication": "forced_circulation",
    },

    "experiments": {
        "set1": {
            "dir_name": "1st_test",
            "description": "Inner race defect (bearing 3), roller element defect (bearing 4)",
            "duration": "Feb 12 - Feb 19, 2004 (~7 days)",
            "n_channels": 8,
            "failures": {"bearing3": "inner_race", "bearing4": "roller_element"},
        },
        "set2": {
            "dir_name": "2nd_test",
            "description": "Outer race failure (bearing 1)",
            "duration": "Feb 12 - Feb 19, 2004 (~7 days)",
            "n_channels": 4,
            "failures": {"bearing1": "outer_race"},
        },
        "set3": {
            "dir_name": "4th_test/txt",
            "description": "Outer race failure (bearing 3)",
            "duration": "Mar 4 - Apr 4, 2004 (~31 days)",
            "n_channels": 4,
            "failures": {"bearing3": "outer_race"},
        },
    },

    "feature_settings": {
        "primary_feature": "kurtosis",
        "window_samples": 20480,
        "features": ["rms", "kurtosis", "crest_factor", "peak_to_peak", "skewness",
                     "bpfo_energy", "bpfi_energy", "bsf_energy", "ftf_energy"],
    }
}
