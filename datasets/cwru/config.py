CWRU_CONFIG = {
    "name": "cwru",
    "equipment": "SKF 6205-2RS JEM deep groove ball bearing",
    "equipment_type": "ball_bearing",
    "prior_quality": "exact_oem",
    "is_run_to_failure": False,  # Synthetic trajectory

    "oem_specs": {
        "designation": "SKF 6205-2RS JEM",
        "bore_mm": 25.0,
        "C_kn": 14.8,
        "C0_kn": 7.8,
        "life_exponent": 3.0,
        "bpfi_mult": 5.4152,
        "bpfo_mult": 3.5848,
        "bsf_mult": 4.7135,
        "ftf_mult": 0.39828,
    },

    "operating_conditions": {
        "rpm": 1797,
        "load_hp": [0, 1, 2, 3],
        "sampling_rate_hz": 12000,
    },

    "trajectory_construction": {
        "fault_type": "inner_race",
        "severity_order": ["normal", "0.007", "0.014", "0.021"],
        "primary_feature": "kurtosis",
    }
}
