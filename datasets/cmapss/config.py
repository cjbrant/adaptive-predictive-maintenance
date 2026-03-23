CMAPSS_CONFIG = {
    "name": "cmapss",
    "equipment": "Simulated commercial turbofan engine (C-MAPSS)",
    "equipment_type": "turbofan_engine",
    "prior_quality": "fleet_derived",
    "is_run_to_failure": True,

    "sub_datasets": {
        "FD001": {"n_train": 100, "n_test": 100, "fault_modes": 1, "op_conditions": 1},
        "FD002": {"n_train": 260, "n_test": 259, "fault_modes": 1, "op_conditions": 6},
        "FD003": {"n_train": 100, "n_test": 100, "fault_modes": 2, "op_conditions": 1},
        "FD004": {"n_train": 249, "n_test": 248, "fault_modes": 2, "op_conditions": 6},
    },

    "columns": {
        "index": ["unit_nr", "time_cycles"],
        "settings": ["setting_1", "setting_2", "setting_3"],
        "sensors": [f"s_{i}" for i in range(1, 22)],
    },

    "informative_sensors": [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],

    "rul_cap": 125,

    "feature_settings": {
        "primary_feature": "health_index",
    }
}
