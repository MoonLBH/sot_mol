import copy


def _default_constraints():
    return {
        "validity": {
            "type": "min",
            "source": "validity_indicator",
            "threshold": 0.95,
            "weight_init": 1.0,
        },
        "connected_validity": {
            "type": "min",
            "source": "connected_validity_indicator",
            "threshold": 0.95,
            "weight_init": 1.0,
        },
        "strain": {
            "type": "max",
            "source": "strain_energy_per_atom",
            "threshold": 0.15,
            "weight_init": 0.5,
        },
    }


TASK_PRESETS = {
    "mpo_task_1_tanimoto_qed_tpsa": {
        "reward_groups": {
            "2d_soft": {
                "terms": {
                    "tanimoto_reward": 1.0,
                    "qed_reward": 1.0,
                    "tpsa_window_reward": 1.0,
                },
                "coefficient": 1.0,
            }
        },
        "constraint_specs": _default_constraints(),
        "transforms": {
            "tpsa_window_reward": {"source": "tpsa", "lower": 20.0, "upper": 120.0, "mode": "triangle"},
        },
        "routed_loss_weights": {"2d_soft": {"coord-loss": 0.0, "type-loss": 1.0, "bond-loss": 1.0, "charge-loss": 0.5}},
    },
    "mpo_task_2_tanimoto_qed_logp": {
        "reward_groups": {
            "2d_soft": {
                "terms": {
                    "tanimoto_reward": 1.0,
                    "qed_reward": 1.0,
                    "logp_window_reward": 1.0,
                },
                "coefficient": 1.0,
            }
        },
        "constraint_specs": _default_constraints(),
        "transforms": {
            "logp_window_reward": {"source": "logp", "lower": 1.0, "upper": 4.0, "mode": "triangle"},
        },
        "routed_loss_weights": {"2d_soft": {"coord-loss": 0.0, "type-loss": 1.0, "bond-loss": 1.0, "charge-loss": 0.5}},
    },
    "mpo_task_3_tanimoto_qed_sa": {
        "reward_groups": {
            "2d_soft": {
                "terms": {
                    "tanimoto_reward": 1.0,
                    "qed_reward": 1.0,
                    "sa_reward": 1.0,
                },
                "coefficient": 1.0,
            }
        },
        "constraint_specs": _default_constraints(),
        "transforms": {},
        "routed_loss_weights": {"2d_soft": {"coord-loss": 0.0, "type-loss": 1.0, "bond-loss": 1.0, "charge-loss": 0.5}},
    },
    "mpo_task_4_tanimoto_qed_mw": {
        "reward_groups": {
            "2d_soft": {
                "terms": {
                    "tanimoto_reward": 1.0,
                    "qed_reward": 1.0,
                    "mw_window_reward": 1.0,
                },
                "coefficient": 1.0,
            }
        },
        "constraint_specs": _default_constraints(),
        "transforms": {
            "mw_window_reward": {"source": "molecular_weight", "lower": 250.0, "upper": 550.0, "mode": "triangle"},
        },
        "routed_loss_weights": {"2d_soft": {"coord-loss": 0.0, "type-loss": 1.0, "bond-loss": 1.0, "charge-loss": 0.5}},
    },
    "mpo_task_5_tanimoto_qed_sa": {
        "reward_groups": {
            "2d_soft": {
                "terms": {
                    "tanimoto_reward": 1.0,
                    "qed_reward": 1.0,
                    "sa_reward": 1.0,
                },
                "coefficient": 1.0,
            }
        },
        "constraint_specs": _default_constraints(),
        "transforms": {},
        "routed_loss_weights": {"2d_soft": {"coord-loss": 0.0, "type-loss": 1.0, "bond-loss": 1.0, "charge-loss": 0.5}},
    },
    "mpo_task_6_tanimoto_qed_logp": {
        "reward_groups": {
            "2d_soft": {
                "terms": {
                    "tanimoto_reward": 1.0,
                    "qed_reward": 1.0,
                    "logp_window_reward": 1.0,
                },
                "coefficient": 1.0,
            }
        },
        "constraint_specs": _default_constraints(),
        "transforms": {
            "logp_window_reward": {"source": "logp", "lower": 1.0, "upper": 4.0, "mode": "triangle"},
        },
        "routed_loss_weights": {"2d_soft": {"coord-loss": 0.0, "type-loss": 1.0, "bond-loss": 1.0, "charge-loss": 0.5}},
    },
    "mpo_task_7_tanimoto_qed_tpsa": {
        "reward_groups": {
            "2d_soft": {
                "terms": {
                    "tanimoto_reward": 1.0,
                    "qed_reward": 1.0,
                    "tpsa_window_reward": 1.0,
                },
                "coefficient": 1.0,
            }
        },
        "constraint_specs": _default_constraints(),
        "transforms": {
            "tpsa_window_reward": {"source": "tpsa", "lower": 20.0, "upper": 120.0, "mode": "triangle"},
        },
        "routed_loss_weights": {"2d_soft": {"coord-loss": 0.0, "type-loss": 1.0, "bond-loss": 1.0, "charge-loss": 0.5}},
    },
    "scaffold_hopping_template": {
        "reward_groups": {
            "2d_soft": {
                "terms": {"qed_reward": 1.0, "sa_reward": 1.0, "scaffold_change_reward": 1.0},
                "coefficient": 1.0,
            },
            "3d_soft": {
                "terms": {"vina_reward": 1.0, "shape_similarity_reward": 1.0},
                "coefficient": 0.5,
            },
        },
        "constraint_specs": {
            **_default_constraints(),
            "required_smarts": {"type": "min", "source": "required_smarts_satisfied_indicator", "threshold": 1.0, "weight_init": 1.0},
            "motif_retention": {"type": "min", "source": "motif_retention_fraction", "threshold": 0.8, "weight_init": 1.0},
        },
        "transforms": {},
        "routed_loss_weights": {
            "2d_soft": {"coord-loss": 0.0, "type-loss": 1.0, "bond-loss": 1.0, "charge-loss": 0.5},
            "3d_soft": {"coord-loss": 1.0, "type-loss": 0.0, "bond-loss": 0.0, "charge-loss": 0.0},
        },
    },
    "admet_mpo_2d3d_template": {
        "reward_groups": {
            "2d_soft": {
                "terms": {"qed_reward": 1.0, "sa_reward": 1.0, "clearance_reward": 1.0},
                "coefficient": 1.0,
            },
            "3d_soft": {
                "terms": {"vina_reward": 1.0},
                "coefficient": 1.0,
            },
        },
        "constraint_specs": {
            **_default_constraints(),
            "posebusters": {"type": "min", "source": "posebusters_validity_indicator", "threshold": 1.0, "weight_init": 0.5},
        },
        "transforms": {
            "clearance_reward": {"source": "clearance_pred_raw", "lower": 0.0, "upper": 0.5, "mode": "double_sigmoid"},
        },
        "routed_loss_weights": {
            "2d_soft": {"coord-loss": 0.0, "type-loss": 1.0, "bond-loss": 1.0, "charge-loss": 0.5},
            "3d_soft": {"coord-loss": 1.0, "type-loss": 0.0, "bond-loss": 0.0, "charge-loss": 0.0},
        },
    },
    "generic_similarity_mpo_template": {
        "reward_groups": {
            "2d_soft": {
                "terms": {"tanimoto_reward": 1.0, "qed_reward": 1.0, "logp_window_reward": 1.0},
                "coefficient": 1.0,
            }
        },
        "constraint_specs": _default_constraints(),
        "transforms": {
            "logp_window_reward": {"source": "logp", "lower": 1.0, "upper": 4.0, "mode": "triangle"},
        },
        "routed_loss_weights": {"2d_soft": {"coord-loss": 0.0, "type-loss": 1.0, "bond-loss": 1.0, "charge-loss": 0.5}},
    },
}


def get_task_preset(name: str):
    if name not in TASK_PRESETS:
        available = ", ".join(sorted(TASK_PRESETS))
        raise KeyError(f"Unknown task preset '{name}'. Available presets: {available}")
    return copy.deepcopy(TASK_PRESETS[name])
