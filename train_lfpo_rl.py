from pathlib import Path
import argparse as arg
import os

import lightning as L
import torch

from sot_mol.comparm import GP, Update_PARAMS
from sot_mol.models.rl_lfpo_interface import MolGen_LFPOModel


parser = arg.ArgumentParser(description="LFPO-F FM RL quick test")
parser.add_argument("--config", type=str, default="rl.json")
parser.add_argument("--objective_name", type=str, default="qed")
parser.add_argument("--partition_mode", type=str, default="scalar_top_bottom")
parser.add_argument("--oracle_log_path", type=str, default="")
parser.add_argument("--run_block1", action="store_true")
args = parser.parse_args()

script_dir = Path(__file__).resolve().parent
config_path = Path(args.config)
if not config_path.is_absolute():
    config_path = script_dir / config_path

GP = Update_PARAMS(GP, str(config_path))

os.environ["CUDA_VISIBLE_DEVICES"] = GP.CUDA_VISIBLE_DEVICES
torch.set_float32_matmul_precision("high")
L.seed_everything(12345)

import torch._dynamo

torch._dynamo.config.suppress_errors = True

BLOCK1_TASKS = ["Ranolazine_MPO", "Osimertinib_MPO", "Fexofenadine_MPO", "Sitagliptin_MPO"]
objective_name = args.objective_name
objective_config = {"aggregate": "official" if objective_name != "qed" else "geometric", "use_official_guacamol": True, "fallback_aggregate": "geometric", "pareto_component_names": ["sim_ranolazine_AP", "logP", "TPSA", "num_F"], "feasibility": {"valid": True, "connected": True}}
partition_config = {"mode": args.partition_mode, "top_ratio": 0.25, "bottom_ratio": 0.25, "pareto_rank_max": 1, "top_candidate_quantile": 0.7, "diversity_mode": "scaffold", "bottom_mode": "unusable_region", "bottom_priority": ["invalid", "severe", "dominated", "low_score"], "bad_rank_quantile": 0.75, "low_score_quantile": 0.25, "top_selection_score_mode": "score", "top_selection_component_weights": {}, "use_min_component_bonus": False, "min_component_weight": 0.0, "bottom_component_floor": {}, "bottom_floor_type": "component", "log_component_floor": {"num_F": 0.60, "sim_ranolazine_AP": 0.05}, "enable_component_floor_bottom": False, "enable_component_balanced_top": False, "enable_min_component_bonus": False}
metric_config = {"enabled": bool(args.oracle_log_path), "oracle_log_path": args.oracle_log_path if args.oracle_log_path else str(script_dir / "oracle_logs" / f"{objective_name}.csv"), "novelty_reference_path": str(script_dir / "train_smiles.txt"), "log_ref_train": True, "log_current_eval": True}

model = MolGen_LFPOModel(
    d_model=GP.D_MODEL,
    atom_tokens=GP.TOKENS,
    n_bond_types=GP.N_BOND_TYPES,
    coord_std=GP.COORDS_STD_DEV,
    scale_ot=GP.SCALE_OT,
    self_cond=True,
    coord_noise_std=0.2,
    formulation="endpoint",
    eval_3D_props=False,
    ot_bond_weight=1,
    reward_name="qed",
    objective_name=objective_name,
    objective_config=objective_config,
    partition_config=partition_config,
    metric_config=metric_config,
    anchor_weight=0.1,
    anchor_loss_weight=1.0,
    use_reference_anchor=True,
    # lfpo_hparams={
    #     "lfpo_num_time_samples": 2,
    #     "lfpo_reward_temperature": 0.3,
    #     "lfpo_beta_types": 2.0,
    #     "lfpo_beta_bonds": 2.0,
    #     "lfpo_beta_charges": 2.0,
    #     "lfpo_beta_coord": 1.5,
    #     "lfpo_gamma_coord": 1.0,
    #     "lfpo_lambda_coord_rect": 0.0,
    #     "lfpo_lambda_types_rect": 1.0,
    #     "lfpo_lambda_bonds_rect": 1.0,
    #     "lfpo_lambda_charges_rect": 1.0,
    #     "lfpo_aux_fm_weight": 0.0,
    #     "ref_ema_decay": 0.9,
    #     "lfpo_use_charge_head": True,
    #     "lfpo_detach_targets": True,
    #     "lfpo_time_chunk_size": 64,
    #     "anchor_weight": 0.0,
    # },
    lfpo_hparams = {
        "lfpo_num_time_samples": 2,
        "lfpo_reward_temperature": 0.5,

        "lfpo_beta_types": 1.5,
        "lfpo_beta_bonds": 1.5,
        "lfpo_beta_charges": 1.5,
        "lfpo_beta_coord": 1.5,
        "lfpo_gamma_coord": 1.0,

        "lfpo_lambda_coord_rect": 0.0,
        "lfpo_lambda_types_rect": 1.0,
        "lfpo_lambda_bonds_rect": 1.0,
        "lfpo_lambda_charges_rect": 1.0,

        "lfpo_aux_fm_weight": 0.0,
        "anchor_weight": 0.0,

        "lfpo_top_weight_mode": "uniform",
        "lfpo_top_ratio": 0.25,
        "lfpo_bottom_ratio": 0.25,
        "lfpo_bottom_repulsion_weight": 0.5,
        "lfpo_middle_weight": 0.0,
        "lfpo_use_top_bottom": True,

        "ref_ema_decay": 0.9,
        "lfpo_use_charge_head": True,
        "lfpo_detach_targets": False,
        "lfpo_time_chunk_size": 64,

        "lfpo_eval_current_every": 100,
        "lfpo_log_current_reward": True,
        "lfpo_eval_current_samples": 1000,
        "lfpo_eval_current_batch_size": 256,
    }
)

prior_ckpt = script_dir / "prior.ckpt"
datasets_dir = script_dir.parent / "datasets"

model.Train(
    train_datafile=datasets_dir / "train.smol",
    val_datafile=datasets_dir / "val.smol",
    test_datafile=datasets_dir / "test.smol",
    epochs=10,
    save_path=str(script_dir / "models"),
    project_name="SOTMOL_LIFT_QED_TEST",
    load_ckpt=str(prior_ckpt),
    lr=GP.LR,
    debug=False,
    ngpus=1,
    batchsize=60,
    log_steps=1,
)
