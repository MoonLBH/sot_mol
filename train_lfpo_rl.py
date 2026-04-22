from pathlib import Path
import argparse as arg
import os

import lightning as L
import torch

from sot_mol.comparm import GP, Update_PARAMS
from sot_mol.models.rl_lfpo_interface import MolGen_LFPOModel


parser = arg.ArgumentParser(description="LFPO-F FM RL quick test")
parser.add_argument("--config", type=str, default="rl.json")
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
    anchor_weight=0.1,
    anchor_loss_weight=1.0,
    use_reference_anchor=True,
    lfpo_hparams={
        "lfpo_num_time_samples": 4,
        "lfpo_reward_temperature": 1.0,
        "lfpo_beta_types": 1.0,
        "lfpo_beta_bonds": 1.0,
        "lfpo_beta_charges": 1.0,
        "lfpo_beta_coord": 1.5,
        "lfpo_gamma_coord": 1.0,
        "lfpo_lambda_coord_rect": 1.0,
        "lfpo_lambda_types_rect": 1.0,
        "lfpo_lambda_bonds_rect": 1.0,
        "lfpo_lambda_charges_rect": 1.0,
        "lfpo_aux_fm_weight": 0.05,
        "ref_ema_decay": 0.999,
        "lfpo_use_charge_head": True,
        "lfpo_detach_targets": True,
    },
)

prior_ckpt = script_dir / "prior.ckpt"
datasets_dir = script_dir.parent / "datasets"

model.Train(
    train_datafile=datasets_dir / "train.smol",
    val_datafile=datasets_dir / "val.smol",
    test_datafile=datasets_dir / "test.smol",
    epochs=5,
    save_path=str(script_dir / "models"),
    project_name="SOTMOL_LFPO_QED_TEST",
    load_ckpt=str(prior_ckpt),
    lr=GP.LR,
    debug=False,
    ngpus=1,
    batchsize=48,
    log_steps=1,
)
