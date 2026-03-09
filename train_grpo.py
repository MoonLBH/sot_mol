from pathlib import Path
import argparse as arg
import os

import lightning as L
import torch
from datetime import datetime

from sot_mol.comparm import GP, Update_PARAMS
from sot_mol.models.grpo_interface import MolGen_GRPOModel


parser = arg.ArgumentParser(description="Flow-GRPO fine-tuning quick test")
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

model = MolGen_GRPOModel(
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
    group_size=8,
    clip_eps=0.2,
    kl_beta=1e-3,
    sde_noise_scale=0.7,
    ratio_max=20.0,
    use_reference_policy=True,
)

prior_ckpt = script_dir / "prior.ckpt"
datasets_dir = script_dir.parent / "datasets"

model.Train(
    train_datafile=datasets_dir / "train.smol",
    val_datafile=datasets_dir / "val.smol",
    test_datafile=datasets_dir / "test.smol",
    epochs=100,
    save_path=str(script_dir / "models"),
    project_name="SOTMOL_GRPO_QED_TEST",
    load_ckpt=str(prior_ckpt),
    lr=GP.LR,
    debug=False,
    ngpus=4,
    batchsize=64,
    mini_batchsize=1,
    max_steps=32,
    cache_on_cpu=True,
    log_steps=1,
    exp_tag='version_4',
)
