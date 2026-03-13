from pathlib import Path
import argparse as arg
import os

import lightning as L
import torch

from sot_mol.comparm import GP, Update_PARAMS
from sot_mol.models.online_rl_interface import MolGen_OnlineRLModel


parser = arg.ArgumentParser(description="Online Finetuning Trainer (separate entry)")
parser.add_argument("--config", type=str, default="rl.json")
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--group_size", type=int, default=4)
parser.add_argument("--replay_batch_size", type=int, default=64)
parser.add_argument("--rollout_buffer_size", type=int, default=4096)
parser.add_argument("--eta_max", type=float, default=0.5)
parser.add_argument("--eta_scale", type=float, default=1e-3)
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

model = MolGen_OnlineRLModel(
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
    group_size=args.group_size,
    replay_batch_size=args.replay_batch_size,
    rollout_buffer_size=args.rollout_buffer_size,
    beta=args.beta,
    eta_max=args.eta_max,
    eta_scale=args.eta_scale,
)

prior_ckpt = script_dir / "prior.ckpt"
datasets_dir = script_dir.parent / "datasets"

model.Train(
    train_datafile=datasets_dir / "train.smol",
    val_datafile=datasets_dir / "val.smol",
    test_datafile=datasets_dir / "test.smol",
    epochs=1,
    save_path=str(script_dir / "models"),
    project_name="SOTMOL_ONLINE_FINETUNE_SEPARATE",
    load_ckpt=str(prior_ckpt),
    lr=GP.LR,
    debug=False,
    ngpus=1,
    batchsize=4,
    log_steps=1,
)
