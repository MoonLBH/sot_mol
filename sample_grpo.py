from pathlib import Path
import argparse as arg
import os

import lightning as L
import torch

from sot_mol.comparm import GP, Update_PARAMS
from sot_mol.models.grpo_interface import MolGen_GRPOModel


parser = arg.ArgumentParser(description="Sample molecules from a GRPO checkpoint")
parser.add_argument("--config", type=str, default="rl.json")
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--test_datafile", type=str, default="")
parser.add_argument("--save_path", type=str, default="/data/bhli/Project/Mol-RL/test")
parser.add_argument("--max_steps", type=int, default=32)
parser.add_argument("--n_samples", type=int, default=500)
parser.add_argument("--n_replicates", type=int, default=1)
parser.add_argument("--batchsize", type=int, default=100)
args = parser.parse_args()

script_dir = Path(__file__).resolve().parent
config_path = Path(args.config)
if not config_path.is_absolute():
    config_path = script_dir / config_path

ckpt_path = Path(args.ckpt)
if not ckpt_path.is_absolute():
    ckpt_path = script_dir / ckpt_path

GP = Update_PARAMS(GP, str(config_path))
os.environ["CUDA_VISIBLE_DEVICES"] = GP.CUDA_VISIBLE_DEVICES

torch.set_float32_matmul_precision("high")
L.seed_everything(12345)

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

if args.test_datafile:
    test_datafile = Path(args.test_datafile)
    if not test_datafile.is_absolute():
        test_datafile = script_dir / test_datafile
else:
    test_datafile = script_dir.parent / "datasets" / "test.smol"

model.Sample(
    test_datafile=test_datafile,
    save_path=args.save_path,
    inference_steps=args.max_steps,
    n_samples=args.n_samples,
    n_replicates=args.n_replicates,
    load_ckpt=str(ckpt_path),
    batchsize=args.batchsize,
)
