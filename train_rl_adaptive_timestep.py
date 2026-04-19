from pathlib import Path
import argparse as arg
import json
import os

import lightning as L
import torch

from sot_mol.comparm import GP, Update_PARAMS
from sot_mol.models.rl_interface import MolGen_RLModel


def parse_args():
    parser = arg.ArgumentParser(description="RL training with adaptive timestep sampler support")
    parser.add_argument("--config", type=str, default="rl.json")
    parser.add_argument("--adaptive-enabled", action="store_true", help="Enable adaptive timestep sampling")
    parser.add_argument(
        "--adaptive-mode",
        type=str,
        default="probe",
        choices=["probe", "one_step_value"],
        help="Adaptive timestep feedback mode",
    )
    parser.add_argument("--adaptive-update-freq", type=int, default=40)
    parser.add_argument("--adaptive-sampler-lr", type=float, default=1e-3)
    parser.add_argument("--adaptive-overrides", type=str, default=None, help="JSON file for RL_Lightning adaptive kwargs")
    parser.add_argument("--project-name", type=str, default="SOTMOL_RL_ADAPTIVE_TIMESTEP")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batchsize", type=int, default=48)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--reward-name", type=str, default="qed")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    gp = Update_PARAMS(GP, str(config_path))
    os.environ["CUDA_VISIBLE_DEVICES"] = gp.CUDA_VISIBLE_DEVICES
    torch.set_float32_matmul_precision("high")
    L.seed_everything(args.seed)

    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    adaptive_hparams = {
        "adaptive_timestep_enabled": args.adaptive_enabled,
        "adaptive_timestep_mode": args.adaptive_mode,
        "adaptive_timestep_update_freq": args.adaptive_update_freq,
        "timestep_sampler_lr": args.adaptive_sampler_lr,
    }
    if args.adaptive_overrides is not None:
        with open(args.adaptive_overrides, "r", encoding="utf-8") as f:
            adaptive_hparams.update(json.load(f))

    model = MolGen_RLModel(
        d_model=gp.D_MODEL,
        atom_tokens=gp.TOKENS,
        n_bond_types=gp.N_BOND_TYPES,
        coord_std=gp.COORDS_STD_DEV,
        scale_ot=gp.SCALE_OT,
        self_cond=True,
        coord_noise_std=0.2,
        formulation="endpoint",
        eval_3D_props=False,
        ot_bond_weight=1,
        reward_name=args.reward_name,
        reward_beta=2.0,
        reward_weight_min=0.1,
        reward_weight_max=10.0,
        anchor_weight=0.1,
        anchor_loss_weight=1.0,
        use_reference_anchor=True,
        adaptive_hparams=adaptive_hparams,
    )

    prior_ckpt = script_dir / "prior.ckpt"
    datasets_dir = script_dir.parent / "datasets"
    project_name = args.project_name
    if args.adaptive_enabled:
        project_name = f"{project_name}_{args.adaptive_mode}_fs{args.adaptive_update_freq}"
    else:
        project_name = f"{project_name}_baseline"

    print("=== RL adaptive timestep config ===")
    print(json.dumps(adaptive_hparams, indent=2, ensure_ascii=False))
    print(f"project_name: {project_name}")
    print("===================================")

    model.Train(
        train_datafile=datasets_dir / "train.smol",
        val_datafile=datasets_dir / "val.smol",
        test_datafile=datasets_dir / "test.smol",
        epochs=args.epochs,
        save_path=str(script_dir / "models"),
        project_name=project_name,
        load_ckpt=str(prior_ckpt),
        lr=gp.LR,
        debug=False,
        ngpus=args.ngpus,
        batchsize=args.batchsize,
        log_steps=1,
    )


if __name__ == "__main__":
    main()
