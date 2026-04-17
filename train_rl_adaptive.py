from pathlib import Path
import argparse as arg
import json
import os

import lightning as L
import torch

from sot_mol.comparm import GP, Update_PARAMS
from sot_mol.models.rl_adaptive_interface import MolGen_AdaptiveRLModel


def parse_args():
    parser = arg.ArgumentParser(description="Adaptive reward-weighted FM RL training")
    parser.add_argument("--config", type=str, default="rl.json")
    parser.add_argument("--preset", type=str, default=None, help="task preset name from models/reward_presets.py")
    parser.add_argument("--reward-name", type=str, default="qed")
    parser.add_argument("--adaptive-overrides", type=str, default=None, help="JSON file for AdaptiveRL_Lightning kwargs")
    parser.add_argument("--mode", type=str, default="all", choices=["baseline", "module_a", "module_b", "module_c", "all"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batchsize", type=int, default=48)
    parser.add_argument("--project-name", type=str, default="SOTMOL_RL_ADAPTIVE")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--load-ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--ngpus", type=int, default=1)
    return parser.parse_args()


def build_mode_config(mode: str):
    mode_cfg = {
        "adaptive_time_sampling": False,
        "reward_routing_enabled": False,
        "constraints_enabled": False,
    }
    if mode == "module_a":
        mode_cfg["adaptive_time_sampling"] = True
    elif mode == "module_b":
        mode_cfg["reward_routing_enabled"] = True
    elif mode == "module_c":
        mode_cfg["constraints_enabled"] = True
    elif mode == "all":
        mode_cfg.update(
            {
                "adaptive_time_sampling": True,
                "reward_routing_enabled": True,
                "constraints_enabled": True,
            }
        )
    return mode_cfg


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

    adaptive_hparams = build_mode_config(args.mode)
    adaptive_hparams["reward_name"] = args.reward_name
    adaptive_hparams["task_preset_name"] = args.preset
    if args.adaptive_overrides is not None:
        with open(args.adaptive_overrides, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        adaptive_hparams.update(user_cfg)

    model = MolGen_AdaptiveRLModel(
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
    if args.load_ckpt is not None:
        prior_ckpt = Path(args.load_ckpt)
    datasets_dir = script_dir.parent / "datasets"
    save_path = args.save_path or str(script_dir / "models")

    project_name = args.project_name
    if args.preset is not None:
        project_name = f"{project_name}_{args.preset}_{args.mode}"
    else:
        project_name = f"{project_name}_{args.reward_name}_{args.mode}"

    print("=== Adaptive RL run config ===")
    print(json.dumps(adaptive_hparams, indent=2, ensure_ascii=False))
    print(f"project_name: {project_name}")
    print(f"load_ckpt: {prior_ckpt}")
    print("==============================")

    model.Train(
        train_datafile=datasets_dir / "train.smol",
        val_datafile=datasets_dir / "val.smol",
        test_datafile=datasets_dir / "test.smol",
        epochs=args.epochs,
        save_path=save_path,
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
