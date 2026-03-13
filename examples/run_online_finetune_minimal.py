"""最简 Online Finetuning Trainer 运行示例。

运行方式（在仓库根目录）:
python examples/run_online_finetune_minimal.py --config rl.json --load_ckpt prior.ckpt
"""

from pathlib import Path
import argparse

from sot_mol.comparm import GP, Update_PARAMS
from sot_mol.models.rl_interface import MolGen_RLModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="rl.json")
    parser.add_argument("--load_ckpt", type=str, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    gp = Update_PARAMS(GP, str(config_path))

    model = MolGen_RLModel(
        d_model=gp.D_MODEL,
        atom_tokens=gp.TOKENS,
        n_bond_types=gp.N_BOND_TYPES,
        coord_std=gp.COORDS_STD_DEV,
        scale_ot=gp.SCALE_OT,
        self_cond=True,
        formulation="endpoint",
        eval_3D_props=False,
        reward_name="qed",
        group_size=4,
        rollout_batch_size=32,
        rollout_buffer_size=1024,
        beta=1.0,
        eta_max=0.5,
    )

    data_dir = repo_root.parent / "datasets"
    model.Train(
        train_datafile=data_dir / "train.smol",
        val_datafile=data_dir / "val.smol",
        test_datafile=data_dir / "test.smol",
        epochs=1,
        save_path=str(repo_root / "models"),
        project_name="SOTMOL_ONLINE_FINETUNE_MINIMAL",
        load_ckpt=args.load_ckpt,
        lr=gp.LR,
        ngpus=1,
        batchsize=4,
        debug=True,
    )


if __name__ == "__main__":
    main()
