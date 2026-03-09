import os
from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from ..data.datamodule import MGDataModule
from .interface import MolGen_Model
from .grpo_diff import GRPO_Lightning


class MolGen_GRPOModel(MolGen_Model):
    def __init__(
        self,
        atom_tokens,
        n_bond_types,
        coord_std,
        reward_name="qed",
        group_size=8,
        clip_eps=0.2,
        kl_beta=1e-3,
        sde_noise_scale=0.7,
        reward_norm_eps=1e-6,
        ratio_max=20.0,
        use_reference_policy=True,
        **kwargs,
    ):
        super().__init__(
            atom_tokens=atom_tokens,
            n_bond_types=n_bond_types,
            coord_std=coord_std,
            **kwargs,
        )

        self.reward_name = reward_name
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        self.sde_noise_scale = sde_noise_scale
        self.reward_norm_eps = reward_norm_eps
        self.ratio_max = ratio_max
        self.use_reference_policy = use_reference_policy

    def create_lightning_module(self, hparams=None, load_ckpt=None):
        default_hparams = {
            "use_ema": True,
            "coord_scale": self.coord_std,
            "lr": 1e-4,
            "self_cond": self.self_cond,
            "max_steps": self.max_steps,
            "default_coord_noise_std": self.coord_noise_std,
            "default_cat_noise_level": self.noise_level_for_types_bonds,
            "loss_weight": self.loss_weight,
            "formulation": self.formulation,
            "eval_3D_props": self.eval_3D_props,
            "reward_name": self.reward_name,
            "group_size": self.group_size,
            "clip_eps": self.clip_eps,
            "kl_beta": self.kl_beta,
            "sde_noise_scale": self.sde_noise_scale,
            "reward_norm_eps": self.reward_norm_eps,
            "ratio_max": self.ratio_max,
            "use_reference_policy": self.use_reference_policy,
            "cache_on_cpu": True,
            "val_save_path": None,
        }

        if hparams is not None:
            default_hparams.update(hparams)

        if load_ckpt is not None:
            lightning_module = GRPO_Lightning.load_from_checkpoint(
                load_ckpt,
                gen=self.network,
                vocab=self.vocab,
                map_location="cpu",
                strict=False,
                **default_hparams,
            )
        else:
            lightning_module = GRPO_Lightning(
                gen=self.network,
                vocab=self.vocab,
                **default_hparams,
            )

        return lightning_module

    def Train(
        self,
        train_datafile,
        val_datafile,
        test_datafile,
        epochs,
        save_path="./models",
        project_name="SOTMOL_GRPO",
        load_ckpt=None,
        lr=1e-4,
        warm_up_steps=10000,
        acc_batches=1,
        log_steps=1,
        val_check_epochs=1,
        debug=False,
        gradient_clip_val=1.0,
        ngpus=1,
        batchsize=16,
        mini_batchsize=4,
        max_steps=None,
        cache_on_cpu=None,
    ):
        if not debug:
            os.makedirs("./TensorBoard", exist_ok=True)
            logger = TensorBoardLogger("./TensorBoard", name=project_name, version=None)
            exp_dir = logger.log_dir
        else:
            logger = None
            exp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = os.path.join(save_path, f"{project_name}_debug_{exp_tag}")
            
        self.data_module = MGDataModule(
            self.vocab,
            self.n_bond_types,
            train_datafile=train_datafile,
            val_datafile=val_datafile,
            test_datafile=test_datafile,
            max_atoms=self.max_atoms,
            coord_std=self.coord_std,
            scale_ot=self.scale_ot,
            scale_ot_factor=0.2,
            batchsize=batchsize,
            mini_batchsize=mini_batchsize,
            with_Hs=self.with_Hs,
            ot_geo_weight=self.ot_geo_weight,
            ot_type_weight=self.ot_type_weight,
            ot_bond_weight=self.ot_bond_weight,
        )

        val_save_dir = os.path.join(exp_dir, "val_samples")
        os.makedirs(val_save_dir, exist_ok=True)
        
        training_hparams = {
            "lr": lr,
            "warm_up_steps": warm_up_steps,
            "val_save_path": val_save_dir,
        }
        if max_steps is not None:
            training_hparams["max_steps"] = max_steps
        if cache_on_cpu is not None:
            training_hparams["cache_on_cpu"] = cache_on_cpu
        self.lightning_module = self.create_lightning_module(
            hparams=training_hparams,
            load_ckpt=load_ckpt,
        )


        # Keep checkpoints under the same experiment directory as tensorboard logs
        # to avoid cross-run mixing/overwriting when multiple experiments share save_path.
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpointing = ModelCheckpoint(
            dirpath=ckpt_dir,
            save_top_k=5,
            every_n_epochs=1,
            monitor="train-grpo-reward-mean",
            mode="max",
            save_last=True,
        )

        trainer = L.Trainer(
            devices=ngpus,
            min_epochs=epochs,
            max_epochs=epochs,
            logger=logger,
            log_every_n_steps=log_steps,
            accumulate_grad_batches=acc_batches,
            # gradient_clip_val=gradient_clip_val,
            callbacks=[lr_monitor, checkpointing],
            precision="32",
            strategy="ddp_find_unused_parameters_true",
            limit_val_batches=0,
            num_sanity_val_steps=0,
        )

        trainer.fit(self.lightning_module, self.data_module)
        return
