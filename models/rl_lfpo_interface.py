import os

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from ..data.datamodule import MGDataModule
from .interface import MolGen_Model
# from .lfpo_f import LFPOF_Lightning
# from .lfpo_f_top_imitation import LFPOF_Lightning
# from .lfpo_f_TopBottom_Rectification import LFPOF_Lightning
from .lift import LIFT_Lightning


class MolGen_LFPOModel(MolGen_Model):
    def __init__(
        self,
        atom_tokens,
        n_bond_types,
        coord_std,
        reward_name="qed",
        reward_beta=2.0,
        reward_weight_min=0.1,
        reward_weight_max=10.0,
        reward_norm_eps=1e-6,
        anchor_weight=0.1,
        anchor_loss_weight=1.0,
        use_reference_anchor=True,
        lfpo_hparams=None,
        **kwargs,
    ):
        super().__init__(atom_tokens=atom_tokens, n_bond_types=n_bond_types, coord_std=coord_std, **kwargs)
        self.reward_name = reward_name
        self.reward_beta = reward_beta
        self.reward_weight_min = reward_weight_min
        self.reward_weight_max = reward_weight_max
        self.reward_norm_eps = reward_norm_eps
        self.anchor_weight = anchor_weight
        self.anchor_loss_weight = anchor_loss_weight
        self.use_reference_anchor = use_reference_anchor
        self.lfpo_hparams = lfpo_hparams or {}

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
            "reward_beta": self.reward_beta,
            "reward_weight_min": self.reward_weight_min,
            "reward_weight_max": self.reward_weight_max,
            "reward_norm_eps": self.reward_norm_eps,
            "anchor_weight": self.anchor_weight,
            "anchor_loss_weight": self.anchor_loss_weight,
            "use_reference_anchor": self.use_reference_anchor,
        }
        default_hparams.update(self.lfpo_hparams)
        if hparams is not None:
            default_hparams.update(hparams)

        if load_ckpt is not None:
            return LIFT_Lightning.load_from_checkpoint(
                load_ckpt,
                gen=self.network,
                vocab=self.vocab,
                map_location="cpu",
                strict=False,
                **default_hparams,
            )
        return LIFT_Lightning(gen=self.network, vocab=self.vocab, **default_hparams)

    def Train(
        self,
        train_datafile,
        val_datafile,
        test_datafile,
        epochs,
        save_path="./models",
        project_name="SOTMOL_LIFT",
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
    ):
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
            mini_batchsize=1,
            with_Hs=self.with_Hs,
            ot_geo_weight=self.ot_geo_weight,
            ot_type_weight=self.ot_type_weight,
            ot_bond_weight=self.ot_bond_weight,
        )

        self.lightning_module = self.create_lightning_module(
            hparams={"lr": lr, "warm_up_steps": warm_up_steps},
            load_ckpt=load_ckpt,
        )

        logger = None
        if not debug:
            os.makedirs("./TensorBoard", exist_ok=True)
            logger = TensorBoardLogger("./TensorBoard", name=project_name, version=None)

        lr_monitor = LearningRateMonitor(logging_interval="step")
        if logger is not None:
            ckpt_dir = os.path.join(logger.log_dir, "checkpoints")
        else:
            ckpt_dir = save_path

        os.makedirs(ckpt_dir, exist_ok=True)

        checkpointing = ModelCheckpoint(
            dirpath=ckpt_dir,
            save_top_k=3,
            every_n_epochs=1,
            monitor="train-lfpof-reward-current-mean_epoch",
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
