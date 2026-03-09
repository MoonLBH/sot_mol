from ..comparm import GP
from ..util.tokeniser import Vocabulary
from pathlib import Path
from functools import partial

from ..util.initlib import mol_transform,disable_lib_stdout,configure_fs,calc_train_steps
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from lightning.pytorch.loggers import TensorBoardLogger
from ..data.datamodule import MGDataModule


from .diff import SC_Lightning  
from ..util.metrics import *

import os
from torchmetrics import MetricCollection
import numpy as np
from tqdm import tqdm
from rdkit import Chem

class MolGen_Model:
    def __init__(
        self,
        atom_tokens,
        n_bond_types,
        coord_std,
        d_model=512,
        d_message=128,
        d_edge=128,
        d_message_hidden=128,
        n_layers=12,
        n_coord_sets=64,
        n_attn_heads=32,
        size_emb=64,
        max_atoms=256,
        scale_ot=True,
        ot_geo_weight=1.0,
        ot_type_weight=1.0,
        ot_bond_weight=0.5,
        noise_level_for_types_bonds=1.0,
        coord_noise_std=0.2,
        loss_weight={"types":0.2,"bonds":1.0,"charges":1.0},
        max_steps=128,
        with_Hs=True,
        self_cond=False, # Add self_cond param
        formulation="endpoint",
        eval_3D_props=True,
        **kwargs,
    ):

        self.d_model=d_model
        self.d_message=d_message
        self.d_edge=d_edge
        self.d_message_hidden=d_message_hidden
        self.n_layers=n_layers
        self.n_coord_sets=n_coord_sets
        self.n_attn_heads=n_attn_heads
        self.size_emb=size_emb
        self.max_atoms=max_atoms

        self.atom_tokens=atom_tokens
        self.vocab=Vocabulary(atom_tokens)
        self.n_atom_feats=self.vocab.size
        self.n_bond_types=n_bond_types
        self.coord_std=coord_std

        self.scale_ot=scale_ot

        self.ot_geo_weight=ot_geo_weight
        self.ot_type_weight=ot_type_weight
        self.ot_bond_weight=ot_bond_weight

        self.noise_level_for_types_bonds=noise_level_for_types_bonds
        self.coord_noise_std=coord_noise_std
        self.eval_3D_props=eval_3D_props

        self.loss_weight=loss_weight
        self.max_steps=max_steps
        self.with_Hs=with_Hs
        self.self_cond=self_cond
        self.formulation=formulation

        disable_lib_stdout()
        configure_fs()
        self.__build_network_arch()
        return

    def __build_network_arch(self):
        from .mixnet import EquiInvDynamics, DenoisingNet
        backbone = EquiInvDynamics(
            d_model=self.d_model,
            d_message=self.d_message,
            d_message_hidden=self.d_message_hidden,
            d_edge=self.d_edge,
            n_coord_sets=self.n_coord_sets,
            n_layers=self.n_layers,
            n_attn_heads=self.n_attn_heads,
            bond_refine=True,
            self_cond=self.self_cond,
            coord_norm=GP.COORD_NORM,
        )

        self.network = DenoisingNet(
            d_model=self.d_model,
            dynamics=backbone,
            n_atom_feats=self.n_atom_feats,
            d_edge=self.d_edge,
            n_edge_types=self.n_bond_types,
            self_cond=self.self_cond,
            size_emb=self.size_emb,
            max_atoms=self.max_atoms,
        )
            
        return

    def create_lightning_module(self,hparams=None,load_ckpt=None):
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
            "eval_3D_props":self.eval_3D_props, 
        }

        if hparams is not None:
            default_hparams.update(hparams)

        if load_ckpt is not None:
            lightning_module = SC_Lightning.load_from_checkpoint(load_ckpt,
                                                    gen=self.network,
                                                    vocab=self.vocab,
                                                    map_location="cpu",
                                                    **default_hparams)
        else:
            lightning_module = SC_Lightning(gen=self.network,
                                                vocab=self.vocab,
                                                **default_hparams)

        return lightning_module

    def Train(self,
                train_datafile,val_datafile,test_datafile,epochs,save_path='./models',project_name="SOTMOL",
                load_ckpt=None, lr=1e-4, warm_up_steps=10000,
                acc_batches=1,
                log_steps=50,
                val_check_epochs=1,
                debug=False,
                gradient_clip_val=1.0,
                ngpus=1,
                batchsize=16,
            ):

        self.data_module = MGDataModule(
                 self.vocab,self.n_bond_types,
                 train_datafile=train_datafile,
                 val_datafile=val_datafile,
                 test_datafile=test_datafile,
                 max_atoms=self.max_atoms,coord_std=self.coord_std,
                 scale_ot=self.scale_ot,
                 scale_ot_factor=0.2,
                 batchsize=batchsize,mini_batchsize=4,with_Hs=self.with_Hs,
                 ot_geo_weight=self.ot_geo_weight, ot_type_weight=self.ot_type_weight,
                 ot_bond_weight=self.ot_bond_weight, 
        )

        training_hparams = {
            "lr": lr,
            "warm_up_steps": warm_up_steps,
        }

        self.lightning_module=self.create_lightning_module(hparams=training_hparams,load_ckpt=load_ckpt)

        if not debug:
            os.makedirs(f"./TensorBoard", exist_ok=True)
            logger = TensorBoardLogger(f"./TensorBoard", name=project_name, version=None)
        else:
            logger = None

        lr_monitor = LearningRateMonitor(logging_interval="step")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        checkpointing = ModelCheckpoint(dirpath=save_path,save_top_k=3,every_n_epochs=1, monitor="val-validity", mode="max", save_last=True)

        trainer = L.Trainer(
            devices=ngpus,
            min_epochs=epochs,
            max_epochs=epochs,
            logger=logger,
            log_every_n_steps=log_steps,
            accumulate_grad_batches=acc_batches,
            gradient_clip_val=gradient_clip_val,
            check_val_every_n_epoch=val_check_epochs,
            callbacks=[lr_monitor, checkpointing],
            precision="32",
            strategy="ddp_find_unused_parameters_true",
        )

        trainer.fit(self.lightning_module, self.data_module)
        return

    def record_mols(self,mols,save_path):
        writer = Chem.SDWriter(save_path)
        for m in mols:
            if m is not None:
                writer.write(m)
        writer.close()
        return

    def Sample(self,test_datafile=None,
                save_path='./samples',
                inference_steps=128,
                n_samples=1000,
                n_replicates=1,
                load_ckpt=None,
                batchsize=4,
                ):

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        self.data_module = MGDataModule(
            self.vocab, self.n_bond_types,
            train_datafile=test_datafile,
            val_datafile=test_datafile,
            test_datafile=test_datafile,
            max_atoms=self.max_atoms, coord_std=self.coord_std,
            scale_ot=self.scale_ot, 
            scale_ot_factor=0.2, 
            batchsize=batchsize, mini_batchsize=4,with_Hs=self.with_Hs,
            ot_geo_weight=self.ot_geo_weight, ot_type_weight=self.ot_type_weight,
            ot_bond_weight=self.ot_bond_weight,
        )

        self.data_module.setup(stage="test")
        self.data_module.testset.sample(n_samples)

        self.lightning_module=self.create_lightning_module(
            load_ckpt=load_ckpt,
        )

        metrics, stability_metrics = self.init_metrics()

        results_list = []
        for replicate_index in range(n_replicates):
            print(f"Running replicate {replicate_index + 1} out of {n_replicates}")
            molecules, _, stabilities = self.generate_molecules(
                self.lightning_module, self.data_module, inference_steps, stabilities=True
            )
            print("Calculating metrics...")
            if self.eval_3D_props:
                self.record_mols(molecules,save_path=f"{save_path}/mols_{replicate_index}.sdf")
            smiles=[Chem.MolToSmiles(m) for m in molecules if m is not None]
            molecules_2D=[Chem.MolFromSmiles(s) for s in smiles]
            self.record_mols(molecules_2D,save_path=f"{save_path}/mols_2D_{replicate_index}.sdf")

            results = self.calc_metrics(molecules, metrics, stab_metrics=stability_metrics, mol_stabs=stabilities)

            results_list.append(results)

        results_dict = {key: [] for key in results_list[0].keys()}
        for results in results_list:
            for metric, value in results.items():
                results_dict[metric].append(value.item())

        mean_results = {metric: np.mean(values) for metric, values in results_dict.items()}
        std_results = {metric: np.std(values) for metric, values in results_dict.items()}
        with open(f"{save_path}/metrics.txt", "w") as f:
            f.write("Mean Results:\n")
            for key in mean_results.keys():
                f.write(f"{key}: {mean_results[key]} ( {std_results[key]} )\n")

        return mean_results, std_results, results_dict

    def predict(self,):
        return

    def generate_molecules(self, lightning_module, data_module, steps,  stabilities=False):
        test_dl = data_module.test_dataloader()
        lightning_module.eval()
        cuda_model = lightning_module.to("cuda")

        outputs = []
        for batch in tqdm(test_dl):
            batch = {k: v.cuda() for k, v in batch.items()}
            batch = cuda_model.flatten_batch(batch)

            noise={"coords":batch["noise_coords"],
                    "bonds":batch["noise_bonds"],
                    "atomics":batch["noise_atomics"],
                    "masks":batch["masks"],
                    "flag_3Ds":batch["flag_3Ds"],
                    }
            
            coms = batch.get("coms", None)
            # Sampling dataloaders from different pipelines may carry CoMs as [B, 3]
            # or expanded/padded tensors like [B, N, 3]. Normalise to [B, 3]
            # so `_generate` can broadcast-add safely to coords [B, n_atoms, 3].
            if coms is not None and coms.dim() == 3:
                coms = coms.mean(dim=1)

            output = cuda_model._generate(noise, steps, coms=coms)
            outputs.append(output)

        molecules = [cuda_model._generate_mols(output) for output in outputs]
        molecules = [mol for mol_list in molecules for mol in mol_list]

        if not stabilities:
            return molecules, outputs

        stabilities = [cuda_model._generate_stabilities(output) for output in outputs]
        stabilities = [mol_stab for mol_stabs in stabilities for mol_stab in mol_stabs]
        return molecules, outputs, stabilities

    def save_rdkit_sdf(self, mols, save_path,):
        from rdkit import Chem
        writer = Chem.SDWriter(save_path)
        for m in mols:
            if m is not None:
                writer.write(m)
        writer.close()

    def init_metrics(self,):
        metrics_2D={
            "validity": Validity(),
            "connected-validity": Validity(connected=True),
            "uniqueness": Uniqueness(),
        }

        metrics_3D = {
            "energy-validity": EnergyValidity(),
            "opt-energy-validity": EnergyValidity(optimise=True),
            "energy": AverageEnergy(),
            "energy-per-atom": AverageEnergy(per_atom=True),
            "strain": AverageStrainEnergy(),
            "strain-per-atom": AverageStrainEnergy(per_atom=True),
            "opt-rmsd": AverageOptRmsd(),
        }
        metrics = {**metrics_2D}
        if self.eval_3D_props:
            metrics = {**metrics, **metrics_3D}

        stability_metrics = {"atom-stability": AtomStability(), "molecule-stability": MoleculeStability()}

        metrics = MetricCollection(metrics, compute_groups=False)
        stability_metrics = MetricCollection(stability_metrics, compute_groups=False)
        return metrics, stability_metrics

    def calc_metrics(self,rdkit_mols, metrics, stab_metrics=None, mol_stabs=None):
        metrics.reset()
        metrics.update(rdkit_mols)
        results = metrics.compute()

        if stab_metrics is None:
            return results

        stab_metrics.reset()
        stab_metrics.update(mol_stabs)
        stab_results = stab_metrics.compute()

        results = {**results, **stab_results}
        return results
