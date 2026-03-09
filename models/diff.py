from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

import lightning as L
import inspect
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, OneCycleLR
from torchmetrics import MetricCollection
from ..comparm import GP

from ..util.functional import bonds_from_adj, adj_from_edges, one_hot_encode_tensor,adj_from_node_mask
from ..util.metrics import * 
from ..util.rdkit import mol_from_smiles, mol_from_atoms

from ..util.tokeniser import Vocabulary
from .molbuilder import MolBuilder


_T = torch.Tensor
_BatchT = dict[str, _T]

class SC_Lightning(L.LightningModule):
    def __init__(
        self,
        gen: torch.nn.Module,
        vocab: Vocabulary,
        lr: float,
        coord_scale: float = 1.0,
        use_ema: bool = True,
        compile_model: bool = True,
        warm_up_steps: Optional[int] = None,
        max_steps: int = 128,
        default_coord_noise_std: float = 0.2,
        default_cat_noise_level: float = 1.0,
        self_cond: bool = False,
        loss_weight: dict = {"types":0.2,"bonds":1.0,"charges":1.0},
        formulation="endpoint",
        eval_3D_props: bool = True,
        #**kwargs
    ):
        super().__init__()

        self.gen = gen
        self.vocab = vocab
        self.lr = lr
        self.coord_scale = coord_scale
        self.compile_model = compile_model
        self.warm_up_steps = warm_up_steps

        self.max_steps = max_steps
        self.default_coord_noise_std= default_coord_noise_std
        self.default_cat_noise_level = default_cat_noise_level
        self.eps=1e-5
        self.time_dist=torch.distributions.Beta(torch.tensor(2.0),torch.tensor(1.0))
        # Self-consistency training parameters
        self.weight_schedule_epochs = 100
        self.use_ema_for_consistency = True
        self.self_cond = self_cond
        self.loss_weight = loss_weight
        self.formulation=formulation
        self.eval_3D_props=eval_3D_props
        builder = MolBuilder(vocab)

        if use_ema:
            avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
            ema_gen = torch.optim.swa_utils.AveragedModel(gen, multi_avg_fn=avg_fn)

        self.builder = builder
        self.ema_gen = ema_gen if use_ema else None

        stability_metrics = {
            "atom-stability": AtomStability(),
            "molecule-stability": MoleculeStability()
        }
        gen_metrics_2D={"validity": Validity(),
                        "uniqueness": Uniqueness(),
                        "fc-validity": Validity(connected=True),
                        "MaxRingSize": MaxRingSize(),
                        "LargeRingRatio": LargeRingRatio(),
                        "ConnectedRatio": ConnectedRatio(),
                        "AverageFragments": AverageFragments()
                        }

        gen_metrics_3D = {
            "energy-validity": EnergyValidity(),
            "opt-energy-validity": EnergyValidity(optimise=True),
            "energy": AverageEnergy(),
            "energy-per-atom": AverageEnergy(per_atom=True),
            "strain": AverageStrainEnergy(),
            "strain-per-atom": AverageStrainEnergy(per_atom=True),
            "opt-rmsd": AverageOptRmsd()
        }

        if self.eval_3D_props:
            gen_metrics={**gen_metrics_2D, **gen_metrics_3D}
        else:
            gen_metrics=gen_metrics_2D

        self.stability_metrics = MetricCollection(stability_metrics, compute_groups=False)
        self.gen_metrics = MetricCollection(gen_metrics, compute_groups=False)
        self._init_params()

    def forward(self, data, t, training=False, cond_batch={"coords":None, "atomics":None, "bonds":None}, flag_3Ds=None):

        """Predict molecular coordinates and atom types
        Args:
            batch (dict[str, Tensor]): Batched pointcloud data
            t (torch.Tensor): Interpolation times between 0 and 1, shape [batch_size]
            training (bool): Whether to run forward in training mode
            cond_batch (dict[str, Tensor]): Predictions from previous step, if we are using self conditioning

        Returns:
            (predicted coordinates, atom type logits (unnormalised probabilities))
            Both torch.Tensor, shapes [batch_size, num_atoms, 3] and [batch_size, num atoms, vocab_size]
        """

        coords=data["coords"]
        atom_types=data["atomics"]
        bonds=data["bonds"]
        masks=data["masks"]
        
        # Whether to use the EMA version of the model or not
        if not training and self.ema_gen is not None:
            model = self.ema_gen
        else:
            model = self.gen
         
        t=t.view(-1,1,1) 
        cond_coords,cond_atomics,cond_bonds= cond_batch["coords"], cond_batch["atomics"], cond_batch["bonds"]

        model_kwargs = {
            "edge_feats": bonds,
            "t": t,
            "cond_coords": cond_coords,
            "cond_atomics": cond_atomics,
            "cond_bonds": cond_bonds,
            "atom_mask": masks,
            "flag_3Ds": flag_3Ds,
        }

        out = model(coords, atom_types, **model_kwargs)
        
        return out
        
    @staticmethod
    def flatten_batch(batch):
        max_atoms_in_batch= batch["natoms"].max().item()
        for key, value in batch.items():
            if key not in ["flag_3Ds"]:
                value = value.reshape(-1, *value.shape[2:])
            else:
                value = value.reshape(-1)

            if "coords" in key or "atomics" in key or "charges" in key:
                value = value [:,:max_atoms_in_batch,:]
            elif key == "coms":
                # dataloader 如果给的是 [B, N, 3]，压成 [B, 3]
                if value.dim() == 3:
                    value = value[:, 0, :]
            elif "masks" in key :
                value = value[:,:max_atoms_in_batch]
            elif "bonds" in key:
                value = value [:,:max_atoms_in_batch,:max_atoms_in_batch,:]
            else:
                pass 
            batch[key] = value
            #print (key,value.shape,value.dtype)
        return batch

    def interpolate(self, batch, t, flag_3Ds=None):
        """
        Interpolate between noise and real data at time t
        Args:
            batch: dict containing noise_coords, noise_atomics, noise_bonds, real_coords, real_atomics, real_bonds
            t: time tensor of shape (batch_size,) or scalar
        Returns:
            dict with interpolated coords, atomics, bonds
        """

        noise_coords = batch["noise_coords"]
        noise_atomics = batch["noise_atomics"]
        noise_bonds = batch["noise_bonds"]

        real_coords = batch["real_coords"]
        real_atomics = batch["real_atomics"]
        real_bonds = batch["real_bonds"]

        natoms = batch["natoms"]
        masks = batch["masks"]
        batchsize = noise_coords.size(0)
        maxatoms = noise_coords.size(1)

        # Ensure t is the right shape
        if isinstance(t, (int, float)):
            t = torch.tensor([t] * batchsize, device=noise_coords.device, dtype=noise_coords.dtype)
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(batchsize)

        # t is now shape (batchsize,)
        t_coord = t.view(-1, 1, 1)  # (batchsize, 1, 1) for coords broadcasting
        t_atom = t.view(-1, 1)      # (batchsize, 1) for atomics broadcasting
        t_bond = t.view(-1, 1, 1)   # (batchsize, 1, 1) for bonds broadcasting

        # Linear interpolation for coordinates
        interpolate_coords = (1 - t_coord) * noise_coords + t_coord * real_coords
        coords_noise = torch.randn_like(interpolate_coords) * self.default_coord_noise_std
        interpolate_coords = interpolate_coords + coords_noise

        if flag_3Ds is None:
            flag_3Ds = torch.ones(batchsize, device=noise_coords.device, dtype=noise_coords.dtype)

        flag_3Ds = flag_3Ds.view(-1, 1, 1)
        interpolate_coords = interpolate_coords * flag_3Ds

        # Discrete interpolation for atomics
        noise_atomics_idx = torch.argmax(noise_atomics, dim=-1)
        real_atomics_idx = torch.argmax(real_atomics, dim=-1)

        atom_mask = torch.rand(batchsize, maxatoms, device=noise_coords.device) > t_atom
        interpolate_atomics_idx = torch.where(atom_mask, noise_atomics_idx, real_atomics_idx)
        interpolate_atomics = one_hot_encode_tensor(interpolate_atomics_idx, noise_atomics.size(-1))

        # Discrete interpolation for bonds
        noise_bonds_idx = torch.argmax(noise_bonds, dim=-1)
        real_bonds_idx = torch.argmax(real_bonds, dim=-1)
        bond_mask = torch.rand(batchsize, maxatoms, maxatoms, device=noise_coords.device) > t_bond

        interpolate_bonds_idx = torch.where(bond_mask, noise_bonds_idx, real_bonds_idx)
        interpolate_bonds = one_hot_encode_tensor(interpolate_bonds_idx, noise_bonds.size(-1))

        return {"coords": interpolate_coords, "atomics": interpolate_atomics, "bonds": interpolate_bonds, "masks": masks}


    def FM_training_step(self, batch):
        """
        Standard Flow Matching training step with randomized dt:
        - dt: sampled randomly from [0, log2(max_steps)] to cover all resolutions
        - t: sampled on the grid defined by dt
        """
        batchsize = batch["natoms"].size(0)
        device = batch["real_coords"].device

        t = self.time_dist.sample((batchsize,)).to(device)

        flag_3Ds = batch["flag_3Ds"]
        # Interpolate to get x_t
        interp_data = self.interpolate(batch, t, flag_3Ds=flag_3Ds)

        # Pass the randomized dt indices to the model
        cond_batch = None

        # If training with self conditioning, half the time generate a conditional batch by setting cond to zeros
        if self.self_cond:
            cond_batch = {
                "coords": torch.zeros_like(interp_data["coords"]),
                "atomics": torch.zeros_like(interp_data["atomics"]),
                "bonds": torch.zeros_like(interp_data["bonds"])
            }

            if torch.rand(1).item() > 0.5:
                with torch.no_grad():
                    cond_coords, cond_types, cond_bonds, _ = self(
                        interp_data,
                        t,
                        training=True,
                        cond_batch=cond_batch,
                        flag_3Ds=flag_3Ds,
                    )
                    
                    cond_batch = {
                        "coords": cond_coords * flag_3Ds.view(-1,1,1),
                        "atomics": F.softmax(cond_types, dim=-1),
                        "bonds": F.softmax(cond_bonds, dim=-1)
                    }

        coords, types, bonds, charges = self(
            interp_data,
            t,
            training=True,
            cond_batch=cond_batch,
            flag_3Ds=flag_3Ds,
        )

        predicted = {
            "coords": coords,
            "atomics": types,
            "bonds": bonds,
            "charges": charges
        }

        if self.formulation=="endpoint":
            coords_target=batch["real_coords"]
        else:
            coords_target=batch["real_coords"]-batch["noise_coords"]

        # Prepare targets
        target = {
            "coords": coords_target,
            "atomics": batch["real_atomics"],
            "bonds": batch["real_bonds"],
            "charges": batch["real_charges"],
            "masks": batch["masks"]
        }

        # Calculate losses
        losses = self._loss(target, predicted, flag_3Ds=flag_3Ds)
        fm_loss = sum(list(losses.values()))

        # Log individual losses
        for name, loss_val in losses.items():
            self.log(f"train-fm-{name}", loss_val, on_step=True, logger=True)

        return fm_loss

    def training_step(self, batch, b_idx):
        """
        Combined training step with both FM and SC losses
        FM and SC use independent sampling strategies (diff.py style)
        """
        batch = self.flatten_batch(batch)

        # FM loss (uses its own uniform t sampling + fixed dt)
        fm_loss = self.FM_training_step(batch)
        self.log("train-fm-loss", fm_loss, prog_bar=True, on_step=True, logger=True)

        # Log combined metrics        
        self.log("train-loss", fm_loss, prog_bar=True, on_step=True, logger=True)
        return fm_loss

    def on_train_batch_end(self, outputs, batch, b_idx):
        if self.ema_gen is not None:
            self.ema_gen.update_parameters(self.gen)

    def validation_step(self, batch, b_idx):
        batch= self.flatten_batch(batch)
        
        noise={"coords":batch["noise_coords"],
               "atomics":batch["noise_atomics"],
               "bonds":batch["noise_bonds"],
               "masks":batch["masks"],
               "flag_3Ds":batch["flag_3Ds"],}
        
        gen_batch = self._generate(noise, inference_steps=self.max_steps,
                                   coord_noise_std=self.default_coord_noise_std,
                                   cat_noise_level=self.default_cat_noise_level,)
        
        stabilities = self._generate_stabilities(gen_batch)
        gen_mols = self._generate_mols(gen_batch)

        self.stability_metrics.update(stabilities)
        self.gen_metrics.update(gen_mols)

    def on_validation_epoch_end(self):
        stability_metrics_results = self.stability_metrics.compute()
        gen_metrics_results = self.gen_metrics.compute()

        metrics = {
            **stability_metrics_results, 
            **gen_metrics_results,
        }

        for metric, value in metrics.items():
            progbar = True if metric == "validity" else False
            self.log(f"val-{metric}", value, on_epoch=True, logger=True, prog_bar=progbar)

        self.stability_metrics.reset()
        self.gen_metrics.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def predict_step(self, batch, batch_idx):
        batch= self.flatten_batch(batch)

        noise={
            "coords":batch["noise_coords"],
            "atomics":batch["noise_atomics"],
            "bonds":batch["noise_bonds"],
            "masks":batch["masks"],
            "flag_3Ds":batch["flag_3Ds"],
        }

        gen_batch = self._generate(noise, inference_steps=self.default_inference_steps, 
                                   coord_noise_std=self.default_coord_noise_std,
                                   cat_noise_level= self.default_cat_noise_level,)
        gen_mols = self._generate_mols(gen_batch)
        return gen_mols

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.gen.parameters(),
            lr=self.lr,
            amsgrad=True,
            foreach=True,
            weight_decay=0.0
        )

        warm_up_steps = 0 if self.warm_up_steps is None else self.warm_up_steps
        scheduler = LinearLR(opt, start_factor=1e-2, total_iters=warm_up_steps)

        config = {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }
        return config

    def _loss(self, target, predicted, flag_3Ds=None):
        pred_coords = predicted["coords"]
        target_coords = target["coords"]
        mask = target["masks"].unsqueeze(2)

        coord_loss = F.mse_loss(pred_coords, target_coords, reduction="none")
        coord_loss = (coord_loss * mask).mean(dim=(1, 2))
        coord_loss = coord_loss * flag_3Ds.view(-1)
        denom = flag_3Ds.sum().clamp_min(1.0)
        coord_loss = coord_loss.sum() / denom

        type_loss = self._type_loss(target,  predicted)
        bond_loss = self._bond_loss(target,  predicted)
        charge_loss = self._charge_loss(target, predicted)

        coord_loss = coord_loss.mean()
        type_loss = type_loss.mean()*self.loss_weight["types"]
        bond_loss = bond_loss.mean() * self.loss_weight["bonds"]
        charge_loss = charge_loss.mean() * self.loss_weight["charges"]

        losses = {
            "coord-loss": coord_loss,
            "type-loss": type_loss,
            "bond-loss": bond_loss,
            "charge-loss": charge_loss
        }
        return losses

    def _type_loss(self, target,  predicted, eps=1e-3):
        pred_logits = predicted["atomics"]
        atomics_dist = target["atomics"]
        mask = target["masks"].unsqueeze(2)
        batch_size, num_atoms, _ = pred_logits.size()

        atomics = torch.argmax(atomics_dist, dim=-1).flatten(0, 1)
        type_loss = F.cross_entropy(pred_logits.flatten(0, 1), atomics, reduction="none")
        type_loss = type_loss.unflatten(0, (batch_size, num_atoms)).unsqueeze(2)

        n_atoms = mask.sum(dim=(1, 2)) + eps
        type_loss = (type_loss * mask).sum(dim=(1, 2)) / n_atoms
        return type_loss

    def _bond_loss(self, target,  predicted, eps=1e-3):
        pred_logits = predicted["bonds"]
        mask = target["masks"]
        bonds = torch.argmax(target["bonds"], dim=-1)
        batch_size, num_atoms, _, _ = pred_logits.size()

        bond_loss = F.cross_entropy(pred_logits.flatten(0, 2), bonds.flatten(0, 2), reduction="none")
        bond_loss = bond_loss.unflatten(0, (batch_size, num_atoms, num_atoms))

        adj_matrix = adj_from_node_mask(mask, self_connect=True)
        n_bonds = adj_matrix.sum(dim=(1, 2)) + eps

        bond_loss = (bond_loss * adj_matrix).sum(dim=(1, 2)) / n_bonds
        return bond_loss

    def _charge_loss(self, target, predicted, eps=1e-3):
        pred_logits = predicted["charges"]
        charges = target["charges"]
        mask = target["masks"]
        batch_size, num_atoms, _ = pred_logits.size()

        charges = torch.argmax(charges, dim=-1).flatten(0, 1)
        charge_loss = F.cross_entropy(pred_logits.flatten(0, 1), charges, reduction="none")
        charge_loss = charge_loss.unflatten(0, (batch_size, num_atoms))

        n_atoms = mask.sum(dim=1) + eps
        charge_loss = (charge_loss * mask).sum(dim=1) / n_atoms
        return charge_loss

    def _generate(self, noise,  inference_steps=100, coord_noise_std=0.0, cat_noise_level=1.0, eps=1e-5, coms=None):
        
        time_points = np.linspace(0, 1, inference_steps + 1).tolist()

        times = torch.zeros(noise["coords"].size(0), device=self.device)
        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]
        curr = {k: v.clone() for k, v in noise.items()}
        flag_3Ds= noise["flag_3Ds"]
        cond_batch = {
            "coords": torch.zeros_like(noise["coords"]),
            "atomics": torch.zeros_like(noise["atomics"]),
            "bonds": torch.zeros_like(noise["bonds"])
        }

        with torch.no_grad():
            for step_size in step_sizes:
                cond = cond_batch if self.self_cond else None
                coords, type_logits, bond_logits, charge_logits = self(
                    curr,
                    times,
                    training=False,
                    cond_batch=cond,
                    flag_3Ds=flag_3Ds,
                )

                type_probs = F.softmax(type_logits, dim=-1)
                bond_probs = F.softmax(bond_logits, dim=-1)
                charge_probs = F.softmax(charge_logits, dim=-1)

                cond_batch = {
                    "coords": coords*flag_3Ds.view(-1,1,1),
                    "atomics": type_probs,
                    "bonds": bond_probs,
                }
                predicted = {
                    "coords": coords*flag_3Ds.view(-1,1,1),
                    "atomics": type_probs,
                    "bonds": bond_probs,
                    "charges": charge_probs,
                    "masks": curr["masks"],
                    "flag_3Ds": flag_3Ds,
                }

                curr = self._integrate_step(
                    curr,
                    predicted,
                    noise,
                    times,
                    step_size,
                    coord_noise_std,
                    cat_noise_level,
                    flag_3Ds=flag_3Ds,
                )
                times = times + step_size

        if self.formulation == "endpoint":
            predicted["coords"] = predicted["coords"] * self.coord_scale
        else:
            predicted["coords"] = curr["coords"] * self.coord_scale

        # if coms is not None:
        #     predicted["coords"] = predicted["coords"] + coms 
        if coms is not None:
            if coms.dim() == 2:
                coms = coms.unsqueeze(1)
            elif coms.dim() == 3 and coms.size(1) != 1:
                coms = coms[:, :1, :]
            predicted["coords"] = predicted["coords"] + coms

        return predicted

    def _integrate_step(self, curr, predicted, prior, t, step_size, coord_noise_std, cat_noise_level, flag_3Ds=None):
        device = curr["coords"].device
        vocab_size = predicted["atomics"].size(-1)
        n_bonds = predicted["bonds"].size(-1)

        # *** Coord update step ***
        if self.formulation=="endpoint":
            coord_velocity = (predicted["coords"] - curr["coords"]) / (1 - t.view(-1, 1, 1))
            coord_velocity += (torch.randn_like(coord_velocity) * coord_noise_std)
            coords = curr["coords"] + (step_size * coord_velocity)
        else:
            coord_velocity = predicted["coords"]
            coord_velocity += (torch.randn_like(coord_velocity) * coord_noise_std)
            coords = curr["coords"] + (step_size * coord_velocity)

        coords = coords * flag_3Ds.view(-1,1,1)

        # *** Atomic update step ***

        atomics = self._uniform_sample_step(curr["atomics"], predicted["atomics"], t, step_size, cat_noise_level=cat_noise_level)

        # *** Bond update step ***

        bonds = self._uniform_sample_step(curr["bonds"], predicted["bonds"], t, step_size, cat_noise_level=cat_noise_level)

        updated = {
            "coords": coords,
            "atomics": atomics,
            "bonds": bonds,
            "masks": curr["masks"]
        }
        return updated
    
    def _uniform_sample_step(self, curr_dist, pred_dist, t, step_size, cat_noise_level=None):
        n_categories = pred_dist.size(-1)

        curr = torch.argmax(curr_dist, dim=-1).unsqueeze(-1)
        pred_probs_curr = torch.gather(pred_dist, -1, curr)

        # Setup batched time tensor and noise tensor
        ones = [1] * (len(pred_dist.shape) - 1)
        times = t.view(-1, *ones).clamp(min=self.eps, max=1.0 - self.eps)
        noise = torch.zeros_like(times)
        noise[times + step_size < 1.0] = cat_noise_level

        # Off-diagonal step probs
        mult = ((1 + ((2 * noise) * (n_categories - 1) * times)) / (1 - times))
        first_term = step_size * mult * pred_dist
        second_term = step_size * noise * pred_probs_curr
        step_probs = (first_term + second_term).clamp(max=1.0)

        # On-diagonal step probs
        step_probs.scatter_(-1, curr, 0.0)
        diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        step_probs.scatter_(-1, curr, diags)

        # Sample and convert back to one-hot so that all strategies represent data the same way
        samples = torch.distributions.Categorical(step_probs).sample()
        return one_hot_encode_tensor(samples, n_categories)

    def _generate_mols(self, generated, sanitise=True):
        coords = generated["coords"]
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["masks"]
        flag_3Ds = generated["flag_3Ds"]

        mols = self.builder.mols_from_tensors(
            coords,
            atom_dists,
            masks,
            bond_dists=bond_dists,
            charge_dists=charge_dists,
            sanitise=sanitise,
        )

        return mols

    def _generate_stabilities(self, generated):
        coords = generated["coords"]
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["masks"]
        stabilities = self.builder.mol_stabilities(coords, atom_dists, masks, bond_dists, charge_dists)
        return stabilities

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)


