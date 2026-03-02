import copy
from typing import Optional

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import QED
from .diff import SC_Lightning

class RL_Lightning(SC_Lightning):
    def __init__(
        self,
        gen: torch.nn.Module,
        vocab,
        lr: float,
        coord_scale: float = 1.0,
        use_ema: bool = True,
        compile_model: bool = True,
        warm_up_steps: Optional[int] = None,
        max_steps: int = 128,
        default_coord_noise_std: float = 0.2,
        default_cat_noise_level: float = 1.0,
        self_cond: bool = False,
        loss_weight: dict = {"types": 0.2, "bonds": 1.0, "charges": 1.0},
        formulation: str = "endpoint",
        eval_3D_props: bool = True,
        reward_name: str = "qed",
        reward_beta: float = 2.0,
        reward_weight_min: float = 0.1,
        reward_weight_max: float = 10.0,
        reward_norm_eps: float = 1e-6,
        anchor_weight: float = 0.1,
        anchor_loss_weight: float = 1.0,
        use_reference_anchor: bool = True,
    ):
        super().__init__(
            gen=gen,
            vocab=vocab,
            lr=lr,
            coord_scale=coord_scale,
            use_ema=use_ema,
            compile_model=compile_model,
            warm_up_steps=warm_up_steps,
            max_steps=max_steps,
            default_coord_noise_std=default_coord_noise_std,
            default_cat_noise_level=default_cat_noise_level,
            self_cond=self_cond,
            loss_weight=loss_weight,
            formulation=formulation,
            eval_3D_props=eval_3D_props,
        )

        self.reward_name = reward_name
        self.reward_beta = reward_beta
        self.reward_weight_min = reward_weight_min
        self.reward_weight_max = reward_weight_max
        self.reward_norm_eps = reward_norm_eps
        self.anchor_weight = anchor_weight
        self.anchor_loss_weight = anchor_loss_weight
        self.use_reference_anchor = use_reference_anchor
        self.ref_gen = None

    def on_fit_start(self):
        super().on_fit_start()
        if self.use_reference_anchor and self.ref_gen is None:
            self.ref_gen = copy.deepcopy(self.gen)
            self.ref_gen.eval()
            for param in self.ref_gen.parameters():
                param.requires_grad = False

    def _build_noise_batch(self, batch):
        return {
            "coords": batch["noise_coords"],
            "atomics": batch["noise_atomics"],
            "bonds": batch["noise_bonds"],
            "masks": batch["masks"],
            "flag_3Ds": batch["flag_3Ds"],
        }

    def _build_generated_target_batch(self, batch, generated):
        target_coords = generated["coords"] / max(self.coord_scale, self.reward_norm_eps)

        return {
            "noise_coords": batch["noise_coords"],
            "noise_atomics": batch["noise_atomics"],
            "noise_bonds": batch["noise_bonds"],
            "real_coords": target_coords,
            "real_atomics": generated["atomics"],
            "real_bonds": generated["bonds"],
            "real_charges": generated["charges"],
            "masks": batch["masks"],
            "natoms": batch["natoms"],
            "flag_3Ds": batch["flag_3Ds"],
        }

    def _compute_rewards_from_generated(self, generated):
        mols = self._generate_mols(generated, sanitise=True)
        return self._compute_rewards_from_mols(mols, dtype=generated["coords"].dtype, device=generated["coords"].device), mols

    def _compute_rewards_from_mols(self, mols, dtype, device):
        rewards = []
        for mol in mols:
            if mol is None:
                rewards.append(0.0)
                continue

            if self.reward_name == "qed":
                try:
                    rewards.append(float(QED.qed(mol)))
                except Exception:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)

        return torch.tensor(rewards, dtype=dtype, device=device)

    def _compute_generation_quality_from_mols(self, mols, dtype, device):
        total = max(len(mols), 1)
        valid_mols = [mol for mol in mols if mol is not None]
        n_valid = len(valid_mols)

        smiles = []
        n_connected = 0
        for mol in valid_mols:
            try:
                smi = Chem.MolToSmiles(mol)
                smiles.append(smi)
            except Exception:
                pass

            try:
                if len(Chem.GetMolFrags(mol)) == 1:
                    n_connected += 1
            except Exception:
                pass

        validity = n_valid / total
        uniqueness = len(set(smiles)) / max(len(smiles), 1)
        connected_validity = n_connected / max(n_valid, 1)

        return {
            "validity": torch.tensor(validity, dtype=dtype, device=device),
            "uniqueness": torch.tensor(uniqueness, dtype=dtype, device=device),
            "connected-validity": torch.tensor(connected_validity, dtype=dtype, device=device),
            "n-valid": torch.tensor(float(n_valid), dtype=dtype, device=device),
            "n-total": torch.tensor(float(total), dtype=dtype, device=device),
        }

    def _reward_to_weights(self, rewards):
        centered = rewards - rewards.mean()
        normed = centered / (centered.std(unbiased=False) + self.reward_norm_eps)
        weights = torch.exp(self.reward_beta * normed)
        weights = torch.clamp(weights, min=self.reward_weight_min, max=self.reward_weight_max)
        return weights

    def _loss_per_sample(self, target, predicted, flag_3Ds=None):
        pred_coords = predicted["coords"]
        target_coords = target["coords"]
        mask = target["masks"].unsqueeze(2)

        coord_loss = F.mse_loss(pred_coords, target_coords, reduction="none")
        coord_loss = (coord_loss * mask).mean(dim=(1, 2))
        coord_loss = coord_loss * flag_3Ds.view(-1)

        type_loss = self._type_loss(target, predicted)
        bond_loss = self._bond_loss(target, predicted)
        charge_loss = self._charge_loss(target, predicted)

        type_loss = type_loss * self.loss_weight["types"]
        bond_loss = bond_loss * self.loss_weight["bonds"]
        charge_loss = charge_loss * self.loss_weight["charges"]

        return {
            "coord-loss": coord_loss,
            "type-loss": type_loss,
            "bond-loss": bond_loss,
            "charge-loss": charge_loss,
        }

    def _forward_with_model(self, model, data, t, cond_batch=None, flag_3Ds=None):
        coords = data["coords"]
        atom_types = data["atomics"]
        bonds = data["bonds"]
        masks = data["masks"]

        t = t.view(-1, 1, 1)
        if cond_batch is None:
            cond_coords, cond_atomics, cond_bonds = None, None, None
        else:
            cond_coords = cond_batch["coords"]
            cond_atomics = cond_batch["atomics"]
            cond_bonds = cond_batch["bonds"]

        return model(
            coords,
            atom_types,
            edge_feats=bonds,
            t=t,
            cond_coords=cond_coords,
            cond_atomics=cond_atomics,
            cond_bonds=cond_bonds,
            atom_mask=masks,
            flag_3Ds=flag_3Ds,
        )

    def _anchor_loss_per_sample(self, interp_data, t, predicted, cond_batch=None, flag_3Ds=None):
        if (not self.use_reference_anchor) or (self.ref_gen is None) or (self.anchor_weight <= 0):
            zeros = torch.zeros(interp_data["coords"].size(0), device=interp_data["coords"].device)
            return {
                "coord": zeros,
                "types": zeros,
                "bonds": zeros,
                "charges": zeros,
            }

        with torch.no_grad():
            ref_coords, ref_types, ref_bonds, ref_charges = self._forward_with_model(
                self.ref_gen,
                interp_data,
                t,
                cond_batch=cond_batch,
                flag_3Ds=flag_3Ds,
            )

        mask = interp_data["masks"]
        mask3 = mask.unsqueeze(-1)
        n_atoms = mask.sum(dim=1).clamp_min(1.0)

        coord_kl = F.mse_loss(predicted["coords"], ref_coords, reduction="none")
        coord_kl = (coord_kl * mask3).sum(dim=(1, 2)) / n_atoms
        coord_kl = coord_kl * flag_3Ds.view(-1)

        type_kl = F.kl_div(
            F.log_softmax(predicted["atomics"], dim=-1),
            F.softmax(ref_types, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        type_kl = (type_kl * mask).sum(dim=1) / n_atoms

        charge_kl = F.kl_div(
            F.log_softmax(predicted["charges"], dim=-1),
            F.softmax(ref_charges, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        charge_kl = (charge_kl * mask).sum(dim=1) / n_atoms

        bond_mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).float()
        n_bonds = bond_mask.sum(dim=(1, 2)).clamp_min(1.0)
        bond_kl = F.kl_div(
            F.log_softmax(predicted["bonds"], dim=-1),
            F.softmax(ref_bonds, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        bond_kl = (bond_kl * bond_mask).sum(dim=(1, 2)) / n_bonds

        return {
            "coord": coord_kl,
            "types": type_kl,
            "bonds": bond_kl,
            "charges": charge_kl,
        }

    def FM_training_step(self, batch):
        noise = self._build_noise_batch(batch)
        with torch.no_grad():
            generated = self._generate(
                noise,
                inference_steps=self.max_steps,
                coord_noise_std=self.default_coord_noise_std,
                cat_noise_level=self.default_cat_noise_level,
            )

        rewards, generated_mols = self._compute_rewards_from_generated(generated)

        quality_metrics = self._compute_generation_quality_from_mols(
            generated_mols,
            dtype=generated["coords"].dtype,
            device=generated["coords"].device,
        )

        weights = self._reward_to_weights(rewards)
        weights = weights / weights.mean().clamp_min(self.reward_norm_eps)

        train_batch = self._build_generated_target_batch(batch, generated)

        batchsize = train_batch["natoms"].size(0)
        device = train_batch["real_coords"].device

        t = self.time_dist.sample((batchsize,)).to(device)
        flag_3Ds = train_batch["flag_3Ds"]

        interp_data = self.interpolate(train_batch, t, flag_3Ds=flag_3Ds)

        cond_batch = None
        if self.self_cond:
            cond_batch = {
                "coords": torch.zeros_like(interp_data["coords"]),
                "atomics": torch.zeros_like(interp_data["atomics"]),
                "bonds": torch.zeros_like(interp_data["bonds"]),
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
                        "coords": cond_coords * flag_3Ds.view(-1, 1, 1),
                        "atomics": F.softmax(cond_types, dim=-1),
                        "bonds": F.softmax(cond_bonds, dim=-1),
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
            "charges": charges,
        }

        if self.formulation == "endpoint":
            coords_target = train_batch["real_coords"]
        else:
            coords_target = train_batch["real_coords"] - train_batch["noise_coords"]

        target = {
            "coords": coords_target,
            "atomics": train_batch["real_atomics"],
            "bonds": train_batch["real_bonds"],
            "charges": train_batch["real_charges"],
            "masks": train_batch["masks"],
        }

        losses = self._loss_per_sample(target, predicted, flag_3Ds=flag_3Ds)

        weighted_losses = {
            name: (loss_vals * weights).mean() for name, loss_vals in losses.items()
        }

        anchor_losses = self._anchor_loss_per_sample(
            interp_data,
            t,
            predicted,
            cond_batch=cond_batch,
            flag_3Ds=flag_3Ds,
        )

        anchor_loss = (
            anchor_losses["coord"]
            + anchor_losses["types"]
            + anchor_losses["bonds"]
            + anchor_losses["charges"]
        ).mean()

        fm_loss = sum(list(weighted_losses.values()))
        total_loss = fm_loss + (self.anchor_weight * self.anchor_loss_weight) * anchor_loss

        raw_losses = {name: loss_vals.mean() for name, loss_vals in losses.items()}

        for name, loss_val in weighted_losses.items():
            self.log(f"train-fm-{name}", loss_val, on_step=True, logger=True, sync_dist=True)
        for name, loss_val in raw_losses.items():
            self.log(f"train-fm-raw-{name}", loss_val, on_step=True, logger=True, sync_dist=True)

        self.log("train-rl-reward-mean", rewards.mean(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-rl-reward-max", rewards.max(), on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-weight-mean", weights.mean(), on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-weight-max", weights.max(), on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-anchor-loss", anchor_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-fm-loss", fm_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-total-loss", total_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-gen-validity", quality_metrics["validity"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-uniqueness", quality_metrics["uniqueness"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-connected-validity", quality_metrics["connected-validity"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-n-valid", quality_metrics["n-valid"], on_step=True, logger=True, sync_dist=True)
        self.log("train-gen-n-total", quality_metrics["n-total"], on_step=True, logger=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, b_idx):
        return

    def on_validation_epoch_end(self):
        return

    def test_step(self, batch, batch_idx):
        return

    def on_test_epoch_end(self):
        return
    
    

