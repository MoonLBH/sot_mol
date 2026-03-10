import copy
from typing import Optional

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import QED
import os
import numpy as np

from .diff import SC_Lightning


class GRPO_Lightning(SC_Lightning):
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
        group_size: int = 8,
        clip_eps: float = 0.2,
        kl_beta: float = 1e-3,
        sde_noise_scale: float = 0.7,
        reward_norm_eps: float = 1e-6,
        ratio_max: float = 20.0,
        use_reference_policy: bool = True,
        cache_on_cpu: bool = True,
        grad_clip_val: float = 1.0,
        val_save_path: Optional[str] = None,
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
        self.group_size = max(1, int(group_size))
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        self.sde_noise_scale = sde_noise_scale
        self.reward_norm_eps = reward_norm_eps
        self.ratio_max = ratio_max
        self.use_reference_policy = use_reference_policy
        self.cache_on_cpu = cache_on_cpu
        self.grad_clip_val = grad_clip_val
        self.val_save_path = val_save_path
        self.coord_noise_std = default_coord_noise_std

        self.ref_gen = None
        # Use manual optimization so each transition can backward independently,
        # avoiding a single giant graph over all rollout steps.
        self.automatic_optimization = False

    def on_fit_start(self):
        super().on_fit_start()
        if self.use_reference_policy and self.ref_gen is None:
            self.ref_gen = copy.deepcopy(self.gen)
            self.ref_gen.eval()
            for param in self.ref_gen.parameters():
                param.requires_grad = False

    def _cache_tensor(self, tensor):
        if tensor is None:
            return None
        tensor = tensor.detach()
        if self.cache_on_cpu:
            return tensor.cpu()
        return tensor

    def _transition_to_device(self, transition, device):
        if not self.cache_on_cpu:
            return transition

        moved = {}
        for key, value in transition.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    def _compute_rewards_from_generated(self, generated):
        mols = self._generate_mols(generated, sanitise=True)
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

        rewards = torch.tensor(
            rewards,
            dtype=generated["coords"].dtype,
            device=generated["coords"].device,
        )
        return rewards, mols

    def _compute_generation_quality_from_mols(self, mols, dtype, device):
        total = max(len(mols), 1)
        valid_mols = [mol for mol in mols if mol is not None]
        n_valid = len(valid_mols)

        smiles = []
        n_connected = 0
        for mol in valid_mols:
            try:
                smiles.append(Chem.MolToSmiles(mol))
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

    def _coord_mean_update(self, curr_coords, pred_coords, times, step_size):
        if self.formulation == "endpoint":
            velocity = (pred_coords - curr_coords) / (1 - times.view(-1, 1, 1)).clamp_min(self.eps)
        else:
            velocity = pred_coords

        return curr_coords + (step_size * velocity)

    def _coord_std(self, times, step_size):
        t = times.view(-1, 1, 1).clamp(min=self.eps, max=1.0 - self.eps)
        sigma_t = self.sde_noise_scale * torch.sqrt(t / (1.0 - t + self.eps))
        return sigma_t * (step_size ** 0.5)

    def _categorical_step_probs(self, curr_dist, pred_dist, times, step_size, cat_noise_level):
        n_categories = pred_dist.size(-1)

        curr = torch.argmax(curr_dist, dim=-1).unsqueeze(-1)
        pred_probs_curr = torch.gather(pred_dist, -1, curr)

        ones = [1] * (len(pred_dist.shape) - 1)
        times = times.view(-1, *ones).clamp(min=self.eps, max=1.0 - self.eps)
        step_size = step_size.view(-1, *ones)
        noise = torch.zeros_like(times)
        noise[times + step_size < 1.0] = cat_noise_level

        mult = (1 + ((2 * noise) * (n_categories - 1) * times)) / (1 - times)
        first_term = step_size * mult * pred_dist
        second_term = step_size * noise * pred_probs_curr
        step_probs = (first_term + second_term).clamp(max=1.0)

        step_probs.scatter_(-1, curr, 0.0)
        diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        step_probs.scatter_(-1, curr, diags)

        return step_probs

    def _gaussian_logprob(self, sample, mean, std, masks, flag_3Ds=None):
        safe_std = std.clamp_min(self.reward_norm_eps)
        var = safe_std * safe_std
        log2pi = torch.log(torch.tensor(2.0 * torch.pi, device=sample.device, dtype=sample.dtype))
        logp = -0.5 * (((sample - mean) ** 2) / var + 2.0 * torch.log(safe_std) + log2pi)

        atom_mask = masks.unsqueeze(-1).float()
        logp = (logp * atom_mask).sum(dim=(1, 2))

        if flag_3Ds is not None:
            logp = logp * flag_3Ds.view(-1)

        return logp

    def _categorical_logprob(self, sample_idx, probs, masks, pairwise=False):
        safe_probs = probs.clamp_min(self.reward_norm_eps)
        sample_idx = sample_idx.long().unsqueeze(-1)
        chosen = torch.gather(safe_probs, dim=-1, index=sample_idx).squeeze(-1)
        logp = torch.log(chosen)

        if pairwise:
            pair_mask = (masks.unsqueeze(1) * masks.unsqueeze(2)).float()
            logp = (logp * pair_mask).sum(dim=(1, 2))
        else:
            atom_mask = masks.float()
            logp = (logp * atom_mask).sum(dim=1)

        return logp

    def _categorical_kl(self, probs_new, probs_ref, masks, pairwise=False):
        kl = F.kl_div(
            torch.log(probs_new.clamp_min(self.reward_norm_eps)),
            probs_ref.clamp_min(self.reward_norm_eps),
            reduction="none",
        ).sum(dim=-1)

        if pairwise:
            pair_mask = (masks.unsqueeze(1) * masks.unsqueeze(2)).float()
            denom = pair_mask.sum(dim=(1, 2)).clamp_min(1.0)
            kl = (kl * pair_mask).sum(dim=(1, 2)) / denom
        else:
            atom_mask = masks.float()
            denom = atom_mask.sum(dim=1).clamp_min(1.0)
            kl = (kl * atom_mask).sum(dim=1) / denom

        return kl

    def _coord_kl(self, mean_new, mean_ref, std, masks, flag_3Ds=None):
        kl = ((mean_new - mean_ref) ** 2) / (2.0 * (std ** 2).clamp_min(self.reward_norm_eps))
        atom_mask = masks.unsqueeze(-1).float()
        denom = atom_mask.sum(dim=(1, 2)).clamp_min(1.0)
        kl = (kl * atom_mask).sum(dim=(1, 2)) / denom
        if flag_3Ds is not None:
            kl = kl * flag_3Ds.view(-1)
        return kl

    def _group_advantages(self, rewards):
        advantages = torch.zeros_like(rewards)
        n = rewards.size(0)

        for start in range(0, n, self.group_size):
            end = min(start + self.group_size, n)
            group_rewards = rewards[start:end]
            mean = group_rewards.mean()
            std = group_rewards.std(unbiased=False)
            advantages[start:end] = (group_rewards - mean) / (std + self.reward_norm_eps)

        return advantages

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

    def _collect_rollout(self, noise, model, inference_steps, cat_noise_level):
        time_points = np.linspace(0, 1, inference_steps + 1).tolist()
        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]

        curr = {k: v.clone() for k, v in noise.items()}
        times = torch.zeros(noise["coords"].size(0), device=noise["coords"].device)
        flag_3Ds = noise["flag_3Ds"]

        cond_batch = {
            "coords": torch.zeros_like(noise["coords"]),
            "atomics": torch.zeros_like(noise["atomics"]),
            "bonds": torch.zeros_like(noise["bonds"]),
        }

        transitions = []

        with torch.no_grad():
            for step_size in step_sizes:
                cond = cond_batch if self.self_cond else None
                coords, type_logits, bond_logits, charge_logits = self._forward_with_model(
                    model,
                    curr,
                    times,
                    cond_batch=cond,
                    flag_3Ds=flag_3Ds,
                )

                type_probs = F.softmax(type_logits, dim=-1)
                bond_probs = F.softmax(bond_logits, dim=-1)
                charge_probs = F.softmax(charge_logits, dim=-1)
                
                coord_velocity = (coords - curr["coords"]) / (1 - times.view(-1, 1, 1)).clamp_min(self.eps)
                coord_mean = curr["coords"] + step_size * coord_velocity
                step_size_tensor = torch.full_like(times, step_size)
                coord_std = self._coord_std(times, step_size_tensor.view(-1, 1, 1))
                coord_noise = torch.randn_like(coord_mean) * coord_std
                next_coords = (coord_mean + coord_noise)

                coord_logprob = self._gaussian_logprob(
                    next_coords,
                    coord_mean,
                    coord_std,
                    curr["masks"],
                    flag_3Ds=flag_3Ds,
                )

                type_step_probs = self._categorical_step_probs(
                    curr["atomics"],
                    type_probs,
                    times,
                    step_size_tensor,
                    cat_noise_level=cat_noise_level,
                )
                bond_step_probs = self._categorical_step_probs(
                    curr["bonds"],
                    bond_probs,
                    times,
                    step_size_tensor,
                    cat_noise_level=cat_noise_level,
                )

                next_atomics = self._uniform_sample_step(
                    curr["atomics"],
                    type_probs,
                    times,
                    step_size,
                    cat_noise_level=cat_noise_level,
                )
                next_bonds = self._uniform_sample_step(
                    curr["bonds"],
                    bond_probs,
                    times,
                    step_size,
                    cat_noise_level=cat_noise_level,
                )

                next_atomics_idx = torch.argmax(next_atomics, dim=-1)
                next_bonds_idx = torch.argmax(next_bonds, dim=-1)

                atom_logprob = self._categorical_logprob(next_atomics_idx, type_step_probs, curr["masks"], pairwise=False)
                bond_logprob = self._categorical_logprob(next_bonds_idx, bond_step_probs, curr["masks"], pairwise=True)
                old_logprob = coord_logprob + atom_logprob + bond_logprob

                transitions.append(
                    {
                        "coords": self._cache_tensor(curr["coords"]),
                        "atomics": self._cache_tensor(curr["atomics"]),
                        "bonds": self._cache_tensor(curr["bonds"]),
                        "masks": self._cache_tensor(curr["masks"]),
                        "flag_3Ds": self._cache_tensor(flag_3Ds),
                        "cond_coords": self._cache_tensor(None if cond is None else cond["coords"]),
                        "cond_atomics": self._cache_tensor(None if cond is None else cond["atomics"]),
                        "cond_bonds": self._cache_tensor(None if cond is None else cond["bonds"]),
                        "times": self._cache_tensor(times),
                        "step_size": self._cache_tensor(step_size_tensor),
                        "next_coords": self._cache_tensor(next_coords),
                        "next_atomics_idx": self._cache_tensor(next_atomics_idx),
                        "next_bonds_idx": self._cache_tensor(next_bonds_idx),
                        "old_logprob": self._cache_tensor(old_logprob),
                    }
                )

                cond_batch = {
                    "coords": coords * flag_3Ds.view(-1, 1, 1),
                    "atomics": type_probs,
                    "bonds": bond_probs,
                }

                curr = {
                    "coords": next_coords,
                    "atomics": next_atomics,
                    "bonds": next_bonds,
                    "masks": curr["masks"],
                    "flag_3Ds": flag_3Ds,
                }
                times = times + step_size

        generated = {
            "coords": curr["coords"] * self.coord_scale,
            "atomics": curr["atomics"],
            "bonds": curr["bonds"],
            "charges": charge_probs,
            "masks": curr["masks"],
            "flag_3Ds": flag_3Ds,
        }
        return generated, transitions

    def _transition_step_stats(self, transition):
        state = {
            "coords": transition["coords"],
            "atomics": transition["atomics"],
            "bonds": transition["bonds"],
            "masks": transition["masks"],
        }

        times = transition["times"]
        step_size = transition["step_size"]
        flag_3Ds = transition["flag_3Ds"]

        cond_batch = {
            "coords": transition["cond_coords"],
            "atomics": transition["cond_atomics"],
            "bonds": transition["cond_bonds"],
        }

        # Merge current-policy computations into a single forward pass per transition.
        coords_new, type_logits_new, bond_logits_new, _ = self(
            state,
            times,
            training=True,
            cond_batch=cond_batch,
            flag_3Ds=flag_3Ds,
        )

        coord_velocity = (coords_new - state["coords"]) / (1 - times.view(-1, 1, 1)).clamp_min(self.eps)
        coord_mean = state["coords"] + step_size.view(-1, 1, 1) * coord_velocity
        coord_std = self._coord_std(times, step_size.view(-1, 1, 1))

        type_probs_new = F.softmax(type_logits_new, dim=-1)
        bond_probs_new = F.softmax(bond_logits_new, dim=-1)
        type_step_probs_new = self._categorical_step_probs(
            state["atomics"],
            type_probs_new,
            times,
            step_size,
            cat_noise_level=self.default_cat_noise_level,
        )
        bond_step_probs_new = self._categorical_step_probs(
            state["bonds"],
            bond_probs_new,
            times,
            step_size,
            cat_noise_level=self.default_cat_noise_level,
        )

        coord_logprob = self._gaussian_logprob(
            transition["next_coords"],
            coord_mean,
            coord_std,
            transition["masks"],
            flag_3Ds=flag_3Ds,
        )
        atom_logprob = self._categorical_logprob(
            transition["next_atomics_idx"],
            type_step_probs_new,
            transition["masks"],
            pairwise=False,
        )
        bond_logprob = self._categorical_logprob(
            transition["next_bonds_idx"],
            bond_step_probs_new,
            transition["masks"],
            pairwise=True,
        )
        new_logprob = coord_logprob + atom_logprob + bond_logprob

        if (not self.use_reference_policy) or (self.ref_gen is None):
            step_kl = torch.zeros((), device=new_logprob.device, dtype=new_logprob.dtype)
        else:
            # Reference policy is only used for KL, so run it no-grad.
            with torch.no_grad():
                coords_ref, type_logits_ref, bond_logits_ref, _ = self._forward_with_model(
                    self.ref_gen,
                    state,
                    times,
                    cond_batch=cond_batch,
                    flag_3Ds=flag_3Ds,
                )

            # mean_ref = self._coord_mean_update(state["coords"], coords_ref, times, step_size[0].item())
            coord_velocity_ref = (coords_ref - state["coords"]) / (1 - times.view(-1, 1, 1)).clamp_min(self.eps)
            coord_mean_ref = state["coords"] + step_size.view(-1, 1, 1) * coord_velocity_ref
            type_probs_ref = F.softmax(type_logits_ref, dim=-1)
            bond_probs_ref = F.softmax(bond_logits_ref, dim=-1)
            type_step_probs_ref = self._categorical_step_probs(
                state["atomics"],
                type_probs_ref,
                times,
                step_size,
                cat_noise_level=self.default_cat_noise_level,
            )
            bond_step_probs_ref = self._categorical_step_probs(
                state["bonds"],
                bond_probs_ref,
                times,
                step_size,
                cat_noise_level=self.default_cat_noise_level,
            )

            coord_kl = self._coord_kl(coord_mean, coord_mean_ref, coord_std, transition["masks"], flag_3Ds=flag_3Ds)
            atom_kl = self._categorical_kl(type_step_probs_new, type_step_probs_ref, transition["masks"], pairwise=False)
            bond_kl = self._categorical_kl(bond_step_probs_new, bond_step_probs_ref, transition["masks"], pairwise=True)
            # Masked means prevent bond KL (O(N^2)) from dominating due to tensor size.
            step_kl = (coord_kl + atom_kl + bond_kl).mean()

        return new_logprob, step_kl

    def _save_mols_to_sdf(self, mols, path):
        writer = Chem.SDWriter(path)
        for mol in mols:
            if mol is not None:
                writer.write(mol)
        writer.close()
    
    def FM_training_step(self, batch):
        noise = {
            "coords": batch["noise_coords"],
            "atomics": batch["noise_atomics"],
            "bonds": batch["noise_bonds"],
            "masks": batch["masks"],
            "flag_3Ds": batch["flag_3Ds"],
        }

        # No old-policy deepcopy: old policy info is represented by cached old_logprob from rollout.
        old_training_state = self.gen.training
        self.gen.eval()
        with torch.no_grad():
            generated, transitions = self._collect_rollout(
                noise,
                self.gen,
                inference_steps=self.max_steps,
                cat_noise_level=self.default_cat_noise_level,
            )
        if old_training_state:
            self.gen.train()

        rewards, generated_mols = self._compute_rewards_from_generated(generated)
        advantages = self._group_advantages(rewards)

        quality_metrics = self._compute_generation_quality_from_mols(
            generated_mols,
            dtype=generated["coords"].dtype,
            device=generated["coords"].device,
        )
        if (
            self.val_save_path is not None
            and self.global_rank == 0
            and self.global_step % 100 == 0
        ):
            os.makedirs(self.val_save_path, exist_ok=True)
            save_file = os.path.join(
                self.val_save_path,
                f"generated_epoch_{self.current_epoch}.sdf"
            )
            self._save_mols_to_sdf(generated_mols, save_file)

        # Per-transition backward to avoid keeping one giant graph across all transitions.
        opt = self.optimizers()
        opt.zero_grad()

        n_steps = max(len(transitions), 1)
        clip_obj_sum = torch.zeros((), device=self.device)
        kl_sum = torch.zeros((), device=self.device)

        for transition in transitions:
            transition = self._transition_to_device(transition, self.device)
            old_logprob = transition["old_logprob"]

            new_logprob, step_kl = self._transition_step_stats(transition)

            ratio = torch.exp(new_logprob - old_logprob)
            ratio = torch.clamp(ratio, max=self.ratio_max)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            step_clip_obj = torch.minimum(surr1, surr2).mean()

            step_loss = (-step_clip_obj + (self.kl_beta * step_kl)) / n_steps
            self.manual_backward(step_loss)

            clip_obj_sum = clip_obj_sum + step_clip_obj.detach()
            kl_sum = kl_sum + step_kl.detach()

            del transition

        if self.grad_clip_val is not None and self.grad_clip_val > 0:
            self.clip_gradients(
                opt,
                gradient_clip_val=self.grad_clip_val,
                gradient_clip_algorithm="norm",
            )

        opt.step()
        opt.zero_grad()

        clip_obj = clip_obj_sum / n_steps
        kl_loss = kl_sum / n_steps
        total_loss = -clip_obj + (self.kl_beta * kl_loss)

        self.log("train-grpo-objective", clip_obj, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-grpo-kl", kl_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-grpo-loss", total_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-grpo-reward-mean", rewards.mean(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-grpo-reward-max", rewards.max(), on_step=True, logger=True, sync_dist=True)
        self.log("train-grpo-adv-mean", advantages.mean(), on_step=True, logger=True, sync_dist=True)
        self.log("train-grpo-adv-std", advantages.std(unbiased=False), on_step=True, logger=True, sync_dist=True)

        self.log("train-gen-validity", quality_metrics["validity"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-uniqueness", quality_metrics["uniqueness"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-connected-validity", quality_metrics["connected-validity"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-n-valid", quality_metrics["n-valid"], on_step=True, logger=True, sync_dist=True)
        self.log("train-gen-n-total", quality_metrics["n-total"], on_step=True, logger=True, sync_dist=True)

        return total_loss.detach()

    def validation_step(self, batch, b_idx):
        return

    def on_validation_epoch_end(self):
        return

    def test_step(self, batch, batch_idx):
        return

    def on_test_epoch_end(self):
        return
