import copy
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import QED
from ..comparm import GP
from .diff import SC_Lightning


class MoleculeTimestepSampler(torch.nn.Module):
    """Lightweight timestep sampler (masked-pooling features -> Beta(a,b)).

    This module takes per-sample molecular summary features built from current generated x0
    and predicts Beta parameters. We keep this intentionally simple and dependency-free.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, hidden_depth: int = 2, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(0, hidden_depth - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, features):
        h = self.backbone(features)
        raw = self.head(h)
        raw_a = raw[:, 0]
        raw_b = raw[:, 1]
        a = F.softplus(raw_a) + self.eps
        b = F.softplus(raw_b) + self.eps
        return a, b

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
        adaptive_timestep_enabled: bool = False,
        adaptive_timestep_mode: str = "probe",
        timestep_sampler_hidden_dim: int = 256,
        timestep_sampler_hidden_depth: int = 2,
        timestep_sampler_eps: float = 1e-4,
        timestep_sampler_lr: float = 1e-3,
        adaptive_timestep_update_freq: int = 40,
        adaptive_timestep_per_sample: bool = True,
        probe_num_grid: int = 16,
        probe_queue_size: int = 20,
        probe_topk: int = 3,
        probe_use_batch_mean_for_queue: bool = True,
        value_eval_num_reuse_current_batch: bool = True,
        value_baseline_momentum: float = 0.9,
        value_use_baseline: bool = False,
        adaptive_value_include_anchor: bool = False,
        adaptive_timestep_mix_base_prob: float = 0.0,
        timestep_sampler_feature_dim: Optional[int] = None,
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

        # Adaptive timestep sampler configs (default off, old behavior preserved).
        self.adaptive_timestep_enabled = adaptive_timestep_enabled
        self.adaptive_timestep_mode = adaptive_timestep_mode
        self.timestep_sampler_hidden_dim = timestep_sampler_hidden_dim
        self.timestep_sampler_hidden_depth = timestep_sampler_hidden_depth
        self.timestep_sampler_eps = timestep_sampler_eps
        self.timestep_sampler_lr = timestep_sampler_lr
        self.adaptive_timestep_update_freq = max(1, int(adaptive_timestep_update_freq))
        self.adaptive_timestep_per_sample = adaptive_timestep_per_sample
        self.probe_num_grid = max(2, int(probe_num_grid))
        self.probe_topk = max(1, int(probe_topk))
        self.probe_use_batch_mean_for_queue = probe_use_batch_mean_for_queue
        self.value_eval_num_reuse_current_batch = value_eval_num_reuse_current_batch
        self.value_baseline_momentum = value_baseline_momentum
        self.value_use_baseline = value_use_baseline
        self.adaptive_value_include_anchor = adaptive_value_include_anchor
        self.adaptive_timestep_mix_base_prob = float(adaptive_timestep_mix_base_prob)
        self.timestep_sampler_charge_dim = len(GP.IDX_CHARGE_MAP)
        self.timestep_sampler_feature_dim = (
            timestep_sampler_feature_dim
            if timestep_sampler_feature_dim is not None
            else (1 + 3 + 3 + int(vocab.size) + self.timestep_sampler_charge_dim)
        )

        self.timestep_sampler = MoleculeTimestepSampler(
            input_dim=self.timestep_sampler_feature_dim,
            hidden_dim=self.timestep_sampler_hidden_dim,
            hidden_depth=self.timestep_sampler_hidden_depth,
            eps=self.timestep_sampler_eps,
        )
        self.probe_queue = deque(maxlen=max(1, int(probe_queue_size)))
        self.value_baseline = None
        self._adaptive_warn_flag = False

        if self.adaptive_timestep_mode not in {"probe", "one_step_value"}:
            raise ValueError("adaptive_timestep_mode must be one of {'probe', 'one_step_value'}")

        # Two-optimizer update requires manual optimization.
        if self.adaptive_timestep_enabled:
            self.automatic_optimization = False

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

    def _build_timestep_sampler_features(self, generated, batch_or_train_batch):
        # generated_pre is exactly current x0 sample from p_theta(x); we use it as sampler input.
        coords = generated["coords"]
        atomics = generated["atomics"]
        charges = generated.get("charges")
        batch_size = coords.size(0)
        masks = batch_or_train_batch["masks"].float()
        if masks.dim() == 3:
            masks = masks.squeeze(-1)
        if masks.dim() != 2:
            masks = masks.reshape(batch_size, -1)
        mask3 = masks.unsqueeze(-1)
        denom = masks.sum(dim=1, keepdim=True).clamp_min(1.0)

        natoms = batch_or_train_batch["natoms"].float()
        natoms = natoms.reshape(batch_size, -1)[:, :1]
        natoms_norm = natoms / max(float(coords.size(1)), 1.0)

        coords_mean = (coords * mask3).sum(dim=1) / denom
        coords_centered = coords - coords_mean.unsqueeze(1)
        coords_var = ((coords_centered ** 2) * mask3).sum(dim=1) / denom
        coords_std = torch.sqrt(coords_var + self.reward_norm_eps)

        atomics_mean = (atomics * mask3).sum(dim=1) / denom
        feature_chunks = [natoms_norm, coords_mean, coords_std, atomics_mean]

        if charges is not None:
            charges_mean = (charges * mask3).sum(dim=1) / denom
            feature_chunks.append(charges_mean)
        else:
            feature_chunks.append(
                torch.zeros(
                    coords.size(0),
                    self.timestep_sampler_charge_dim,
                    device=coords.device,
                    dtype=coords.dtype,
                )
            )

        normalized_chunks = []
        for idx, feat in enumerate(feature_chunks):
            feat = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            elif feat.dim() > 2:
                feat = feat.reshape(feat.size(0), -1)
            assert feat.dim() == 2, f"timestep sampler feature[{idx}] must be 2D, got shape={tuple(feat.shape)}"
            assert feat.size(0) == batch_size, (
                f"timestep sampler feature[{idx}] batch mismatch: {feat.size(0)} vs {batch_size}"
            )
            normalized_chunks.append(feat)

        features = torch.cat(normalized_chunks, dim=-1)
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        assert features.dim() == 2, f"timestep sampler cat features must be 2D, got shape={tuple(features.shape)}"
        return features

    def _sample_adaptive_t(self, batchsize, generated, batch_or_train_batch):
        device = generated["coords"].device
        dtype = generated["coords"].dtype
        base_t = self.time_dist.sample((batchsize,)).to(device=device, dtype=dtype)
        if not self.adaptive_timestep_enabled:
            return base_t, None, {}

        feats = self._build_timestep_sampler_features(generated, batch_or_train_batch)
        a, b = self.timestep_sampler(feats)
        a = torch.clamp(a, min=self.timestep_sampler_eps)
        b = torch.clamp(b, min=self.timestep_sampler_eps)
        beta_dist = torch.distributions.Beta(a, b)
        learned_t = beta_dist.sample()
        learned_t = torch.clamp(learned_t, min=self.timestep_sampler_eps, max=1.0 - self.timestep_sampler_eps)
        log_prob = beta_dist.log_prob(learned_t)

        if self.adaptive_timestep_per_sample:
            t = learned_t
        else:
            t_scalar = learned_t.mean().expand(batchsize)
            t = torch.clamp(t_scalar, min=self.timestep_sampler_eps, max=1.0 - self.timestep_sampler_eps)
            log_prob = beta_dist.log_prob(t)

        if self.adaptive_timestep_mix_base_prob > 0.0:
            mix_mask = torch.rand(batchsize, device=device) < self.adaptive_timestep_mix_base_prob
            t = torch.where(mix_mask, base_t, t)
            log_prob = torch.where(mix_mask, torch.zeros_like(log_prob), log_prob)

        aux = {
            "a": a.detach(),
            "b": b.detach(),
            "t": t.detach(),
        }
        return t, log_prob, aux

    def _compute_reward_weighted_losses_at_t(
        self,
        model,
        train_batch,
        t,
        weights,
        use_self_cond=False,
        stochastic_self_cond=False,
        include_anchor=False,
    ):
        flag_3Ds = train_batch["flag_3Ds"]
        interp_data = self.interpolate(train_batch, t, flag_3Ds=flag_3Ds)

        cond_batch = None
        if use_self_cond:
            cond_batch = {
                "coords": torch.zeros_like(interp_data["coords"]),
                "atomics": torch.zeros_like(interp_data["atomics"]),
                "bonds": torch.zeros_like(interp_data["bonds"]),
            }
            if stochastic_self_cond and torch.rand(1).item() > 0.5:
                with torch.no_grad():
                    cond_coords, cond_types, cond_bonds, _ = self._forward_with_model(
                        model,
                        interp_data,
                        t,
                        cond_batch=cond_batch,
                        flag_3Ds=flag_3Ds,
                    )
                cond_batch = {
                    "coords": cond_coords * flag_3Ds.view(-1, 1, 1),
                    "atomics": F.softmax(cond_types, dim=-1),
                    "bonds": F.softmax(cond_bonds, dim=-1),
                }

        coords, types, bonds, charges = self._forward_with_model(
            model,
            interp_data,
            t,
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
        per_sample_loss = (
            losses["coord-loss"]
            + losses["type-loss"]
            + losses["bond-loss"]
            + losses["charge-loss"]
        )
        weighted_utility = -weights * per_sample_loss

        anchor_per_sample = torch.zeros_like(per_sample_loss)
        if include_anchor:
            anchor_losses = self._anchor_loss_per_sample(
                interp_data,
                t,
                predicted,
                cond_batch=cond_batch,
                flag_3Ds=flag_3Ds,
            )
            anchor_per_sample = (
                anchor_losses["coord"]
                + anchor_losses["types"]
                + anchor_losses["bonds"]
                + anchor_losses["charges"]
            )
            weighted_utility = weighted_utility - (self.anchor_weight * self.anchor_loss_weight) * anchor_per_sample

        return {
            "losses": losses,
            "per_sample_loss": per_sample_loss,
            "weighted_utility": weighted_utility,
            "anchor_per_sample": anchor_per_sample,
        }

    def _build_probe_grid(self, device, dtype):
        l = self.probe_num_grid
        return torch.linspace(
            1.0 / (l + 1.0),
            l / (l + 1.0),
            steps=l,
            device=device,
            dtype=dtype,
        )

    def _compute_probe_delta(self, train_batch, weights, probe_grid, include_anchor=False):
        # Algorithm 2A: probe mode compares pre/post weighted surrogate utilities at probe taus.
        pre_util_cols = []
        post_util_cols = []

        for tau in probe_grid:
            tau_vec = tau.expand(train_batch["natoms"].size(0))
            pre_out = self._compute_reward_weighted_losses_at_t(
                self._probe_gen_before_update,
                train_batch,
                tau_vec,
                weights,
                use_self_cond=False,
                stochastic_self_cond=False,
                include_anchor=include_anchor,
            )
            post_out = self._compute_reward_weighted_losses_at_t(
                self.gen,
                train_batch,
                tau_vec,
                weights,
                use_self_cond=False,
                stochastic_self_cond=False,
                include_anchor=include_anchor,
            )
            pre_util_cols.append(pre_out["weighted_utility"])
            post_util_cols.append(post_out["weighted_utility"])

        pre_probe_matrix = torch.stack(pre_util_cols, dim=1)
        post_probe_matrix = torch.stack(post_util_cols, dim=1)
        delta_probe_matrix = post_probe_matrix - pre_probe_matrix
        delta_probe_mean = delta_probe_matrix.mean(dim=0)
        return delta_probe_matrix, delta_probe_mean

    def _update_probe_queue_and_select_Sk(self, delta_probe_mean):
        queue_item = delta_probe_mean.detach().cpu()
        self.probe_queue.append(queue_item)
        if len(self.probe_queue) == 0:
            topk = min(self.probe_topk, self.probe_num_grid)
            return torch.arange(topk, device=delta_probe_mean.device, dtype=torch.long)

        hist = torch.stack(list(self.probe_queue), dim=0)  # [N_hist, L]
        score = hist.abs().mean(dim=0)
        topk = min(self.probe_topk, score.numel())
        if topk <= 0:
            return torch.arange(1, device=delta_probe_mean.device, dtype=torch.long) * 0
        selected = torch.topk(score, k=topk, largest=True).indices.to(delta_probe_mean.device)
        return selected

    def _compute_one_step_value_delta(self, noise, rewards_pre, dtype, device):
        # Algorithm 2B: one-step value mode uses paired pre/post reward on same noise batch.
        with torch.no_grad():
            generated_post = self._generate(
                noise,
                inference_steps=self.max_steps,
                coord_noise_std=self.default_coord_noise_std,
                cat_noise_level=self.default_cat_noise_level,
            )
        rewards_post, _ = self._compute_rewards_from_generated(generated_post)
        rewards_post = rewards_post.to(device=device, dtype=dtype)
        rewards_pre = rewards_pre.to(device=device, dtype=dtype)
        feedback = rewards_post - rewards_pre

        if self.value_use_baseline:
            post_mean = rewards_post.mean().detach()
            if self.value_baseline is None:
                self.value_baseline = post_mean
            else:
                self.value_baseline = (
                    self.value_baseline_momentum * self.value_baseline
                    + (1.0 - self.value_baseline_momentum) * post_mean
                )
            feedback = rewards_post - self.value_baseline

        return feedback, rewards_post

    def _run_standard_fm(self, batch):
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

    def _run_adaptive_timestep_fm(self, batch):
        # ===== Algorithm 1: Main adaptive-timestep training flow =====
        opt_gen, opt_sampler = self.optimizers()
        sch_gen, sch_sampler = self.lr_schedulers()
        opt_gen.zero_grad()

        noise = self._build_noise_batch(batch)
        with torch.no_grad():
            generated_pre = self._generate(
                noise,
                inference_steps=self.max_steps,
                coord_noise_std=self.default_coord_noise_std,
                cat_noise_level=self.default_cat_noise_level,
            )

        rewards_pre, generated_mols = self._compute_rewards_from_generated(generated_pre)
        quality_metrics = self._compute_generation_quality_from_mols(
            generated_mols,
            dtype=generated_pre["coords"].dtype,
            device=generated_pre["coords"].device,
        )
        weights = self._reward_to_weights(rewards_pre)
        weights = weights / weights.mean().clamp_min(self.reward_norm_eps)
        train_batch = self._build_generated_target_batch(batch, generated_pre)
        batchsize = train_batch["natoms"].size(0)
        device = train_batch["real_coords"].device
        dtype = train_batch["real_coords"].dtype

        t, log_prob, sampler_aux = self._sample_adaptive_t(batchsize, generated_pre, train_batch)
        t = t.to(device=device, dtype=dtype)
        if log_prob is not None:
            log_prob = log_prob.to(device=device, dtype=dtype)
        need_sampler_update = (
            log_prob is not None
            and (self.global_step % self.adaptive_timestep_update_freq == 0)
        )
        if need_sampler_update and self.adaptive_timestep_mode == "probe":
            self._probe_gen_before_update = copy.deepcopy(self.gen)
            self._probe_gen_before_update.eval()
            for param in self._probe_gen_before_update.parameters():
                param.requires_grad = False

        main_out = self._compute_reward_weighted_losses_at_t(
            self.gen,
            train_batch,
            t,
            weights,
            use_self_cond=self.self_cond,
            stochastic_self_cond=True,
            include_anchor=False,
        )
        losses = main_out["losses"]
        weighted_losses = {name: (val * weights).mean() for name, val in losses.items()}
        fm_loss = sum(list(weighted_losses.values()))

        anchor_loss = torch.zeros(1, device=device, dtype=dtype).squeeze(0)
        if self.anchor_weight > 0:
            anchor_out = self._compute_reward_weighted_losses_at_t(
                self.gen,
                train_batch,
                t,
                torch.ones_like(weights),
                use_self_cond=self.self_cond,
                stochastic_self_cond=True,
                include_anchor=True,
            )
            anchor_loss = anchor_out["anchor_per_sample"].mean()
        total_loss = fm_loss + (self.anchor_weight * self.anchor_loss_weight) * anchor_loss

        self.manual_backward(total_loss)
        opt_gen.step()
        if sch_gen is not None:
            sch_gen.step()

        # Sampler update safety fallback (generator update should never be blocked).
        sampler_loss = torch.zeros(1, device=device, dtype=dtype).squeeze(0)
        sampler_feedback = None

        if need_sampler_update:
            try:
                if self.adaptive_timestep_mode == "probe":
                    # ===== Algorithm 2A: probe mode =====
                    probe_grid = self._build_probe_grid(device=device, dtype=dtype)
                    delta_probe_matrix, delta_probe_mean = self._compute_probe_delta(
                        train_batch,
                        weights,
                        probe_grid,
                        include_anchor=self.adaptive_value_include_anchor,
                    )
                    selected = self._update_probe_queue_and_select_Sk(delta_probe_mean)
                    sampler_feedback = delta_probe_matrix[:, selected].mean(dim=1)
                    self.log("train-adapt-probe-feedback-mean", sampler_feedback.mean(), on_step=True, logger=True, sync_dist=True)
                    self.log("train-adapt-probe-feedback-std", sampler_feedback.std(unbiased=False), on_step=True, logger=True, sync_dist=True)
                    self.log("train-adapt-probe-num-S", torch.tensor(float(selected.numel()), device=device), on_step=True, logger=True, sync_dist=True)
                    self.log("train-adapt-probe-selected-indices", selected.float().mean(), on_step=True, logger=True, sync_dist=True)
                else:
                    # ===== Algorithm 2B: one-step value mode =====
                    feedback, rewards_post = self._compute_one_step_value_delta(
                        noise if self.value_eval_num_reuse_current_batch else self._build_noise_batch(batch),
                        rewards_pre,
                        dtype=dtype,
                        device=device,
                    )
                    sampler_feedback = feedback
                    self.log("train-adapt-value-feedback-mean", feedback.mean(), on_step=True, logger=True, sync_dist=True)
                    self.log("train-adapt-value-feedback-std", feedback.std(unbiased=False), on_step=True, logger=True, sync_dist=True)
                    self.log("train-adapt-reward-pre-mean", rewards_pre.mean(), on_step=True, logger=True, sync_dist=True)
                    self.log("train-adapt-reward-post-mean", rewards_post.mean(), on_step=True, logger=True, sync_dist=True)

                sampler_loss = -(sampler_feedback.detach() * log_prob).mean()
                if torch.isfinite(sampler_loss):
                    opt_sampler.zero_grad()
                    self.manual_backward(sampler_loss)
                    opt_sampler.step()
                    if sch_sampler is not None:
                        sch_sampler.step()
                else:
                    self.log("train-adapt-sampler-skip", torch.tensor(1.0, device=device), on_step=True, logger=True, sync_dist=True)
            except Exception:
                if not self._adaptive_warn_flag:
                    self.print("Warning: adaptive timestep sampler update failed once, skipping sampler step.")
                    self._adaptive_warn_flag = True
                self.log("train-adapt-sampler-skip", torch.tensor(1.0, device=device), on_step=True, logger=True, sync_dist=True)

        # Keep original logs and add adaptive logs.
        for name, loss_val in weighted_losses.items():
            self.log(f"train-fm-{name}", loss_val, on_step=True, logger=True, sync_dist=True)
            self.log(f"train-fm-raw-{name}", losses[name].mean(), on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-reward-mean", rewards_pre.mean(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-rl-reward-max", rewards_pre.max(), on_step=True, logger=True, sync_dist=True)
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

        if log_prob is not None:
            self.log("train-adapt-logprob-mean", log_prob.mean(), on_step=True, logger=True, sync_dist=True)
        if len(sampler_aux) > 0:
            self.log("train-adapt-t-mean", sampler_aux["t"].mean(), on_step=True, logger=True, sync_dist=True)
            self.log("train-adapt-a-mean", sampler_aux["a"].mean(), on_step=True, logger=True, sync_dist=True)
            self.log("train-adapt-b-mean", sampler_aux["b"].mean(), on_step=True, logger=True, sync_dist=True)
        self.log("train-adapt-sampler-loss", sampler_loss, on_step=True, logger=True, sync_dist=True)
        return total_loss.detach()

    def FM_training_step(self, batch):
        if not self.adaptive_timestep_enabled:
            return self._run_standard_fm(batch)
        return self._run_adaptive_timestep_fm(batch)

    def training_step(self, batch, b_idx):
        batch = self.flatten_batch(batch)
        fm_loss = self.FM_training_step(batch)
        self.log("train-fm-loss", fm_loss, prog_bar=True, on_step=True, logger=True)
        self.log("train-loss", fm_loss, prog_bar=True, on_step=True, logger=True)
        return fm_loss

    def configure_optimizers(self):
        if not self.adaptive_timestep_enabled:
            return super().configure_optimizers()

        gen_opt = torch.optim.Adam(
            self.gen.parameters(),
            lr=self.lr,
            amsgrad=True,
            foreach=True,
            weight_decay=0.0,
        )
        sampler_opt = torch.optim.AdamW(
            self.timestep_sampler.parameters(),
            lr=self.timestep_sampler_lr,
            weight_decay=0.0,
        )

        warm_up_steps = 0 if self.warm_up_steps is None else self.warm_up_steps
        gen_scheduler = torch.optim.lr_scheduler.LinearLR(
            gen_opt,
            start_factor=1e-2,
            total_iters=warm_up_steps,
        )
        sampler_scheduler = torch.optim.lr_scheduler.LinearLR(
            sampler_opt,
            start_factor=1.0,
            total_iters=max(1, warm_up_steps),
        )

        return [gen_opt, sampler_opt], [
            {
                "scheduler": gen_scheduler,
                "interval": "step",
            },
            {
                "scheduler": sampler_scheduler,
                "interval": "step",
            },
        ]

    def validation_step(self, batch, b_idx):
        return

    def on_validation_epoch_end(self):
        return

    def test_step(self, batch, batch_idx):
        return

    def on_test_epoch_end(self):
        return
    
    
