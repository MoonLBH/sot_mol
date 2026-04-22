import torch
import torch.nn.functional as F

from .rl_diff import RL_Lightning
from ..util.functional import adj_from_node_mask


class LFPOF_Lightning(RL_Lightning):
    def __init__(
        self,
        *args,
        lfpo_num_time_samples: int = 4,
        lfpo_reward_temperature: float = 1.0,
        lfpo_beta_types: float = 1.0,
        lfpo_beta_bonds: float = 1.0,
        lfpo_beta_charges: float = 1.0,
        lfpo_beta_coord: float = 1.5,
        lfpo_gamma_coord: float = 1.0,
        lfpo_lambda_coord_rect: float = 1.0,
        lfpo_lambda_types_rect: float = 1.0,
        lfpo_lambda_bonds_rect: float = 1.0,
        lfpo_lambda_charges_rect: float = 1.0,
        lfpo_aux_fm_weight: float = 0.05,
        ref_ema_decay: float = 0.999,
        lfpo_use_charge_head: bool = True,
        lfpo_detach_targets: bool = True,
        lfpo_time_chunk_size: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lfpo_num_time_samples = max(1, int(lfpo_num_time_samples))
        self.lfpo_reward_temperature = max(float(lfpo_reward_temperature), 1e-6)

        self.lfpo_beta_types = float(lfpo_beta_types)
        self.lfpo_beta_bonds = float(lfpo_beta_bonds)
        self.lfpo_beta_charges = float(lfpo_beta_charges)
        self.lfpo_beta_coord = float(lfpo_beta_coord)
        self.lfpo_gamma_coord = float(lfpo_gamma_coord)

        self.lfpo_lambda_coord_rect = float(lfpo_lambda_coord_rect)
        self.lfpo_lambda_types_rect = float(lfpo_lambda_types_rect)
        self.lfpo_lambda_bonds_rect = float(lfpo_lambda_bonds_rect)
        self.lfpo_lambda_charges_rect = float(lfpo_lambda_charges_rect)

        self.lfpo_aux_fm_weight = float(lfpo_aux_fm_weight)
        self.ref_ema_decay = float(ref_ema_decay)
        self.lfpo_use_charge_head = bool(lfpo_use_charge_head)
        self.lfpo_detach_targets = bool(lfpo_detach_targets)
        self.lfpo_time_chunk_size = int(lfpo_time_chunk_size)

    def _reward_to_pull_weights(self, rewards):
        centered = rewards - rewards.mean()
        normed = centered / (centered.std(unbiased=False) + self.reward_norm_eps)
        return torch.sigmoid(normed / self.lfpo_reward_temperature)

    def _sample_stratified_timesteps(self, batch_size, K, device, dtype):
        edges = torch.arange(K, device=device, dtype=dtype) / float(K)
        u = torch.rand(batch_size, K, device=device, dtype=dtype) / float(K)
        return edges.unsqueeze(0) + u

    def _repeat_train_batch_for_time_samples(self, train_batch, repeats):
        expanded = {}
        for key, value in train_batch.items():
            if torch.is_tensor(value):
                expanded[key] = value.repeat_interleave(repeats, dim=0)
            else:
                expanded[key] = value
        return expanded

    def _soft_ce_node_per_sample(self, target_probs, logits, masks, eps=1e-3):
        logp = F.log_softmax(logits, dim=-1)
        ce = -(target_probs * logp).sum(dim=-1)
        n_atoms = masks.sum(dim=1).clamp_min(eps)
        return (ce * masks).sum(dim=1) / n_atoms

    def _soft_ce_edge_per_sample(self, target_probs, logits, masks, eps=1e-3):
        logp = F.log_softmax(logits, dim=-1)
        ce = -(target_probs * logp).sum(dim=-1)
        bond_mask = adj_from_node_mask(masks, self_connect=True).float()
        n_bonds = bond_mask.sum(dim=(1, 2)).clamp_min(eps)
        return (ce * bond_mask).sum(dim=(1, 2)) / n_bonds

    def _lfpof_discrete_loss_per_sample(self, cur_logits, ref_logits, masks, beta, is_edge=False):
        logp_cur = F.log_softmax(cur_logits, dim=-1)
        logp_ref = F.log_softmax(ref_logits, dim=-1)
        delta = (logp_cur - logp_ref).detach()

        logp_plus = logp_ref + beta * delta
        logp_minus = logp_ref - beta * delta

        p_plus = F.softmax(logp_plus, dim=-1)
        p_minus = F.softmax(logp_minus, dim=-1)
        if self.lfpo_detach_targets:
            p_plus = p_plus.detach()
            p_minus = p_minus.detach()

        if is_edge:
            ce_plus = self._soft_ce_edge_per_sample(p_plus, cur_logits, masks)
            ce_minus = self._soft_ce_edge_per_sample(p_minus, cur_logits, masks)
            delta_abs = delta.abs().mean(dim=-1)
            bond_mask = adj_from_node_mask(masks, self_connect=True).float()
            delta_abs = (delta_abs * bond_mask).sum(dim=(1, 2)) / bond_mask.sum(dim=(1, 2)).clamp_min(1.0)
        else:
            ce_plus = self._soft_ce_node_per_sample(p_plus, cur_logits, masks)
            ce_minus = self._soft_ce_node_per_sample(p_minus, cur_logits, masks)
            delta_abs = delta.abs().mean(dim=-1)
            delta_abs = (delta_abs * masks).sum(dim=1) / masks.sum(dim=1).clamp_min(1.0)

        return ce_plus, ce_minus, delta_abs

    def _lfpof_coord_loss_per_sample(self, pred_cur_r, pred_ref_r, masks, flag_3Ds):
        delta_r = (pred_cur_r - pred_ref_r).detach()
        target_plus_r = pred_ref_r + self.lfpo_beta_coord * delta_r
        target_minus_r = pred_ref_r - self.lfpo_gamma_coord * delta_r

        if self.lfpo_detach_targets:
            target_plus_r = target_plus_r.detach()
            target_minus_r = target_minus_r.detach()

        mse_plus = ((pred_cur_r - target_plus_r) ** 2).mean(dim=-1)
        mse_minus = ((pred_cur_r - target_minus_r) ** 2).mean(dim=-1)

        n_atoms = masks.sum(dim=1).clamp_min(1.0)
        mse_plus = (mse_plus * masks).sum(dim=1) / n_atoms
        mse_minus = (mse_minus * masks).sum(dim=1) / n_atoms

        mse_plus = mse_plus * flag_3Ds.view(-1)
        mse_minus = mse_minus * flag_3Ds.view(-1)
        delta_abs = (delta_r.abs().mean(dim=-1) * masks).sum(dim=1) / n_atoms
        return mse_plus, mse_minus, delta_abs

    def _compute_reference_predictions(self, interp_data, t, cond_batch, flag_3Ds):
        with torch.no_grad():
            ref_coords, ref_types, ref_bonds, ref_charges = self._forward_with_model(
                self.ref_gen,
                interp_data,
                t,
                cond_batch=cond_batch,
                flag_3Ds=flag_3Ds,
            )
        return {
            "coords": ref_coords,
            "atomics": ref_types,
            "bonds": ref_bonds,
            "charges": ref_charges,
        }

    def _maybe_update_reference_ema(self):
        if (not self.use_reference_anchor) or (self.ref_gen is None):
            return
        d = self.ref_ema_decay
        with torch.no_grad():
            for ref_param, cur_param in zip(self.ref_gen.parameters(), self.gen.parameters()):
                ref_param.data.mul_(d).add_(cur_param.data, alpha=(1.0 - d))
            for ref_buf, cur_buf in zip(self.ref_gen.buffers(), self.gen.buffers()):
                ref_buf.copy_(cur_buf)
        self.ref_gen.eval()
        for param in self.ref_gen.parameters():
            param.requires_grad = False

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
        pull_w = self._reward_to_pull_weights(rewards)

        train_batch = self._build_generated_target_batch(batch, generated)
        B = train_batch["natoms"].size(0)
        device = train_batch["real_coords"].device
        dtype = train_batch["real_coords"].dtype

        t_bk = self._sample_stratified_timesteps(B, self.lfpo_num_time_samples, device, dtype)
        t_flat = t_bk.reshape(-1)

        expanded_batch = self._repeat_train_batch_for_time_samples(train_batch, self.lfpo_num_time_samples)
        pull_w_flat = pull_w.repeat_interleave(self.lfpo_num_time_samples)

        total_n = t_flat.size(0)
        chunk_size = total_n if self.lfpo_time_chunk_size <= 0 else min(self.lfpo_time_chunk_size, total_n)

        sum_type_rect = torch.zeros((), device=device, dtype=dtype)
        sum_bond_rect = torch.zeros((), device=device, dtype=dtype)
        sum_charge_rect = torch.zeros((), device=device, dtype=dtype)
        sum_coord_rect = torch.zeros((), device=device, dtype=dtype)
        sum_aux_fm = torch.zeros((), device=device, dtype=dtype)
        sum_anchor = torch.zeros((), device=device, dtype=dtype)
        sum_delta_type = torch.zeros((), device=device, dtype=dtype)
        sum_delta_bond = torch.zeros((), device=device, dtype=dtype)
        sum_delta_charge = torch.zeros((), device=device, dtype=dtype)
        sum_delta_coord = torch.zeros((), device=device, dtype=dtype)
        has_charge_head = False

        for start in range(0, total_n, chunk_size):
            end = min(start + chunk_size, total_n)
            sl = slice(start, end)

            chunk_batch = {k: (v[sl] if torch.is_tensor(v) else v) for k, v in expanded_batch.items()}
            t_chunk = t_flat[sl]
            pull_chunk = pull_w_flat[sl]
            flag_3Ds = chunk_batch["flag_3Ds"]

            interp_data = self.interpolate(chunk_batch, t_chunk, flag_3Ds=flag_3Ds)

            cond_batch = None
            if self.self_cond:
                cond_batch = {
                    "coords": torch.zeros_like(interp_data["coords"]),
                    "atomics": torch.zeros_like(interp_data["atomics"]),
                    "bonds": torch.zeros_like(interp_data["bonds"]),
                }
                if torch.rand(1).item() > 0.5:
                    with torch.no_grad():
                        cond_coords, cond_types, cond_bonds, _ = self._forward_with_model(
                            self.gen,
                            interp_data,
                            t_chunk,
                            cond_batch=cond_batch,
                            flag_3Ds=flag_3Ds,
                        )
                    cond_batch = {
                        "coords": cond_coords * flag_3Ds.view(-1, 1, 1),
                        "atomics": F.softmax(cond_types, dim=-1),
                        "bonds": F.softmax(cond_bonds, dim=-1),
                    }

            coords, types, bonds, charges = self._forward_with_model(
                self.gen,
                interp_data,
                t_chunk,
                cond_batch=cond_batch,
                flag_3Ds=flag_3Ds,
            )
            predicted = {"coords": coords, "atomics": types, "bonds": bonds, "charges": charges}
            ref_predicted = self._compute_reference_predictions(interp_data, t_chunk, cond_batch, flag_3Ds)
            masks = interp_data["masks"].float()

            type_plus, type_minus, delta_type_abs = self._lfpof_discrete_loss_per_sample(
                predicted["atomics"], ref_predicted["atomics"], masks, self.lfpo_beta_types, is_edge=False
            )
            type_rect_ps = pull_chunk * type_plus + (1.0 - pull_chunk) * type_minus

            bond_plus, bond_minus, delta_bond_abs = self._lfpof_discrete_loss_per_sample(
                predicted["bonds"], ref_predicted["bonds"], masks, self.lfpo_beta_bonds, is_edge=True
            )
            bond_rect_ps = pull_chunk * bond_plus + (1.0 - pull_chunk) * bond_minus

            coord_plus, coord_minus, delta_coord_abs = self._lfpof_coord_loss_per_sample(
                predicted["coords"], ref_predicted["coords"], masks, flag_3Ds
            )
            coord_rect_ps = pull_chunk * coord_plus + (1.0 - pull_chunk) * coord_minus

            charge_rect_ps = torch.zeros_like(type_rect_ps)
            use_charge = (
                self.lfpo_use_charge_head
                and (predicted.get("charges") is not None)
                and (ref_predicted.get("charges") is not None)
            )
            if use_charge:
                has_charge_head = True
                charge_plus, charge_minus, delta_charge_ps = self._lfpof_discrete_loss_per_sample(
                    predicted["charges"], ref_predicted["charges"], masks, self.lfpo_beta_charges, is_edge=False
                )
                charge_rect_ps = pull_chunk * charge_plus + (1.0 - pull_chunk) * charge_minus
                sum_delta_charge = sum_delta_charge + delta_charge_ps.sum()

            if self.formulation == "endpoint":
                coords_target = chunk_batch["real_coords"]
            else:
                coords_target = chunk_batch["real_coords"] - chunk_batch["noise_coords"]
            aux_target = {
                "coords": coords_target,
                "atomics": chunk_batch["real_atomics"],
                "bonds": chunk_batch["real_bonds"],
                "charges": chunk_batch["real_charges"],
                "masks": chunk_batch["masks"],
            }
            aux_losses = self._loss_per_sample(aux_target, predicted, flag_3Ds=flag_3Ds)
            aux_ps = (
                aux_losses["coord-loss"]
                + aux_losses["type-loss"]
                + aux_losses["bond-loss"]
                + aux_losses["charge-loss"]
            )

            anchor_losses = self._anchor_loss_per_sample(
                interp_data,
                t_chunk,
                predicted,
                cond_batch=cond_batch,
                flag_3Ds=flag_3Ds,
            )
            anchor_ps = (
                anchor_losses["coord"]
                + anchor_losses["types"]
                + anchor_losses["bonds"]
                + anchor_losses["charges"]
            )

            sum_type_rect = sum_type_rect + type_rect_ps.sum()
            sum_bond_rect = sum_bond_rect + bond_rect_ps.sum()
            sum_coord_rect = sum_coord_rect + coord_rect_ps.sum()
            sum_charge_rect = sum_charge_rect + charge_rect_ps.sum()
            sum_aux_fm = sum_aux_fm + aux_ps.sum()
            sum_anchor = sum_anchor + anchor_ps.sum()
            sum_delta_type = sum_delta_type + delta_type_abs.sum()
            sum_delta_bond = sum_delta_bond + delta_bond_abs.sum()
            sum_delta_coord = sum_delta_coord + delta_coord_abs.sum()

        denom = float(total_n)
        type_rect_loss = sum_type_rect / denom
        bond_rect_loss = sum_bond_rect / denom
        coord_rect_loss = sum_coord_rect / denom
        charge_rect_loss = sum_charge_rect / denom
        aux_fm_loss = sum_aux_fm / denom
        anchor_loss = sum_anchor / denom
        delta_type_abs_mean = sum_delta_type / denom
        delta_bond_abs_mean = sum_delta_bond / denom
        delta_coord_abs_mean = sum_delta_coord / denom
        delta_charge_abs_mean = (sum_delta_charge / denom) if has_charge_head else torch.zeros_like(type_rect_loss)

        lfpof_main_loss = (
            self.lfpo_lambda_coord_rect * coord_rect_loss
            + self.lfpo_lambda_types_rect * type_rect_loss
            + self.lfpo_lambda_bonds_rect * bond_rect_loss
            + (self.lfpo_lambda_charges_rect * charge_rect_loss if has_charge_head else 0.0)
        )
        total_loss = lfpof_main_loss + self.lfpo_aux_fm_weight * aux_fm_loss + self.anchor_weight * anchor_loss

        self.log("train-lfpof-reward-mean", rewards.mean(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-lfpof-reward-max", rewards.max(), on_step=True, logger=True, sync_dist=True)
        self.log("train-lfpof-pull-weight-mean", pull_w.mean(), on_step=True, logger=True, sync_dist=True)
        self.log("train-lfpof-pull-weight-min", pull_w.min(), on_step=True, logger=True, sync_dist=True)
        self.log("train-lfpof-pull-weight-max", pull_w.max(), on_step=True, logger=True, sync_dist=True)

        self.log("train-lfpof-main-loss", lfpof_main_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-lfpof-coord-rect-loss", coord_rect_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-lfpof-type-rect-loss", type_rect_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-lfpof-bond-rect-loss", bond_rect_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-lfpof-aux-fm-loss", aux_fm_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-lfpof-anchor-loss", anchor_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-lfpof-total-loss", total_loss, on_step=True, logger=True, sync_dist=True)
        if has_charge_head:
            self.log("train-lfpof-charge-rect-loss", charge_rect_loss, on_step=True, logger=True, sync_dist=True)

        self.log("train-lfpof-delta-type-abs-mean", delta_type_abs_mean, on_step=True, logger=True, sync_dist=True)
        self.log("train-lfpof-delta-bond-abs-mean", delta_bond_abs_mean, on_step=True, logger=True, sync_dist=True)
        self.log("train-lfpof-delta-coord-abs-mean", delta_coord_abs_mean, on_step=True, logger=True, sync_dist=True)
        if has_charge_head:
            self.log("train-lfpof-delta-charge-abs-mean", delta_charge_abs_mean, on_step=True, logger=True, sync_dist=True)

        self.log("train-gen-validity", quality_metrics["validity"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-uniqueness", quality_metrics["uniqueness"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-connected-validity", quality_metrics["connected-validity"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-n-valid", quality_metrics["n-valid"], on_step=True, logger=True, sync_dist=True)
        self.log("train-gen-n-total", quality_metrics["n-total"], on_step=True, logger=True, sync_dist=True)
        for k in [
            "strain-energy-mean",
            "strain-energy-median",
            "strain-energy-per-atom-mean",
            "strain-energy-per-atom-median",
            "n-strain-success",
        ]:
            if k in quality_metrics:
                self.log(f"train-gen-{k}", quality_metrics[k], on_step=True, logger=True, sync_dist=True)

        return total_loss

    def on_train_batch_end(self, outputs, batch, b_idx):
        super().on_train_batch_end(outputs, batch, b_idx)
        self._maybe_update_reference_ema()
