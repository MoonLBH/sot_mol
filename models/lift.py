import copy
import math
from typing import Optional

import torch
import torch.nn.functional as F

from .rl_diff import RL_Lightning
from ..util.functional import adj_from_node_mask
from ..rl_objectives.mpo_tasks import build_objective
from ..rl_objectives.partition import build_partition_selector
from ..rl_objectives.mpo_tasks import ScoringResult
from ..rl_objectives.oracle_metrics import OracleLogger


class LIFT_Lightning(RL_Lightning):
    """
    LFPO-F: an LFPO-inspired post-training method for hybrid flow-matching molecular generation.

    Core design:
      - Discrete heads (atomics / bonds / charges): positive-negative implicit target rectification
        with soft cross-entropy.
      - Continuous head (coords): positive-negative target rectification in prediction space
        with masked MSE.
      - Reference model is updated by EMA.
      - Manual optimization is used so each chunk can backward immediately and release its graph,
        avoiding the "accumulate all chunk graphs then backward once" memory pattern.

    Notes:
      - This implementation intentionally samples trajectories from ref_gen, not ema_gen.
      - By default, the base EMA model is disabled to reduce GPU memory pressure during LFPO-F training.
      - If regularization_type == "Parametric_L2", a frozen parameter snapshot is stored; otherwise it is not.
    """

    def __init__(
        self,
        *args,
        lfpo_num_time_samples: int = 2,
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
        lfpo_disable_base_ema: bool = True,
        anchor_weight: float = 0.1,
        lfpo_eval_current_every: int = 100,
        lfpo_log_current_reward: bool = True,
        lfpo_eval_current_samples: int = 1000,
        lfpo_eval_current_batch_size: int = 256,
        # LFPO-F-v2: top reward imitation + bottom reward repulsion
        lfpo_top_ratio: float = 0.25,
        lfpo_bottom_ratio: float = 0.25,
        lfpo_bottom_repulsion_weight: float = 1.0,
        lfpo_middle_weight: float = 0.0,
        lfpo_use_top_bottom: bool = True,
        lfpo_top_weight_mode: str = "uniform",
        objective_name: str = "qed",
        objective_config: dict | None = None,
        partition_config: dict | None = None,
        metric_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Use manual optimization so each chunk can backward independently.
        self.automatic_optimization = False

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
        self.lfpo_disable_base_ema = bool(lfpo_disable_base_ema)
        self.anchor_weight = anchor_weight
        self.lfpo_eval_current_every = int(lfpo_eval_current_every)
        self.lfpo_log_current_reward = bool(lfpo_log_current_reward)
        self.lfpo_eval_current_samples = int(lfpo_eval_current_samples)
        self.lfpo_eval_current_batch_size = int(lfpo_eval_current_batch_size)
        self.lfpo_top_ratio = float(lfpo_top_ratio)
        self.lfpo_bottom_ratio = float(lfpo_bottom_ratio)
        self.lfpo_bottom_repulsion_weight = float(lfpo_bottom_repulsion_weight)
        self.lfpo_middle_weight = float(lfpo_middle_weight)
        self.lfpo_use_top_bottom = bool(lfpo_use_top_bottom)
        self.lfpo_top_weight_mode = str(lfpo_top_weight_mode).lower()
        if objective_name is None:
            objective_name = kwargs.get("reward_name", "qed")
        self.objective_name = objective_name
        self.objective_config = objective_config or {}
        self.partition_config = partition_config or {"mode":"scalar_top_bottom", "top_ratio": self.lfpo_top_ratio, "bottom_ratio": self.lfpo_bottom_ratio}
        self.metric_config = metric_config or {}
        self.log_debug_metrics = bool(self.metric_config.get("log_debug_metrics", False))
        self.objective = build_objective(self.objective_name, self.objective_config)
        self.partition_selector = build_partition_selector(self.partition_config)
        self.oracle_logger = OracleLogger(self.metric_config.get("oracle_log_path"), self.objective_name, enabled=self.metric_config.get("enabled", False), novelty_reference_path=self.metric_config.get("novelty_reference_path")) if self.metric_config.get("enabled", False) else None

        if self.lfpo_disable_base_ema:
            # Disable the extra averaged model to save GPU memory.
            self.ema_gen = None

    def _log_main(self, name, value, **kwargs):
        self.log(name, value, **kwargs)

    def _log_debug(self, name, value, **kwargs):
        if self.log_debug_metrics:
            self.log(name, value, **kwargs)

    def _merge_scoring_results(self, results: list[ScoringResult]) -> ScoringResult:
        if len(results) == 1:
            return results[0]
        first = results[0]
        return ScoringResult(
            score=torch.cat([r.score for r in results], dim=0),
            component_scores={k: torch.cat([r.component_scores[k] for r in results], dim=0) for k in first.component_scores},
            raw_properties={k: torch.cat([r.raw_properties[k] for r in results], dim=0) for k in first.raw_properties},
            feasible=torch.cat([r.feasible for r in results], dim=0),
            severe_violation=torch.cat([r.severe_violation for r in results], dim=0),
            valid=torch.cat([r.valid for r in results], dim=0),
            connected=torch.cat([r.connected for r in results], dim=0),
            smiles=[s for r in results for s in r.smiles],
            canonical_smiles=[s for r in results for s in r.canonical_smiles],
            scaffolds=[s for r in results for s in r.scaffolds],
            fps=[s for r in results for s in r.fps],
            mols=[s for r in results for s in r.mols],
            metadata={
                k: (
                    torch.cat([r.metadata[k] for r in results], dim=0)
                    if torch.is_tensor(first.metadata.get(k))
                    else first.metadata.get(k)
                )
                for k in first.metadata
            },
        )

    # ---------------------------------------------------------------------
    # initialization / hooks
    # ---------------------------------------------------------------------
    def on_fit_start(self):
        super(RL_Lightning, self).on_fit_start()  # call LightningModule/SC_Lightning chain, skip RL_Lightning copy behavior

        # LFPO-F always needs a reference model for main rectification, not just anchor.
        if self.ref_gen is None:
            self.ref_gen = copy.deepcopy(self.gen)
            self.ref_gen.eval()
            for p in self.ref_gen.parameters():
                p.requires_grad = False

        # Prior-state snapshot is only needed for parametric L2 regularization.
        self.prior_state = None
        if self.regularization_type == "Parametric_L2":
            self.prior_state = {name: p.detach().clone() for name, p in self.gen.named_parameters()}

    def on_train_batch_end(self, outputs, batch, b_idx):
        # Keep optional averaged model update only if user explicitly keeps ema_gen.
        if self.ema_gen is not None:
            self.ema_gen.update_parameters(self.gen)
        self._maybe_update_reference_ema()
    
    def _top_frac_mean(self, values: torch.Tensor, frac: float = 0.1) -> torch.Tensor:
        if values.numel() == 0:
            return torch.zeros((), device=values.device, dtype=values.dtype)
        k = max(1, int(values.numel() * frac))
        return torch.topk(values, k=k).values.mean()

    def _sem(self, values: torch.Tensor) -> torch.Tensor:
        """Standard error of the mean for scalar reward vectors."""
        if values.numel() <= 1:
            return torch.zeros((), device=values.device, dtype=values.dtype)
        return values.std(unbiased=True) / torch.sqrt(
            torch.tensor(float(values.numel()), device=values.device, dtype=values.dtype)
        )


    def _log_masked_mean(self, name, tensor, mask):
        if mask is None or int(mask.sum())==0:
            return
        self.log(name, tensor[mask].mean(), on_step=True, logger=True, sync_dist=True)

    def _safe_corr(self, a, b):
        a=a.float(); b=b.float()
        if a.numel()<2: return torch.zeros((),device=a.device)
        av=a-a.mean(); bv=b-b.mean(); den=(av.std(unbiased=False)*bv.std(unbiased=False)).clamp_min(1e-8)
        return (av*bv).mean()/den

    def _sample_noise_chunk_from_base(self, noise: dict, chunk_size: int) -> dict:
        """Sample a chunk of initial noise/templates from the current training batch.

        The training batch already contains valid node masks, categorical noise, and
        coordinate noise. For low-frequency current-policy evaluation, we resample
        rows from this base batch with replacement to obtain an arbitrary number of
        evaluation molecules without requiring a new dataloader. If the sampler uses
        stochastic integration noise, repeated templates can still yield distinct
        molecules.
        """
        base_n = noise["coords"].size(0)
        idx = torch.randint(base_n, (chunk_size,), device=noise["coords"].device)
        out = {}
        for key, value in noise.items():
            if torch.is_tensor(value) and value.size(0) == base_n:
                out[key] = value.index_select(0, idx)
            else:
                out[key] = value
        return out

    def _evaluate_model_scoring(
        self,
        model: torch.nn.Module,
        base_noise: dict,
        n_samples: int,
        eval_batch_size: int,
    ):
        """Generate n_samples from a chosen model and compute reward/quality metrics.

        This is used for low-frequency current-policy evaluation. It runs generation
        in chunks to avoid OOM when n_samples=1000 or larger.
        """
        n_samples = max(1, int(n_samples))
        eval_batch_size = max(1, int(eval_batch_size))
        scoring_all = []
        mols_all = []

        for start in range(0, n_samples, eval_batch_size):
            chunk_n = min(eval_batch_size, n_samples - start)
            noise_chunk = self._sample_noise_chunk_from_base(base_noise, chunk_n)
            with torch.no_grad():
                generated = self._generate_with_model(
                    model,
                    noise_chunk,
                    inference_steps=self.max_steps,
                    coord_noise_std=self.default_coord_noise_std,
                    cat_noise_level=self.default_cat_noise_level,
                )
            mols_chunk = self._generate_mols(generated, sanitise=True)
            scoring_chunk = self.objective.score_mols(mols_chunk, device=generated["coords"].device, dtype=generated["coords"].dtype)
            scoring_all.append(scoring_chunk)
            mols_all.extend(mols_chunk)

            del noise_chunk, generated

        scoring_cur = self._merge_scoring_results(scoring_all)
        quality_metrics = self._compute_generation_quality_from_mols(
            mols_all,
            dtype=scoring_cur.score.dtype,
            device=scoring_cur.score.device,
        )
        return scoring_cur, mols_all, quality_metrics

    # ---------------------------------------------------------------------
    # reward / time helpers
    # ---------------------------------------------------------------------
    def _reward_to_pull_weights(self, rewards: torch.Tensor) -> torch.Tensor:
        centered = rewards - rewards.mean()
        normed = centered / (rewards.std(unbiased=False) + self.reward_norm_eps)
        return torch.sigmoid(normed / self.lfpo_reward_temperature)

    def _reward_to_top_bottom_masks(self, rewards: torch.Tensor):
        """Return boolean masks for top-reward imitation and bottom-reward repulsion.

        Top samples are used for explicit imitation of the generated pseudo-targets.
        Bottom samples are used for LFPO-style negative implicit repulsion.
        Middle samples do not contribute to the main LFPO-F-v2 loss when
        lfpo_middle_weight == 0.
        """
        n = int(rewards.numel())
        if n == 0:
            empty = torch.zeros_like(rewards, dtype=torch.bool)
            return empty, empty, empty.float()

        if not self.lfpo_use_top_bottom:
            # Compatibility fallback: all samples participate as selected.
            selected = torch.ones_like(rewards, dtype=torch.bool)
            return selected, torch.zeros_like(selected), selected.float()

        n_top = int(math.ceil(n * max(0.0, self.lfpo_top_ratio)))
        n_bottom = int(math.ceil(n * max(0.0, self.lfpo_bottom_ratio)))

        n_top = min(n, max(1, n_top)) if self.lfpo_top_ratio > 0 else 0
        n_bottom = min(n, max(1, n_bottom)) if self.lfpo_bottom_ratio > 0 else 0

        top_mask = torch.zeros_like(rewards, dtype=torch.bool)
        bottom_mask = torch.zeros_like(rewards, dtype=torch.bool)

        if n_top > 0:
            top_idx = torch.topk(rewards, k=n_top, largest=True).indices
            top_mask[top_idx] = True

        if n_bottom > 0:
            bottom_idx = torch.topk(rewards, k=n_bottom, largest=False).indices
            bottom_mask[bottom_idx] = True
            bottom_mask = bottom_mask & (~top_mask)

        selected_mask = (top_mask | bottom_mask).float()
        return top_mask, bottom_mask, selected_mask

    def _sample_stratified_timesteps(self, batch_size: int, K: int, device, dtype) -> torch.Tensor:
        # shape [B, K], kth sample lies in [k/K, (k+1)/K)
        left = torch.arange(K, device=device, dtype=dtype) / float(K)
        u = torch.rand(batch_size, K, device=device, dtype=dtype) / float(K)
        return left.unsqueeze(0) + u

    def _slice_batch(self, batch: dict, sl: slice) -> dict:
        out = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                out[key] = value[sl]
            else:
                out[key] = value
        return out

    # ---------------------------------------------------------------------
    # generation with explicit model choice (avoid ema_gen ambiguity)
    # ---------------------------------------------------------------------
    def _generate_with_model(
        self,
        model: torch.nn.Module,
        noise: dict,
        inference_steps: int = 100,
        coord_noise_std: float = 0.0,
        cat_noise_level: float = 1.0,
        coms=None,
    ):
        """Generate samples with an explicitly selected model.

        This avoids SC_Lightning.forward(training=False) silently switching to ema_gen.
        The selected model is temporarily set to eval mode, then restored to its previous
        training/eval state. This is important for current-model reward evaluation.
        """
        was_training = model.training
        model.eval()
        try:
            time_points = torch.linspace(0.0, 1.0, steps=inference_steps + 1, device=noise["coords"].device)
            step_sizes = time_points[1:] - time_points[:-1]

            times = torch.zeros(noise["coords"].size(0), device=noise["coords"].device, dtype=noise["coords"].dtype)
            curr = {k: v.clone() for k, v in noise.items()}
            flag_3Ds = noise["flag_3Ds"]

            cond_batch = {
                "coords": torch.zeros_like(noise["coords"]),
                "atomics": torch.zeros_like(noise["atomics"]),
                "bonds": torch.zeros_like(noise["bonds"]),
            }

            with torch.no_grad():
                predicted = None
                for step_size in step_sizes.tolist():
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

                    cond_batch = {
                        "coords": coords * flag_3Ds.view(-1, 1, 1),
                        "atomics": type_probs,
                        "bonds": bond_probs,
                    }
                    predicted = {
                        "coords": coords * flag_3Ds.view(-1, 1, 1),
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

            if predicted is None:
                raise RuntimeError("Generation failed: no denoising step was executed.")

            if self.formulation == "endpoint":
                predicted["coords"] = predicted["coords"] * self.coord_scale
            else:
                predicted["coords"] = curr["coords"] * self.coord_scale

            if coms is not None:
                if coms.dim() == 2:
                    coms = coms.unsqueeze(1)
                elif coms.dim() == 3 and coms.size(1) != 1:
                    coms = coms[:, :1, :]
                predicted["coords"] = predicted["coords"] + coms

            return predicted
        finally:
            model.train(was_training)

    # ---------------------------------------------------------------------
    # soft losses / rectification helpers
    # ---------------------------------------------------------------------
    def _soft_ce_node_per_sample(self, target_probs, logits, masks, eps: float = 1e-3):
        logp = F.log_softmax(logits, dim=-1)
        ce = -(target_probs * logp).sum(dim=-1)
        n_atoms = masks.sum(dim=1).clamp_min(eps)
        return (ce * masks).sum(dim=1) / n_atoms

    def _soft_ce_edge_per_sample(self, target_probs, logits, masks, eps: float = 1e-3):
        logp = F.log_softmax(logits, dim=-1)
        ce = -(target_probs * logp).sum(dim=-1)
        bond_mask = adj_from_node_mask(masks, self_connect=True).float()
        n_bonds = bond_mask.sum(dim=(1, 2)).clamp_min(eps)
        return (ce * bond_mask).sum(dim=(1, 2)) / n_bonds

    def _lfpof_discrete_loss_per_sample(self, cur_logits, ref_logits, masks, beta: float, is_edge: bool = False):
        logp_cur = F.log_softmax(cur_logits, dim=-1)
        logp_ref = F.log_softmax(ref_logits, dim=-1)

        delta = logp_cur - logp_ref
        if self.lfpo_detach_targets:
            delta = delta.detach()

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
        delta_r = pred_cur_r - pred_ref_r
        if self.lfpo_detach_targets:
            delta_r = delta_r.detach()

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

    def _anchor_from_reference_per_sample(self, predicted, ref_predicted, masks, flag_3Ds):
        if (not self.use_reference_anchor) or (self.anchor_weight <= 0) or (self.ref_gen is None):
            zeros = torch.zeros(predicted["coords"].size(0), device=predicted["coords"].device, dtype=predicted["coords"].dtype)
            return {"coord": zeros, "types": zeros, "bonds": zeros, "charges": zeros}

        mask3 = masks.unsqueeze(-1)
        n_atoms = masks.sum(dim=1).clamp_min(1.0)

        coord = F.mse_loss(predicted["coords"], ref_predicted["coords"], reduction="none")
        coord = (coord * mask3).sum(dim=(1, 2)) / n_atoms
        coord = coord * flag_3Ds.view(-1)

        types = F.kl_div(
            F.log_softmax(predicted["atomics"], dim=-1),
            F.softmax(ref_predicted["atomics"], dim=-1),
            reduction="none",
        ).sum(dim=-1)
        types = (types * masks).sum(dim=1) / n_atoms

        bond_mask = adj_from_node_mask(masks, self_connect=True).float()
        n_bonds = bond_mask.sum(dim=(1, 2)).clamp_min(1.0)
        bonds = F.kl_div(
            F.log_softmax(predicted["bonds"], dim=-1),
            F.softmax(ref_predicted["bonds"], dim=-1),
            reduction="none",
        ).sum(dim=-1)
        bonds = (bonds * bond_mask).sum(dim=(1, 2)) / n_bonds

        charges = torch.zeros_like(types)
        if (
            predicted.get("charges") is not None
            and ref_predicted.get("charges") is not None
            and self.lfpo_use_charge_head
        ):
            charges = F.kl_div(
                F.log_softmax(predicted["charges"], dim=-1),
                F.softmax(ref_predicted["charges"], dim=-1),
                reduction="none",
            ).sum(dim=-1)
            charges = (charges * masks).sum(dim=1) / n_atoms

        return {"coord": coord, "types": types, "bonds": bonds, "charges": charges}

    def _maybe_update_reference_ema(self):
        if self.ref_gen is None:
            return
        d = self.ref_ema_decay
        with torch.no_grad():
            for ref_param, cur_param in zip(self.ref_gen.parameters(), self.gen.parameters()):
                ref_param.data.mul_(d).add_(cur_param.data, alpha=(1.0 - d))
            for ref_buf, cur_buf in zip(self.ref_gen.buffers(), self.gen.buffers()):
                ref_buf.copy_(cur_buf)
        self.ref_gen.eval()
        for p in self.ref_gen.parameters():
            p.requires_grad = False

    # ---------------------------------------------------------------------
    # main LFPO-F training logic
    # ---------------------------------------------------------------------
    def FM_training_step(self, batch):
        device = batch["real_coords"].device
        dtype = batch["real_coords"].dtype

        # ------------------------------------------------------------------
        # Phase 1: collect data with old/reference policy, as in LFPO/NFT.
        # ------------------------------------------------------------------
        noise = self._build_noise_batch(batch)
        with torch.no_grad():
            generated_ref = self._generate_with_model(
                self.ref_gen,
                noise,
                inference_steps=self.max_steps,
                coord_noise_std=self.default_coord_noise_std,
                cat_noise_level=self.default_cat_noise_level,
            )

        generated_mols_ref = self._generate_mols(generated_ref, sanitise=True)
        scoring_ref = self.objective.score_mols(generated_mols_ref, device=generated_ref["coords"].device, dtype=generated_ref["coords"].dtype)
        quality_metrics_ref = self._compute_generation_quality_from_mols(generated_mols_ref, dtype=generated_ref["coords"].dtype, device=generated_ref["coords"].device)

        pull_w = self._reward_to_pull_weights(scoring_ref.score)
        partition_ref = self.partition_selector.select(scoring_ref)
        rewards_ref = scoring_ref.score
        top_mask = partition_ref.top_mask
        bottom_mask = partition_ref.bottom_mask
        selected_mask = partition_ref.selected_mask
        
        # Weight for top-reward imitation.
        # Default: all selected top samples have weight 1.
        # Optional: reward-normalized top weighting, with mean weight = 1 within top set.
        top_imitation_w = partition_ref.top_weights if partition_ref is not None else torch.ones_like(rewards_ref)

        if self.lfpo_top_weight_mode == "rwr":
            if top_mask.any():
                top_mean = pull_w[top_mask].mean().clamp_min(self.reward_norm_eps)
                top_imitation_w = pull_w / top_mean
            else:
                top_imitation_w = partition_ref.top_weights if partition_ref is not None else torch.ones_like(rewards_ref)

        train_batch = self._build_generated_target_batch(batch, generated_ref)

        # ------------------------------------------------------------------
        # Optional low-frequency evaluation of current policy.
        # ------------------------------------------------------------------
        scoring_cur = None
        quality_metrics_cur = None
        cur_top_mask = None
        cur_bottom_mask = None
        if self.lfpo_log_current_reward and (
            self.global_step % max(1, self.lfpo_eval_current_every) == 0
        ):
            scoring_cur, generated_mols_cur, quality_metrics_cur = self._evaluate_model_scoring(
                model=self.gen,
                base_noise=noise,
                n_samples=self.lfpo_eval_current_samples,
                eval_batch_size=self.lfpo_eval_current_batch_size,
            )
            cur_top_mask, cur_bottom_mask, _ = self._reward_to_top_bottom_masks(scoring_cur.score)

        batch_size = train_batch["natoms"].size(0)
        t_bk = self._sample_stratified_timesteps(
            batch_size=batch_size,
            K=self.lfpo_num_time_samples,
            device=device,
            dtype=dtype,
        )

        opt = self.optimizers()
        opt.zero_grad()

        # Logging accumulators. Values are detached and represent true global
        # averages over selected samples for the main loss, and over all samples
        # for auxiliary/anchor losses.
        lfpof_main_loss = torch.zeros((), device=device, dtype=dtype)
        pos_imitation_loss = torch.zeros((), device=device, dtype=dtype)
        neg_repulsion_loss = torch.zeros((), device=device, dtype=dtype)
        type_rect_loss = torch.zeros((), device=device, dtype=dtype)
        bond_rect_loss = torch.zeros((), device=device, dtype=dtype)
        coord_rect_loss = torch.zeros((), device=device, dtype=dtype)
        charge_rect_loss = torch.zeros((), device=device, dtype=dtype)
        aux_fm_loss = torch.zeros((), device=device, dtype=dtype)
        anchor_loss = torch.zeros((), device=device, dtype=dtype)
        total_loss_log = torch.zeros((), device=device, dtype=dtype)
        delta_type_abs_mean = torch.zeros((), device=device, dtype=dtype)
        delta_bond_abs_mean = torch.zeros((), device=device, dtype=dtype)
        delta_coord_abs_mean = torch.zeros((), device=device, dtype=dtype)
        delta_charge_abs_mean = torch.zeros((), device=device, dtype=dtype)
        has_charge_head = False

        batch_chunk = batch_size if self.lfpo_time_chunk_size <= 0 else min(self.lfpo_time_chunk_size, batch_size)
        total_n = float(batch_size * self.lfpo_num_time_samples)
        selected_total = (selected_mask.sum() * self.lfpo_num_time_samples).clamp_min(1.0)

        for k in range(self.lfpo_num_time_samples):
            t_k = t_bk[:, k]
            for start in range(0, batch_size, batch_chunk):
                end = min(start + batch_chunk, batch_size)
                sl = slice(start, end)

                chunk_batch = self._slice_batch(train_batch, sl)
                t_chunk = t_k[sl]
                flag_3Ds = chunk_batch["flag_3Ds"]
                top_chunk = top_mask[sl].float()
                bottom_chunk = bottom_mask[sl].float()
                middle_chunk = (1.0 - (top_chunk + bottom_chunk).clamp(max=1.0))
                
                top_w_chunk = top_imitation_w[sl].float()

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
                predicted = {
                    "coords": coords,
                    "atomics": types,
                    "bonds": bonds,
                    "charges": charges,
                }
                ref_predicted = self._compute_reference_predictions(interp_data, t_chunk, cond_batch, flag_3Ds)
                masks = interp_data["masks"].float()

                # ----------------------------------------------------------
                # Positive branch: explicit imitation of high-reward samples.
                # This fixes the beta=1 self-target issue in the previous
                # implementation, where CE(p_cur.detach(), p_cur) gives nearly
                # zero gradient for top samples.
                # ----------------------------------------------------------
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

                pos_type_ps = aux_losses["type-loss"]
                pos_bond_ps = aux_losses["bond-loss"]
                pos_charge_ps = aux_losses["charge-loss"]
                pos_coord_ps = aux_losses["coord-loss"]

                # ----------------------------------------------------------
                # Negative branch: LFPO-style implicit repulsion from bottom
                # samples. We use only CE_minus / MSE_minus for bottom samples.
                # ----------------------------------------------------------
                type_plus, type_minus, delta_type_abs = self._lfpof_discrete_loss_per_sample(
                    predicted["atomics"], ref_predicted["atomics"], masks, self.lfpo_beta_types, is_edge=False
                )
                bond_plus, bond_minus, delta_bond_abs = self._lfpof_discrete_loss_per_sample(
                    predicted["bonds"], ref_predicted["bonds"], masks, self.lfpo_beta_bonds, is_edge=True
                )
                coord_plus, coord_minus, delta_coord_abs = self._lfpof_coord_loss_per_sample(
                    predicted["coords"], ref_predicted["coords"], masks, flag_3Ds
                )

                charge_minus = torch.zeros_like(type_minus)
                charge_plus = torch.zeros_like(type_plus)
                delta_charge_abs = torch.zeros_like(type_minus)
                if (
                    self.lfpo_use_charge_head
                    and predicted.get("charges") is not None
                    and ref_predicted.get("charges") is not None
                ):
                    has_charge_head = True
                    charge_plus, charge_minus, delta_charge_abs = self._lfpof_discrete_loss_per_sample(
                        predicted["charges"], ref_predicted["charges"], masks, self.lfpo_beta_charges, is_edge=False
                    )

                # Optional middle fallback: by default zero, so middle samples do
                # not participate in the main loss.
                mid_type_ps = self.lfpo_middle_weight * (pull_w[sl] * type_plus + (1.0 - pull_w[sl]) * type_minus)
                mid_bond_ps = self.lfpo_middle_weight * (pull_w[sl] * bond_plus + (1.0 - pull_w[sl]) * bond_minus)
                mid_coord_ps = self.lfpo_middle_weight * (pull_w[sl] * coord_plus + (1.0 - pull_w[sl]) * coord_minus)
                mid_charge_ps = self.lfpo_middle_weight * (pull_w[sl] * charge_plus + (1.0 - pull_w[sl]) * charge_minus)

                type_rect_ps = (
                    top_chunk * top_w_chunk * pos_type_ps
                    + self.lfpo_bottom_repulsion_weight * bottom_chunk * type_minus
                    + middle_chunk * mid_type_ps
                )

                bond_rect_ps = (
                    top_chunk * top_w_chunk * pos_bond_ps
                    + self.lfpo_bottom_repulsion_weight * bottom_chunk * bond_minus
                    + middle_chunk * mid_bond_ps
                )

                coord_rect_ps = (
                    top_chunk * top_w_chunk * pos_coord_ps
                    + self.lfpo_bottom_repulsion_weight * bottom_chunk * coord_minus
                    + middle_chunk * mid_coord_ps
                )

                charge_rect_ps = (
                    top_chunk * top_w_chunk * pos_charge_ps
                    + self.lfpo_bottom_repulsion_weight * bottom_chunk * charge_minus
                    + middle_chunk * mid_charge_ps
                )

                main_ps = (
                    self.lfpo_lambda_coord_rect * coord_rect_ps
                    + self.lfpo_lambda_types_rect * type_rect_ps
                    + self.lfpo_lambda_bonds_rect * bond_rect_ps
                    + (self.lfpo_lambda_charges_rect * charge_rect_ps if has_charge_head else 0.0)
                )

                pos_ps = top_chunk * top_w_chunk * (
                    self.lfpo_lambda_coord_rect * pos_coord_ps
                    + self.lfpo_lambda_types_rect * pos_type_ps
                    + self.lfpo_lambda_bonds_rect * pos_bond_ps
                    + (self.lfpo_lambda_charges_rect * pos_charge_ps if has_charge_head else 0.0)
                )
                neg_ps = bottom_chunk * (
                    self.lfpo_lambda_coord_rect * coord_minus
                    + self.lfpo_lambda_types_rect * type_minus
                    + self.lfpo_lambda_bonds_rect * bond_minus
                    + (self.lfpo_lambda_charges_rect * charge_minus if has_charge_head else 0.0)
                )

                aux_ps = (
                    aux_losses["coord-loss"]
                    + aux_losses["type-loss"]
                    + aux_losses["bond-loss"]
                    + aux_losses["charge-loss"]
                )

                if self.regularization_type == "KL":
                    anchor_losses = self._anchor_from_reference_per_sample(
                        predicted,
                        ref_predicted,
                        masks,
                        flag_3Ds,
                    )
                    anchor_ps = (
                        anchor_losses["coord"]
                        + anchor_losses["types"]
                        + anchor_losses["bonds"]
                        + anchor_losses["charges"]
                    )
                else:
                    anchor_ps = torch.zeros_like(main_ps)

                # Correct chunk-wise scaling. Main loss is averaged over selected
                # top/bottom samples only; aux/anchor are averaged over all B*K.
                chunk_main = main_ps.sum() / selected_total
                chunk_pos = pos_ps.sum() / selected_total
                chunk_neg = neg_ps.sum() / selected_total
                chunk_type = type_rect_ps.sum() / selected_total
                chunk_bond = bond_rect_ps.sum() / selected_total
                chunk_coord = coord_rect_ps.sum() / selected_total
                chunk_charge = charge_rect_ps.sum() / selected_total
                chunk_aux = aux_ps.sum() / total_n
                chunk_anchor = anchor_ps.sum() / total_n
                chunk_total = chunk_main + self.lfpo_aux_fm_weight * chunk_aux + self.anchor_weight * chunk_anchor

                self.manual_backward(chunk_total)

                lfpof_main_loss = lfpof_main_loss + chunk_main.detach()
                pos_imitation_loss = pos_imitation_loss + chunk_pos.detach()
                neg_repulsion_loss = neg_repulsion_loss + chunk_neg.detach()
                type_rect_loss = type_rect_loss + chunk_type.detach()
                bond_rect_loss = bond_rect_loss + chunk_bond.detach()
                coord_rect_loss = coord_rect_loss + chunk_coord.detach()
                charge_rect_loss = charge_rect_loss + chunk_charge.detach()
                aux_fm_loss = aux_fm_loss + chunk_aux.detach()
                anchor_loss = anchor_loss + chunk_anchor.detach()
                total_loss_log = total_loss_log + chunk_total.detach()
                # Delta logs remain averages over all processed samples.
                n = float(end - start)
                scale = n / total_n
                delta_type_abs_mean = delta_type_abs_mean + delta_type_abs.mean().detach() * scale
                delta_bond_abs_mean = delta_bond_abs_mean + delta_bond_abs.mean().detach() * scale
                delta_coord_abs_mean = delta_coord_abs_mean + delta_coord_abs.mean().detach() * scale
                delta_charge_abs_mean = delta_charge_abs_mean + delta_charge_abs.mean().detach() * scale

                del interp_data, cond_batch, predicted, ref_predicted, aux_losses, aux_target
                del type_plus, type_minus, bond_plus, bond_minus, coord_plus, coord_minus
                del type_rect_ps, bond_rect_ps, coord_rect_ps, charge_rect_ps, main_ps, aux_ps
                del pos_type_ps, pos_bond_ps, pos_charge_ps, pos_coord_ps, pos_ps, neg_ps

        if self.regularization_type == "Parametric_L2" and self.anchor_weight > 0:
            param_l2 = self.compute_l2_regularization()
            self.manual_backward(self.anchor_weight * param_l2)
            anchor_loss = anchor_loss + param_l2.detach()
            total_loss_log = total_loss_log + (self.anchor_weight * param_l2).detach()

        opt.step()
        opt.zero_grad()

        scheduler = self.lr_schedulers()
        if scheduler is not None:
            if isinstance(scheduler, (list, tuple)):
                for sch in scheduler:
                    sch.step()
            else:
                scheduler.step()

        # ----- logs -----
        self._log_debug("train-lfpof-reward-ref-mean", rewards_ref.mean(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self._log_debug("train-lfpof-reward-ref-max", rewards_ref.max(), on_step=True, logger=True, sync_dist=True)
        self._log_debug("train-lfpof-reward-ref-top10-mean", self._top_frac_mean(rewards_ref, 0.1), on_step=True, logger=True, sync_dist=True)
        if top_mask.any():
            self._log_debug(
                "train-lift-top-imitation-weight-mean",
                top_imitation_w[top_mask].mean(),
                on_step=True,
                logger=True,
                sync_dist=True,
            )
            self._log_debug(
                "train-lift-top-imitation-weight-min",
                top_imitation_w[top_mask].min(),
                on_step=True,
                logger=True,
                sync_dist=True,
            )
            self._log_debug(
                "train-lift-top-imitation-weight-max",
                top_imitation_w[top_mask].max(),
                on_step=True,
                logger=True,
                sync_dist=True,
            )
        self._log_debug("train-lfpof-pull-weight-mean", pull_w.mean(), on_step=True, logger=True, sync_dist=True)
        self._log_debug("train-lfpof-pull-weight-min", pull_w.min(), on_step=True, logger=True, sync_dist=True)
        self._log_debug("train-lfpof-pull-weight-max", pull_w.max(), on_step=True, logger=True, sync_dist=True)

        self._log_debug("train-lfpof-main-loss", lfpof_main_loss, on_step=True, logger=True, sync_dist=True)
        self._log_main("train-lfpof-pos-imitation-loss", pos_imitation_loss, on_step=True, logger=True, sync_dist=True)
        self._log_main("train-lfpof-neg-repulsion-loss", neg_repulsion_loss, on_step=True, logger=True, sync_dist=True)
        self._log_debug("train-lfpof-coord-rect-loss", coord_rect_loss, on_step=True, logger=True, sync_dist=True)
        self._log_main("train-lfpof-type-rect-loss", type_rect_loss, on_step=True, logger=True, sync_dist=True)
        self._log_main("train-lfpof-bond-rect-loss", bond_rect_loss, on_step=True, logger=True, sync_dist=True)
        self._log_debug("train-lfpof-aux-fm-loss", aux_fm_loss, on_step=True, logger=True, sync_dist=True)
        self._log_debug("train-lfpof-anchor-loss", anchor_loss, on_step=True, logger=True, sync_dist=True)
        self._log_debug("train-lfpof-total-loss", total_loss_log, on_step=True, logger=True, sync_dist=True)
        if has_charge_head:
            self.log("train-lfpof-charge-rect-loss", charge_rect_loss, on_step=True, logger=True, sync_dist=True)

        self.log("train-lfpof-delta-type-abs-mean", delta_type_abs_mean, on_step=True, logger=True, sync_dist=True)
        self.log("train-lfpof-delta-bond-abs-mean", delta_bond_abs_mean, on_step=True, logger=True, sync_dist=True)
        self.log("train-lfpof-delta-coord-abs-mean", delta_coord_abs_mean, on_step=True, logger=True, sync_dist=True)
        if has_charge_head:
            self.log("train-lfpof-delta-charge-abs-mean", delta_charge_abs_mean, on_step=True, logger=True, sync_dist=True)

        self.log("train-gen-ref-validity", quality_metrics_ref["validity"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-ref-uniqueness", quality_metrics_ref["uniqueness"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-ref-connected-validity", quality_metrics_ref["connected-validity"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self._log_debug("train-gen-ref-n-valid", quality_metrics_ref["n-valid"], on_step=True, logger=True, sync_dist=True)
        self._log_debug("train-gen-ref-n-total", quality_metrics_ref["n-total"], on_step=True, logger=True, sync_dist=True)
        self.log("train-mpo-ref-score-mean", scoring_ref.score.mean(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-mpo-ref-score-top10-mean", self._top_frac_mean(scoring_ref.score, 0.1), on_step=True, logger=True, sync_dist=True)
        self.log("train-partition-ref-feasible-frac", scoring_ref.feasible.float().mean(), on_step=True, logger=True, sync_dist=True)
        self.log("train-partition-ref-severe-frac", scoring_ref.severe_violation.float().mean(), on_step=True, logger=True, sync_dist=True)
        self.log("train-partition-ref-top-frac", top_mask.float().mean(), on_step=True, logger=True, sync_dist=True)
        self.log("train-partition-ref-bottom-frac", bottom_mask.float().mean(), on_step=True, logger=True, sync_dist=True)
        for k, v in scoring_ref.component_scores.items():
            self._log_debug(f"train-mpo-ref-comp-{k}-mean", v.mean(), on_step=True, logger=True, sync_dist=True)
        for k, v in scoring_ref.raw_properties.items():
            self._log_debug(f"train-mpo-ref-raw-{k}-mean", v.mean(), on_step=True, logger=True, sync_dist=True)
        if "sim_ranolazine_AP" in scoring_ref.raw_properties:
            rs = scoring_ref.raw_properties["sim_ranolazine_AP"]; csim = scoring_ref.component_scores.get("sim_ranolazine_AP", rs)
            self.log("train-mpo-ref-all-raw-sim_ranolazine_AP-mean", rs.mean(), on_step=True, logger=True, sync_dist=True)
            self.log("train-mpo-ref-all-comp-sim_ranolazine_AP-mean", csim.mean(), on_step=True, logger=True, sync_dist=True)
            self.log("train-mpo-ref-all-raw-sim_ranolazine_AP-top10", self._top_frac_mean(rs,0.1), on_step=True, logger=True, sync_dist=True)
            self.log("train-mpo-ref-all-raw-sim_ranolazine_AP-max", rs.max(), on_step=True, logger=True, sync_dist=True)
            self.log("train-mpo-ref-all-comp-sim_ranolazine_AP-std", csim.std(unbiased=False), on_step=True, logger=True, sync_dist=True)
            self.log("train-mpo-ref-all-raw-sim_ranolazine_AP-std", rs.std(unbiased=False), on_step=True, logger=True, sync_dist=True)
            self._log_masked_mean("train-mpo-ref-top-raw-sim_ranolazine_AP-mean", rs, top_mask)
            self._log_masked_mean("train-mpo-ref-bottom-raw-sim_ranolazine_AP-mean", rs, bottom_mask)
        if "num_F" in scoring_ref.raw_properties:
            rn = scoring_ref.raw_properties["num_F"]; cn = scoring_ref.component_scores.get("num_F", rn)
            self.log("train-mpo-ref-all-raw-num_F-mean", rn.mean(), on_step=True, logger=True, sync_dist=True)
            self.log("train-mpo-ref-all-comp-num_F-mean", cn.mean(), on_step=True, logger=True, sync_dist=True)
            self.log("train-mpo-ref-all-frac-num_F-eq-1", (rn==1).float().mean(), on_step=True, logger=True, sync_dist=True)
            self.log("train-mpo-ref-all-frac-num_F-gt-0", (rn>0).float().mean(), on_step=True, logger=True, sync_dist=True)
            self._log_masked_mean("train-mpo-ref-top-raw-num_F-mean", rn, top_mask)
            self._log_masked_mean("train-mpo-ref-top-comp-num_F-mean", cn, top_mask)
            self._log_masked_mean("train-mpo-ref-bottom-raw-num_F-mean", rn, bottom_mask)
        if "TPSA" in scoring_ref.raw_properties:
            self.log("train-mpo-ref-raw-TPSA-mean", scoring_ref.raw_properties["TPSA"].mean(), on_step=True, logger=True, sync_dist=True)
        if "logP" in scoring_ref.raw_properties:
            self.log("train-mpo-ref-raw-logP-mean", scoring_ref.raw_properties["logP"].mean(), on_step=True, logger=True, sync_dist=True)
        if partition_ref.pareto_rank is not None:
            self.log("train-partition-ref-pareto-rank-mean", partition_ref.pareto_rank.float().mean(), on_step=True, logger=True, sync_dist=True)
        for dk,dv in partition_ref.diagnostics.items():
            self.log(f"train-partition-ref-{dk.replace('_','-')}", dv, on_step=True, logger=True, sync_dist=True)
        if scoring_ref.metadata.get("official_score") is not None:
            self.log("train-mpo-ref-official-score-mean", scoring_ref.metadata["official_score"].mean(), on_step=True, logger=True, sync_dist=True)
        self.log("train-mpo-ref-geometric-score-mean", scoring_ref.metadata.get("geometric_score", scoring_ref.score).mean(), on_step=True, logger=True, sync_dist=True)
        self.log("train-mpo-ref-tchebycheff-score-mean", scoring_ref.metadata.get("tchebycheff_score", scoring_ref.score).mean(), on_step=True, logger=True, sync_dist=True)
        self.log("train-mpo-ref-min-component-mean", scoring_ref.metadata.get("min_component_score", scoring_ref.score).mean(), on_step=True, logger=True, sync_dist=True)
        if "sim_ranolazine_AP" in scoring_ref.component_scores:
            self.log("train-mpo-ref-corr-score-comp-sim", self._safe_corr(scoring_ref.score, scoring_ref.component_scores["sim_ranolazine_AP"]), on_step=True, logger=True, sync_dist=True)
        if "num_F" in scoring_ref.component_scores:
            self.log("train-mpo-ref-corr-score-comp-numF", self._safe_corr(scoring_ref.score, scoring_ref.component_scores["num_F"]), on_step=True, logger=True, sync_dist=True)

        if scoring_cur is not None:
            self.log("train-lfpof-reward-current-mean", scoring_cur.score.mean(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log("train-lfpof-reward-current-sem", self._sem(scoring_cur.score), on_step=True, on_epoch=True, logger=True, sync_dist=True)
            self.log("train-lfpof-reward-current-max", scoring_cur.score.max(), on_step=True, logger=True, sync_dist=True)
            self.log("train-lfpof-reward-current-top10-mean", self._top_frac_mean(scoring_cur.score, 0.1), on_step=True, on_epoch=True, logger=True, sync_dist=True)
            if cur_top_mask is not None and cur_top_mask.any():
                self.log("train-lfpof-reward-current-top-mean", scoring_cur.score[cur_top_mask].mean(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
                self.log("train-lfpof-reward-current-top-sem", self._sem(scoring_cur.score[cur_top_mask]), on_step=True, on_epoch=True, logger=True, sync_dist=True)
            if cur_bottom_mask is not None and cur_bottom_mask.any():
                self.log("train-lfpof-reward-current-bottom-mean", scoring_cur.score[cur_bottom_mask].mean(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
                self.log("train-lfpof-reward-current-bottom-sem", self._sem(scoring_cur.score[cur_bottom_mask]), on_step=True, on_epoch=True, logger=True, sync_dist=True)

            if quality_metrics_cur is not None:
                self.log("train-gen-current-validity", quality_metrics_cur["validity"], on_step=True, logger=True, sync_dist=True)
                self.log("train-gen-current-uniqueness", quality_metrics_cur["uniqueness"], on_step=True, logger=True, sync_dist=True)
                self.log("train-gen-current-connected-validity", quality_metrics_cur["connected-validity"], on_step=True, logger=True, sync_dist=True)
                self._log_debug("train-gen-current-n-valid", quality_metrics_cur["n-valid"], on_step=True, logger=True, sync_dist=True)
                self._log_debug("train-gen-current-n-total", quality_metrics_cur["n-total"], on_step=True, logger=True, sync_dist=True)
                self.log("train-mpo-current-score-mean", scoring_cur.score.mean(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
                self.log("train-mpo-current-score-sem", self._sem(scoring_cur.score), on_step=True, on_epoch=True, logger=True, sync_dist=True)
                self.log("train-mpo-current-score-top10-mean", self._top_frac_mean(scoring_cur.score, 0.1), on_step=True, on_epoch=True, logger=True, sync_dist=True)
                self.log("train-mpo-current-score-top1", torch.max(scoring_cur.score), on_step=True, on_epoch=True, logger=True, sync_dist=True)
                if "sim_ranolazine_AP" in scoring_cur.raw_properties:
                    rsc = scoring_cur.raw_properties["sim_ranolazine_AP"]
                    self.log("train-mpo-current-all-raw-sim_ranolazine_AP-mean", rsc.mean(), on_step=True, logger=True, sync_dist=True)
                    self.log("train-mpo-current-all-raw-sim_ranolazine_AP-top10", self._top_frac_mean(rsc, 0.1), on_step=True, logger=True, sync_dist=True)
                    self.log("train-mpo-current-all-raw-sim_ranolazine_AP-max", rsc.max(), on_step=True, logger=True, sync_dist=True)
                if "num_F" in scoring_cur.raw_properties:
                    rnf = scoring_cur.raw_properties["num_F"]
                    self.log("train-mpo-current-all-raw-num_F-mean", rnf.mean(), on_step=True, logger=True, sync_dist=True)
                    self.log("train-mpo-current-all-frac-num_F-eq-1", (rnf == 1).float().mean(), on_step=True, logger=True, sync_dist=True)
                    self.log("train-mpo-current-all-frac-num_F-gt-0", (rnf > 0).float().mean(), on_step=True, logger=True, sync_dist=True)
                if "TPSA" in scoring_cur.raw_properties:
                    self.log("train-mpo-current-raw-TPSA-mean", scoring_cur.raw_properties["TPSA"].mean(), on_step=True, logger=True, sync_dist=True)
                if "logP" in scoring_cur.raw_properties:
                    self.log("train-mpo-current-raw-logP-mean", scoring_cur.raw_properties["logP"].mean(), on_step=True, logger=True, sync_dist=True)
                for key in [
                    "strain-energy-mean",
                    "strain-energy-median",
                    "strain-energy-per-atom-mean",
                    "strain-energy-per-atom-median",
                    "n-strain-success",
                ]:
                    if key in quality_metrics_cur:
                        self.log(f"train-gen-current-{key}", quality_metrics_cur[key], on_step=True, on_epoch=True, logger=True, sync_dist=True)


        if self.oracle_logger is not None and getattr(self.trainer, "global_rank", 0) == 0:
            if self.metric_config.get("log_ref_train", True):
                self.oracle_logger.log_batch(int(self.global_step), "ref_train", scoring_ref, partition_ref)
            if scoring_cur is not None and self.metric_config.get("log_current_eval", True):
                self.oracle_logger.log_batch(int(self.global_step), "current_eval", scoring_cur, None)

        return total_loss_log.detach()

    def training_step(self, batch, b_idx):
        batch = self.flatten_batch(batch)
        loss = self.FM_training_step(batch)
        self.log("train-loss", loss, prog_bar=True, on_step=True, logger=True, sync_dist=True)
        return loss
