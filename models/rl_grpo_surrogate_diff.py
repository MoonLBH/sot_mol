from typing import Optional

import torch

from .rl_diff import RL_Lightning


class RL_GRPO_Surrogate_Lightning(RL_Lightning):
    """GRPO-style flow-matching finetuning with switchable surrogate objectives.

    surrogate_mode options:
      - "single_time_surrogate": one sampled time per batch element.
      - "multi_time_surrogate": multiple sampled times per batch element, averaged.
      - "trajectory_surrogate": fixed trajectory-like time grid, averaged.
    """

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
        surrogate_mode: str = "single_time_surrogate",
        adv_clip: float = 5.0,
        multi_time_samples: int = 4,
        trajectory_steps: int = 8,
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
            reward_name=reward_name,
            reward_beta=reward_beta,
            reward_weight_min=reward_weight_min,
            reward_weight_max=reward_weight_max,
            reward_norm_eps=reward_norm_eps,
            anchor_weight=anchor_weight,
            anchor_loss_weight=anchor_loss_weight,
            use_reference_anchor=use_reference_anchor,
        )

        valid_modes = {"single_time_surrogate", "multi_time_surrogate", "trajectory_surrogate"}
        if surrogate_mode not in valid_modes:
            raise ValueError(f"Unsupported surrogate_mode='{surrogate_mode}'. Valid: {sorted(valid_modes)}")

        self.surrogate_mode = surrogate_mode
        self.adv_clip = adv_clip
        self.multi_time_samples = max(1, int(multi_time_samples))
        self.trajectory_steps = max(2, int(trajectory_steps))

    def _standardized_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        centered = rewards - rewards.mean()
        adv = centered / (centered.std(unbiased=False) + self.reward_norm_eps)
        return torch.clamp(adv, min=-self.adv_clip, max=self.adv_clip)

    def _loss_components_at_t(self, train_batch, t, flag_3Ds):
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
                        "atomics": torch.softmax(cond_types, dim=-1),
                        "bonds": torch.softmax(cond_bonds, dim=-1),
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
        )

        return losses, anchor_loss

    def _aggregate_per_sample_losses(self, train_batch, flag_3Ds):
        batchsize = train_batch["natoms"].size(0)
        device = train_batch["real_coords"].device

        if self.surrogate_mode == "single_time_surrogate":
            t = self.time_dist.sample((batchsize,)).to(device)
            losses, anchor = self._loss_components_at_t(train_batch, t, flag_3Ds)
            return losses, anchor

        if self.surrogate_mode == "multi_time_surrogate":
            acc_losses = None
            acc_anchor = 0.0
            for _ in range(self.multi_time_samples):
                t = self.time_dist.sample((batchsize,)).to(device)
                losses, anchor = self._loss_components_at_t(train_batch, t, flag_3Ds)
                if acc_losses is None:
                    acc_losses = {k: v.clone() for k, v in losses.items()}
                else:
                    for k in acc_losses:
                        acc_losses[k] = acc_losses[k] + losses[k]
                acc_anchor = acc_anchor + anchor

            for k in acc_losses:
                acc_losses[k] = acc_losses[k] / self.multi_time_samples
            acc_anchor = acc_anchor / self.multi_time_samples
            return acc_losses, acc_anchor

        # trajectory_surrogate: average over fixed monotonic time grid
        grid = torch.linspace(0.0, 1.0, self.trajectory_steps + 2, device=device)[1:-1]
        acc_losses = None
        acc_anchor = 0.0
        for t_scalar in grid:
            t = torch.full((batchsize,), t_scalar.item(), device=device)
            losses, anchor = self._loss_components_at_t(train_batch, t, flag_3Ds)
            if acc_losses is None:
                acc_losses = {k: v.clone() for k, v in losses.items()}
            else:
                for k in acc_losses:
                    acc_losses[k] = acc_losses[k] + losses[k]
            acc_anchor = acc_anchor + anchor

        denom = grid.numel()
        for k in acc_losses:
            acc_losses[k] = acc_losses[k] / denom
        acc_anchor = acc_anchor / denom
        return acc_losses, acc_anchor

    def _surrogate_weighted_mean(self, per_sample_loss: torch.Tensor, advantages: torch.Tensor):
        return (advantages.detach() * per_sample_loss).mean()

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

        advantages = self._standardized_advantages(rewards)
        train_batch = self._build_generated_target_batch(batch, generated)
        flag_3Ds = train_batch["flag_3Ds"]

        losses, anchor_per_sample = self._aggregate_per_sample_losses(train_batch, flag_3Ds)

        coord_surr = self._surrogate_weighted_mean(losses["coord-loss"], advantages)
        type_surr = self._surrogate_weighted_mean(losses["type-loss"], advantages)
        bond_surr = self._surrogate_weighted_mean(losses["bond-loss"], advantages)
        charge_surr = self._surrogate_weighted_mean(losses["charge-loss"], advantages)
        fm_loss = coord_surr + type_surr + bond_surr + charge_surr

        anchor_loss = anchor_per_sample.mean()
        total_loss = fm_loss + (self.anchor_weight * self.anchor_loss_weight) * anchor_loss

        raw_losses = {name: loss_vals.mean() for name, loss_vals in losses.items()}
        mode_index = {
            "single_time_surrogate": 0,
            "multi_time_surrogate": 1,
            "trajectory_surrogate": 2,
        }[self.surrogate_mode]

        self.log("train-rl-surrogate-mode", float(mode_index), on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-adv-mean", advantages.mean(), on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-adv-std", advantages.std(unbiased=False), on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-weight-mean", advantages.mean(), on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-weight-max", advantages.max(), on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-weight-min", advantages.min(), on_step=True, logger=True, sync_dist=True)

        self.log("train-fm-coord-loss", coord_surr, on_step=True, logger=True, sync_dist=True)
        self.log("train-fm-type-loss", type_surr, on_step=True, logger=True, sync_dist=True)
        self.log("train-fm-bond-loss", bond_surr, on_step=True, logger=True, sync_dist=True)
        self.log("train-fm-charge-loss", charge_surr, on_step=True, logger=True, sync_dist=True)

        for name, loss_val in raw_losses.items():
            self.log(f"train-fm-raw-{name}", loss_val, on_step=True, logger=True, sync_dist=True)

        self.log("train-rl-reward-mean", rewards.mean(), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-rl-reward-max", rewards.max(), on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-anchor-loss", anchor_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-fm-loss", fm_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-total-loss", total_loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-gen-validity", quality_metrics["validity"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-uniqueness", quality_metrics["uniqueness"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-connected-validity", quality_metrics["connected-validity"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-n-valid", quality_metrics["n-valid"], on_step=True, logger=True, sync_dist=True)
        self.log("train-gen-n-total", quality_metrics["n-total"], on_step=True, logger=True, sync_dist=True)

        return total_loss
