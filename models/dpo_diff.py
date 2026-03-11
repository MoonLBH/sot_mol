import copy

import torch
import torch.nn.functional as F

from .rl_diff import RL_Lightning


class DPO_Lightning(RL_Lightning):
    def __init__(
        self,
        *args,
        dpo_beta: float = 0.1,
        dpo_label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dpo_beta = dpo_beta
        self.dpo_label_smoothing = dpo_label_smoothing

    def on_fit_start(self):
        super().on_fit_start()
        if self.ref_gen is None:
            self.ref_gen = copy.deepcopy(self.gen)
            self.ref_gen.eval()
            for param in self.ref_gen.parameters():
                param.requires_grad = False

    def _sample_generated_batch(self, batch):
        noise = self._build_noise_batch(batch)
        with torch.no_grad():
            generated = self._generate(
                noise,
                inference_steps=self.max_steps,
                coord_noise_std=self.default_coord_noise_std,
                cat_noise_level=self.default_cat_noise_level,
            )
        rewards, mols = self._compute_rewards_from_generated(generated)
        target_batch = self._build_generated_target_batch(batch, generated)
        return target_batch, rewards, mols

    def _policy_logprob(self, model, train_batch, t, flag_3Ds):
        interp_data = self.interpolate(train_batch, t, flag_3Ds=flag_3Ds)

        cond_batch = None
        if self.self_cond:
            cond_batch = {
                "coords": torch.zeros_like(interp_data["coords"]),
                "atomics": torch.zeros_like(interp_data["atomics"]),
                "bonds": torch.zeros_like(interp_data["bonds"]),
            }

        coords, types, bonds, charges = self._forward_with_model(
            model,
            interp_data,
            t,
            cond_batch=cond_batch,
            flag_3Ds=flag_3Ds,
        )

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
        predicted = {
            "coords": coords,
            "atomics": types,
            "bonds": bonds,
            "charges": charges,
        }

        losses = self._loss_per_sample(target, predicted, flag_3Ds=flag_3Ds)
        total_loss = losses["coord-loss"] + losses["type-loss"] + losses["bond-loss"] + losses["charge-loss"]
        return -total_loss

    def FM_training_step(self, batch):
        chosen_batch, chosen_rewards, chosen_mols = self._sample_generated_batch(batch)
        rejected_batch, rejected_rewards, rejected_mols = self._sample_generated_batch(batch)

        preference_mask = (chosen_rewards > rejected_rewards).float()

        better_batch = {}
        worse_batch = {}
        for key in chosen_batch.keys():
            expand_dims = [1] * (chosen_batch[key].dim() - 1)
            mask = preference_mask.view(-1, *expand_dims)
            better_batch[key] = (mask * chosen_batch[key]) + ((1.0 - mask) * rejected_batch[key])
            worse_batch[key] = (mask * rejected_batch[key]) + ((1.0 - mask) * chosen_batch[key])

        better_rewards = (preference_mask * chosen_rewards) + ((1.0 - preference_mask) * rejected_rewards)
        worse_rewards = (preference_mask * rejected_rewards) + ((1.0 - preference_mask) * chosen_rewards)

        flag_3Ds = better_batch["flag_3Ds"]
        batchsize = better_batch["natoms"].size(0)
        device = better_batch["real_coords"].device
        t = self.time_dist.sample((batchsize,)).to(device)

        pi_logp_better = self._policy_logprob(self.gen, better_batch, t, flag_3Ds)
        pi_logp_worse = self._policy_logprob(self.gen, worse_batch, t, flag_3Ds)

        with torch.no_grad():
            ref_logp_better = self._policy_logprob(self.ref_gen, better_batch, t, flag_3Ds)
            ref_logp_worse = self._policy_logprob(self.ref_gen, worse_batch, t, flag_3Ds)

        logits = self.dpo_beta * ((pi_logp_better - pi_logp_worse) - (ref_logp_better - ref_logp_worse))
        dpo_loss = -F.logsigmoid(logits)

        if self.dpo_label_smoothing > 0:
            dpo_loss = (1.0 - self.dpo_label_smoothing) * dpo_loss - self.dpo_label_smoothing * F.logsigmoid(-logits)

        dpo_loss = dpo_loss.mean()

        all_mols = chosen_mols + rejected_mols
        quality_metrics = self._compute_generation_quality_from_mols(
            all_mols,
            dtype=chosen_rewards.dtype,
            device=chosen_rewards.device,
        )

        reward_margin = (better_rewards - worse_rewards).mean()
        pair_acc = (logits > 0).float().mean()

        self.log("train-dpo-loss", dpo_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-dpo-logit-mean", logits.mean(), on_step=True, logger=True, sync_dist=True)
        self.log("train-dpo-reward-margin", reward_margin, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-dpo-pair-acc", pair_acc, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-dpo-better-reward", better_rewards.mean(), on_step=True, logger=True, sync_dist=True)
        self.log("train-dpo-worse-reward", worse_rewards.mean(), on_step=True, logger=True, sync_dist=True)

        self.log("train-gen-validity", quality_metrics["validity"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-uniqueness", quality_metrics["uniqueness"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log("train-gen-connected-validity", quality_metrics["connected-validity"], on_step=True, on_epoch=True, logger=True, sync_dist=True)

        return dpo_loss

    def validation_step(self, batch, b_idx):
        return

    def on_validation_epoch_end(self):
        return

    def test_step(self, batch, batch_idx):
        return

    def on_test_epoch_end(self):
        return
