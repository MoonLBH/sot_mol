import copy
import random
from typing import Optional

import torch
from rdkit.Chem import QED

from .diff import SC_Lightning


class RolloutBuffer:
    def __init__(self, capacity: int = 4096):
        self.capacity = max(1, int(capacity))
        self._data = []

    def add_batch(self, records):
        self._data.extend(records)
        if len(self._data) > self.capacity:
            self._data = self._data[-self.capacity :]

    def __len__(self):
        return len(self._data)

    def sample(self, batch_size: int):
        batch_size = min(batch_size, len(self._data))
        if batch_size <= 0:
            return []
        idx = random.sample(range(len(self._data)), k=batch_size)
        return [self._data[i] for i in idx]


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
        group_size: int = 4,
        rollout_batch_size: int = 64,
        rollout_buffer_size: int = 4096,
        beta: float = 1.0,
        eta_max: float = 0.5,
        eta_scale: float = 1e-3,
        reward_norm_eps: float = 1e-6,
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
        self.rollout_batch_size = max(1, int(rollout_batch_size))
        self.beta = float(beta)
        self.eta_max = float(eta_max)
        self.eta_scale = float(eta_scale)
        self.reward_norm_eps = float(reward_norm_eps)

        self.rollout_buffer = RolloutBuffer(capacity=rollout_buffer_size)
        self.model_old = None

    def on_fit_start(self):
        super().on_fit_start()
        if self.model_old is None:
            self.model_old = copy.deepcopy(self.gen)
            self.model_old.eval()
            for p in self.model_old.parameters():
                p.requires_grad = False

    def _build_noise_batch(self, batch):
        return {
            "coords": batch["noise_coords"],
            "atomics": batch["noise_atomics"],
            "bonds": batch["noise_bonds"],
            "masks": batch["masks"],
            "flag_3Ds": batch["flag_3Ds"],
        }

    def _reward_fn(self, generated):
        rewards = []
        mols = self._generate_mols(generated, sanitise=True)
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
        return torch.tensor(rewards, dtype=generated["coords"].dtype, device=generated["coords"].device)

    def _repeat_noise(self, noise, repeats):
        repeated = {}
        for k, v in noise.items():
            if isinstance(v, torch.Tensor):
                repeated[k] = v.repeat_interleave(repeats, dim=0)
            else:
                repeated[k] = v
        return repeated

    def _normalize_group_rewards(self, rewards, group_size):
        rewards = rewards.view(-1, group_size)
        group_mean = rewards.mean(dim=1, keepdim=True)
        centered = rewards - group_mean
        group_std = centered.std(dim=1, unbiased=False, keepdim=True)

        safe_std = torch.where(group_std > 0, group_std, torch.ones_like(group_std))
        scaled = torch.clamp(centered / safe_std, min=-1.0, max=1.0)
        probs = 0.5 + 0.5 * scaled
        probs = torch.where(group_std > 0, probs, torch.full_like(probs, 0.5))
        return probs.view(-1)

    def _collect_rollout(self, batch):
        noise = self._build_noise_batch(batch)
        rollout_noise = self._repeat_noise(noise, self.group_size)

        with torch.no_grad():
            generated = self._generate(
                rollout_noise,
                inference_steps=self.max_steps,
                coord_noise_std=self.default_coord_noise_std,
                cat_noise_level=self.default_cat_noise_level,
            )
            rewards = self._reward_fn(generated)

        reward_probs = self._normalize_group_rewards(rewards, self.group_size)

        records = []
        for i in range(generated["coords"].size(0)):
            records.append(
                {
                    "coords": generated["coords"][i].detach().cpu(),
                    "atomics": generated["atomics"][i].detach().cpu(),
                    "bonds": generated["bonds"][i].detach().cpu(),
                    "charges": generated["charges"][i].detach().cpu(),
                    "masks": generated["masks"][i].detach().cpu(),
                    "flag_3Ds": generated["flag_3Ds"][i].detach().cpu(),
                    "r": reward_probs[i].detach().cpu(),
                }
            )

        self.rollout_buffer.add_batch(records)
        return rewards.mean(), rewards.max(), reward_probs.mean()

    def _batch_from_records(self, records, device):
        keys = ["coords", "atomics", "bonds", "charges", "masks", "flag_3Ds", "r"]
        out = {}
        for key in keys:
            tensors = [row[key] for row in records]
            out[key] = torch.stack(tensors, dim=0).to(device)
        return out

    def _model_predict(self, model, x_t, t, atomics, bonds, masks, flag_3Ds):
        t = t.view(-1, 1, 1)
        cond_batch = None
        if self.self_cond:
            cond_batch = {
                "coords": torch.zeros_like(x_t),
                "atomics": torch.zeros_like(atomics),
                "bonds": torch.zeros_like(bonds),
            }

        pred_coords, _, _, _ = model(
            x_t,
            atomics,
            edge_feats=bonds,
            t=t,
            cond_coords=None if cond_batch is None else cond_batch["coords"],
            cond_atomics=None if cond_batch is None else cond_batch["atomics"],
            cond_bonds=None if cond_batch is None else cond_batch["bonds"],
            atom_mask=masks,
            flag_3Ds=flag_3Ds,
        )
        return pred_coords

    def FM_training_step(self, batch):
        reward_mean, reward_max, reward_prob_mean = self._collect_rollout(batch)

        if len(self.rollout_buffer) < 1:
            return torch.zeros([], device=self.device, requires_grad=True)

        records = self.rollout_buffer.sample(self.rollout_batch_size)
        sampled = self._batch_from_records(records, self.device)

        x0_hat = sampled["coords"]
        atomics = sampled["atomics"]
        bonds = sampled["bonds"]
        masks = sampled["masks"]
        flag_3Ds = sampled["flag_3Ds"]
        rewards = sampled["r"]

        t = torch.rand(x0_hat.size(0), device=self.device, dtype=x0_hat.dtype)
        eps = torch.randn_like(x0_hat)

        alpha_t = t.view(-1, 1, 1)
        sigma_t = (1.0 - t).view(-1, 1, 1)
        x_t = (alpha_t * x0_hat) + (sigma_t * eps)

        with torch.no_grad():
            x0_old = self._model_predict(self.model_old, x_t, t, atomics, bonds, masks, flag_3Ds)
        x0_theta = self._model_predict(self.gen, x_t, t, atomics, bonds, masks, flag_3Ds)

        x0_pos = (1.0 - self.beta) * x0_old + self.beta * x0_theta
        x0_neg = (1.0 + self.beta) * x0_old - self.beta * x0_theta

        mse_pos = ((x0_pos - x0_hat) ** 2).mean(dim=(1, 2))
        mse_neg = ((x0_neg - x0_hat) ** 2).mean(dim=(1, 2))

        raw_loss = rewards * mse_pos + (1.0 - rewards) * mse_neg

        residual = torch.mean(torch.abs(x0_theta - x0_hat), dim=(1, 2)).detach()
        weight = 1.0 / (residual + 1e-6)
        loss = (raw_loss * weight).mean()

        self.log("train-rl-loss", loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-reward-mean", reward_mean, on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-reward-max", reward_max, on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-r-mean", reward_prob_mean, on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-buffer-size", float(len(self.rollout_buffer)), on_step=True, logger=True, sync_dist=True)
        self.log("train-rl-weight-mean", weight.mean(), on_step=True, logger=True, sync_dist=True)

        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure=optimizer_closure)
        self._soft_update_old_model()

    @torch.no_grad()
    def _soft_update_old_model(self):
        if self.model_old is None:
            return

        step_i = max(1, int(self.global_step))
        eta_i = min(self.eta_scale * step_i, self.eta_max)

        for old_param, param in zip(self.model_old.parameters(), self.gen.parameters()):
            old_param.data.mul_(eta_i).add_(param.data, alpha=(1.0 - eta_i))

        for old_buf, buf in zip(self.model_old.buffers(), self.gen.buffers()):
            old_buf.data.mul_(eta_i).add_(buf.data, alpha=(1.0 - eta_i))

        self.log("train-rl-eta", eta_i, on_step=True, logger=True, sync_dist=True)

    def validation_step(self, batch, b_idx):
        return

    def on_validation_epoch_end(self):
        return

    def test_step(self, batch, batch_idx):
        return

    def on_test_epoch_end(self):
        return
