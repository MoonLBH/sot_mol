import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from rdkit.Chem import QED

from .diff import SC_Lightning


@dataclass
class Transition:
    cond_atomics: torch.Tensor
    cond_bonds: torch.Tensor
    cond_masks: torch.Tensor
    cond_flag_3Ds: torch.Tensor
    x0_hat: torch.Tensor
    r: torch.Tensor


class RolloutBuffer:
    def __init__(self, capacity: int = 4096):
        self.capacity = max(1, int(capacity))
        self._items: List[Transition] = []

    def __len__(self):
        return len(self._items)

    def add(self, transitions: List[Transition]):
        self._items.extend(transitions)
        if len(self._items) > self.capacity:
            self._items = self._items[-self.capacity :]

    def sample(self, batch_size: int) -> List[Transition]:
        n = min(int(batch_size), len(self._items))
        if n <= 0:
            return []
        indices = random.sample(range(len(self._items)), k=n)
        return [self._items[i] for i in indices]


class OnlineRL_Lightning(SC_Lightning):
    """Online finetuning trainer with old/new policy split.

    - model_old: rollout + baseline prediction (no grad, no optimizer updates)
    - model_theta (self.gen): trainable policy
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
        eval_3D_props: bool = False,
        reward_name: str = "qed",
        group_size: int = 4,
        rollout_buffer_size: int = 4096,
        replay_batch_size: int = 64,
        beta: float = 1.0,
        eta_max: float = 0.5,
        eta_scale: float = 1e-3,
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
        self.replay_batch_size = max(1, int(replay_batch_size))
        self.beta = float(beta)
        self.eta_max = float(eta_max)
        self.eta_scale = float(eta_scale)

        self.rollout_buffer = RolloutBuffer(rollout_buffer_size)
        self.model_old: Optional[torch.nn.Module] = None

    def on_fit_start(self):
        super().on_fit_start()
        if self.model_old is None:
            self.model_old = copy.deepcopy(self.gen)
            self.model_old.eval()
            for p in self.model_old.parameters():
                p.requires_grad = False

    @staticmethod
    def _alpha_sigma(t: torch.Tensor):
        alpha = t.view(-1, 1, 1)
        sigma = (1.0 - t).view(-1, 1, 1)
        return alpha, sigma

    def _reward_fn(self, generated: Dict[str, torch.Tensor]) -> torch.Tensor:
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
        return torch.tensor(rewards, dtype=generated["coords"].dtype, device=generated["coords"].device)

    def _normalize_group_reward_to_r(self, reward_raw: torch.Tensor, group_size: int) -> torch.Tensor:
        grouped = reward_raw.view(-1, group_size)
        mean = grouped.mean(dim=1, keepdim=True)
        r_norm = grouped - mean
        std = r_norm.std(dim=1, unbiased=False, keepdim=True)

        std_safe = torch.where(std > 0, std, torch.ones_like(std))
        scaled = torch.clamp(r_norm / std_safe, -1.0, 1.0)
        r = 0.5 + 0.5 * scaled
        r = torch.where(std > 0, r, torch.full_like(r, 0.5))
        return r.reshape(-1)

    @staticmethod
    def _repeat_cond(cond: Dict[str, torch.Tensor], repeats: int) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in cond.items():
            out[k] = v.repeat_interleave(repeats, dim=0)
        return out

    def _collect_rollout_to_buffer(self, batch):
        cond = {
            "coords": batch["noise_coords"],
            "atomics": batch["noise_atomics"],
            "bonds": batch["noise_bonds"],
            "masks": batch["masks"],
            "flag_3Ds": batch["flag_3Ds"],
        }
        cond_group = self._repeat_cond(cond, self.group_size)

        with torch.no_grad():
            # 无 CFG：当前求解器接口无 CFG 参数，直接使用 model_old + cat_noise_level=1.0
            generated = self._generate(
                cond_group,
                inference_steps=self.max_steps,
                coord_noise_std=self.default_coord_noise_std,
                cat_noise_level=1.0,
            )
            reward_raw = self._reward_fn(generated)

        r = self._normalize_group_reward_to_r(reward_raw, self.group_size)

        transitions = []
        for i in range(generated["coords"].size(0)):
            transitions.append(
                Transition(
                    cond_atomics=cond_group["atomics"][i].detach().cpu(),
                    cond_bonds=cond_group["bonds"][i].detach().cpu(),
                    cond_masks=cond_group["masks"][i].detach().cpu(),
                    cond_flag_3Ds=cond_group["flag_3Ds"][i].detach().cpu(),
                    x0_hat=generated["coords"][i].detach().cpu(),
                    r=r[i].detach().cpu(),
                )
            )
        self.rollout_buffer.add(transitions)

        return reward_raw.mean(), reward_raw.max(), r.mean()

    def _to_batch(self, items: List[Transition], device):
        return {
            "cond_atomics": torch.stack([x.cond_atomics for x in items], dim=0).to(device),
            "cond_bonds": torch.stack([x.cond_bonds for x in items], dim=0).to(device),
            "cond_masks": torch.stack([x.cond_masks for x in items], dim=0).to(device),
            "cond_flag_3Ds": torch.stack([x.cond_flag_3Ds for x in items], dim=0).to(device),
            "x0_hat": torch.stack([x.x0_hat for x in items], dim=0).to(device),
            "r": torch.stack([x.r for x in items], dim=0).to(device),
        }

    def _predict_x0(self, model, x_t, t, atomics, bonds, masks, flag_3ds):
        t = t.view(-1, 1, 1)
        out_coords, _, _, _ = model(
            x_t,
            atomics,
            edge_feats=bonds,
            t=t,
            cond_coords=None,
            cond_atomics=None,
            cond_bonds=None,
            atom_mask=masks,
            flag_3Ds=flag_3ds,
        )
        return out_coords

    def FM_training_step(self, batch):
        reward_mean, reward_max, r_mean = self._collect_rollout_to_buffer(batch)

        sample_items = self.rollout_buffer.sample(self.replay_batch_size)
        if len(sample_items) == 0:
            return torch.zeros([], device=self.device, requires_grad=True)
        replay = self._to_batch(sample_items, self.device)

        x0_hat = replay["x0_hat"]
        cond_atomics = replay["cond_atomics"]
        cond_bonds = replay["cond_bonds"]
        cond_masks = replay["cond_masks"]
        cond_flag_3Ds = replay["cond_flag_3Ds"]
        r = replay["r"]

        t = torch.rand(x0_hat.size(0), device=x0_hat.device, dtype=x0_hat.dtype)
        eps = torch.randn_like(x0_hat)
        alpha_t, sigma_t = self._alpha_sigma(t)
        x_t = alpha_t * x0_hat + sigma_t * eps

        with torch.no_grad():
            x0_old = self._predict_x0(
                self.model_old,
                x_t,
                t,
                atomics=cond_atomics,
                bonds=cond_bonds,
                masks=cond_masks,
                flag_3ds=cond_flag_3Ds,
            )

        x0_theta = self._predict_x0(
            self.gen,
            x_t,
            t,
            atomics=cond_atomics,
            bonds=cond_bonds,
            masks=cond_masks,
            flag_3ds=cond_flag_3Ds,
        )

        x0_pos = (1.0 - self.beta) * x0_old + self.beta * x0_theta
        x0_neg = (1.0 + self.beta) * x0_old - self.beta * x0_theta

        mse_pos = torch.mean((x0_pos - x0_hat) ** 2, dim=(1, 2))
        mse_neg = torch.mean((x0_neg - x0_hat) ** 2, dim=(1, 2))
        raw_loss = r * mse_pos + (1.0 - r) * mse_neg

        residual = torch.mean(torch.abs(x0_theta - x0_hat), dim=(1, 2)).detach()
        weight = 1.0 / (residual + 1e-6)
        loss = torch.mean(raw_loss * weight)

        self.log("train-online-rl-loss", loss, on_step=True, logger=True, sync_dist=True)
        self.log("train-online-rl-reward-mean", reward_mean, on_step=True, logger=True, sync_dist=True)
        self.log("train-online-rl-reward-max", reward_max, on_step=True, logger=True, sync_dist=True)
        self.log("train-online-rl-r-mean", r_mean, on_step=True, logger=True, sync_dist=True)
        self.log("train-online-rl-buffer-size", float(len(self.rollout_buffer)), on_step=True, logger=True, sync_dist=True)
        self.log("train-online-rl-weight-mean", weight.mean(), on_step=True, logger=True, sync_dist=True)

        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure=optimizer_closure)
        self._soft_ema_old_policy_update()

    @torch.no_grad()
    def _soft_ema_old_policy_update(self):
        if self.model_old is None:
            return

        global_step = max(1, int(self.global_step))
        eta_i = min(self.eta_scale * global_step, self.eta_max)

        for theta_old, theta in zip(self.model_old.parameters(), self.gen.parameters()):
            theta_old.data.mul_(eta_i).add_(theta.data, alpha=(1.0 - eta_i))

        for buf_old, buf in zip(self.model_old.buffers(), self.gen.buffers()):
            buf_old.data.mul_(eta_i).add_(buf.data, alpha=(1.0 - eta_i))

        self.log("train-online-rl-eta", eta_i, on_step=True, logger=True, sync_dist=True)

    def validation_step(self, batch, b_idx):
        return

    def on_validation_epoch_end(self):
        return

    def test_step(self, batch, batch_idx):
        return

    def on_test_epoch_end(self):
        return
