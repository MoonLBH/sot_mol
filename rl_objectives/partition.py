from dataclasses import dataclass
import math
import warnings
import torch

from .desirability import tchebycheff_score
from .diversity import crowding_distance, select_fingerprint_diverse, select_scaffold_diverse


@dataclass
class PartitionResult:
    top_mask: torch.BoolTensor
    bottom_mask: torch.BoolTensor
    selected_mask: torch.Tensor
    top_weights: torch.Tensor
    bottom_weights: torch.Tensor
    pareto_rank: torch.LongTensor | None
    dominated_count: torch.LongTensor | None
    bottom_reasons: list[str]
    top_reasons: list[str]
    diagnostics: dict[str, torch.Tensor]


def _dominates(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.all(a >= b) and torch.any(a > b))


def non_dominated_sort(objs: torch.Tensor, feasible: torch.BoolTensor):
    b = objs.shape[0]
    device = objs.device
    pareto_front_rank = torch.full((b,), 999, dtype=torch.long, device=device)
    dominated_count = torch.zeros((b,), dtype=torch.long, device=device)

    feasible_idx = torch.where(feasible)[0]
    if feasible_idx.numel() == 0:
        return pareto_front_rank, dominated_count

    # Dominated count against feasible samples only.
    for i in feasible_idx.tolist():
        c = 0
        for j in feasible_idx.tolist():
            if i == j:
                continue
            if _dominates(objs[j], objs[i]):
                c += 1
        dominated_count[i] = c

    remaining = feasible_idx.tolist()
    front_rank = 0
    while remaining:
        front = []
        for i in remaining:
            dominated_by_remaining = False
            for j in remaining:
                if i == j:
                    continue
                if _dominates(objs[j], objs[i]):
                    dominated_by_remaining = True
                    break
            if not dominated_by_remaining:
                front.append(i)
        if not front:
            # Safety fallback for numerical/pathological cases.
            front = list(remaining)
        pareto_front_rank[torch.tensor(front, device=device)] = front_rank
        remaining = [i for i in remaining if i not in set(front)]
        front_rank += 1

    return pareto_front_rank, dominated_count


class PartitionSelector:
    def __init__(self, cfg):
        self.cfg = cfg or {}

    def _top_score(self, scoring):
        base = scoring.score.clone()
        mode = self.cfg.get("top_selection_score_mode", "score")
        if mode == "score":
            return base

        comp_names = list(scoring.component_scores.keys())
        if not comp_names:
            return base

        comp_stack = torch.stack([scoring.component_scores[k] for k in comp_names], dim=1)
        min_comp = comp_stack.min(dim=1).values

        if mode in ("component_balanced", "score_plus_min_component"):
            for k, w in self.cfg.get("top_selection_component_weights", {}).items():
                if k in scoring.component_scores:
                    base = base + float(w) * scoring.component_scores[k]

        if self.cfg.get("use_min_component_bonus", False) or mode in ("component_balanced", "score_plus_min_component"):
            base = base + float(self.cfg.get("min_component_weight", 1.0)) * min_comp

        if mode == "tchebycheff":
            base = tchebycheff_score(scoring.component_scores, self.cfg.get("top_selection_component_weights"))

        return base

    def _select_bottom_by_priority(self, top_mask, score, bottom_candidates, priority, bottom_n):
        b = score.numel()
        bottom_mask = torch.zeros_like(top_mask)
        reasons = ["middle"] * b
        if bottom_n <= 0:
            return bottom_mask, reasons

        for name in priority:
            if int(bottom_mask.sum().item()) >= bottom_n:
                break
            cand_mask = bottom_candidates.get(name, torch.zeros_like(top_mask)) & (~top_mask) & (~bottom_mask)
            idx = torch.where(cand_mask)[0]
            if idx.numel() == 0:
                continue
            order = idx[torch.argsort(score.index_select(0, idx), descending=False)]
            take = min(bottom_n - int(bottom_mask.sum().item()), order.numel())
            picked = order[:take]
            bottom_mask[picked] = True
            for i in picked.tolist():
                reasons[i] = name

        if int(bottom_mask.sum().item()) < bottom_n:
            remaining = torch.where((~top_mask) & (~bottom_mask))[0]
            if remaining.numel() > 0:
                order = remaining[torch.argsort(score.index_select(0, remaining), descending=False)]
                take = min(bottom_n - int(bottom_mask.sum().item()), order.numel())
                picked = order[:take]
                bottom_mask[picked] = True
                for i in picked.tolist():
                    reasons[i] = "middle"

        return bottom_mask, reasons

    def _build_pareto_objectives(self, scoring):
        pnames = (
            self.cfg.get("pareto_component_names")
            or scoring.metadata.get("pareto_component_names")
            or list(scoring.component_scores.keys())
        )
        if not pnames:
            warnings.warn("pareto_component_names is empty; fallback to feasible_score")
            return None, []

        used = []
        cols = []
        for pname in pnames:
            if pname not in scoring.component_scores:
                warnings.warn(f"pareto component '{pname}' missing in component_scores; skipping")
                continue
            used.append(pname)
            cols.append(scoring.component_scores[pname].clamp(0.0, 1.0))

        if not cols:
            warnings.warn("No valid pareto components; fallback to feasible_score")
            return None, []
        return torch.stack(cols, dim=1), used

    def select(self, scoring):
        score = scoring.score
        b = score.numel()
        top_n = min(b, int(math.ceil(b * float(self.cfg.get("top_ratio", 0.25)))))
        bottom_n = min(b, int(math.ceil(b * float(self.cfg.get("bottom_ratio", 0.25)))))

        top_mask = torch.zeros(b, dtype=torch.bool, device=score.device)
        pareto_rank = None
        dominated_count = None
        top_reasons = ["middle"] * b

        mode = self.cfg.get("mode", "scalar_top_bottom")
        top_score = self._top_score(scoring)

        top_candidate_mask = torch.zeros_like(top_mask)
        top_candidates = []

        # ---- Top selection by mode ----
        if mode == "scalar_top_bottom":
            order = torch.argsort(top_score, descending=True)
            top_pick = order[:top_n]
            top_mask[top_pick] = True
            top_candidate_mask = torch.ones_like(top_mask)
            top_candidates = order.tolist()
        elif mode in ("feasible_score", "feasible_pareto", "feasible_pareto_diverse"):
            feasible = scoring.feasible
            top_candidate_mask = feasible.clone()
            if mode in ("feasible_pareto", "feasible_pareto_diverse"):
                objs, _ = self._build_pareto_objectives(scoring)
                if objs is not None:
                    pareto_rank, dominated_count = non_dominated_sort(objs, feasible)
                    rank_max = int(self.cfg.get("pareto_rank_max", 1))
                    top_candidate_mask = feasible & (pareto_rank <= rank_max)
                    if int(top_candidate_mask.sum().item()) < top_n:
                        fidx = torch.where(feasible)[0]
                        if fidx.numel() > 0:
                            q = float(self.cfg.get("top_candidate_quantile", 0.7))
                            thresh = torch.quantile(top_score.index_select(0, fidx), q)
                            top_candidate_mask = top_candidate_mask | (feasible & (top_score >= thresh))
                else:
                    mode = "feasible_score"

            top_candidates = torch.where(top_candidate_mask)[0].tolist()
            if mode == "feasible_pareto_diverse":
                diversity_mode = str(self.cfg.get("diversity_mode", "none")).lower()
            else:
                diversity_mode = "none"

            if diversity_mode == "scaffold":
                picked = select_scaffold_diverse(top_candidates, top_score.detach().cpu(), scoring.scaffolds, top_n)
            elif diversity_mode == "fingerprint":
                rank_for_div = pareto_rank.detach().cpu().tolist() if pareto_rank is not None else [999] * b
                picked = select_fingerprint_diverse(top_candidates, top_score.detach().cpu(), scoring.fps, rank_for_div, top_n)
            elif diversity_mode == "crowding" and pareto_rank is not None:
                picked = []
                remaining = set(top_candidates)
                objs_np = objs.detach().cpu().numpy()
                while remaining and len(picked) < top_n:
                    min_rank = min(int(pareto_rank[i].item()) for i in remaining)
                    front = [i for i in remaining if int(pareto_rank[i].item()) == min_rank]
                    cdist = crowding_distance(objs_np, front)
                    front_sorted = sorted(front, key=lambda i: (-float(cdist.get(i, 0.0)), -float(top_score[i])))
                    for i in front_sorted:
                        picked.append(i)
                        remaining.remove(i)
                        if len(picked) >= top_n:
                            break
            else:
                picked = sorted(top_candidates, key=lambda i: float(top_score[i]), reverse=True)[:top_n]
            if picked:
                top_mask[torch.tensor(picked, dtype=torch.long, device=score.device)] = True
        else:
            raise ValueError(f"Unknown partition mode: {mode}")

        # ---- Bottom selection ----
        valid = scoring.valid
        connected = scoring.connected
        feasible = scoring.feasible
        severe = scoring.severe_violation
        low_q = torch.quantile(score, float(self.cfg.get("low_score_quantile", 0.25)))
        low_score = score <= low_q

        comp_floor = torch.zeros_like(top_mask)
        floor_cfg = self.cfg.get("bottom_component_floor", {}) or {}
        for k, v in floor_cfg.items():
            if k in scoring.component_scores:
                comp_floor = comp_floor | (scoring.component_scores[k] < float(v))

        dominated = torch.zeros_like(top_mask)
        if pareto_rank is not None:
            bad_front_rank_threshold = self.cfg.get("bad_front_rank_threshold")
            if bad_front_rank_threshold is None and feasible.any():
                fq = float(self.cfg.get("bad_rank_quantile", 0.75))
                rank_vals = pareto_rank[feasible].float()
                bad_front_rank_threshold = int(torch.quantile(rank_vals, fq).item())
            if bad_front_rank_threshold is not None:
                dominated = feasible & (pareto_rank >= int(bad_front_rank_threshold))

        invalid = (~feasible) | (~valid) | (~connected)
        bottom_candidates = {
            "invalid": invalid,
            "severe": severe,
            "component_floor": comp_floor,
            "dominated": dominated,
            "low_score": low_score,
        }

        if mode == "scalar_top_bottom":
            priority = ["low_score"]
        elif mode == "feasible_score":
            priority = ["invalid", "severe", "low_score"]
        else:
            priority = self.cfg.get("bottom_priority", ["invalid", "severe", "dominated", "low_score"])

        bottom_mask, bottom_reasons = self._select_bottom_by_priority(
            top_mask=top_mask,
            score=score,
            bottom_candidates=bottom_candidates,
            priority=priority,
            bottom_n=bottom_n,
        )

        selected_mask = (top_mask | bottom_mask).float()

        # top reasons
        for i in torch.where(top_mask)[0].tolist():
            top_reasons[i] = "top"

        diagnostics = {
            "top_count": torch.tensor(float(top_mask.sum().item()), device=score.device),
            "bottom_count": torch.tensor(float(bottom_mask.sum().item()), device=score.device),
            "selected_count": torch.tensor(float(selected_mask.sum().item()), device=score.device),
            "top_frac": top_mask.float().mean(),
            "bottom_frac": bottom_mask.float().mean(),
            "selected_frac": selected_mask.mean(),
            "top_candidate_count": torch.tensor(float(len(top_candidates)), device=score.device),
        }

        if pareto_rank is not None:
            diagnostics["pareto_front_rank_mean"] = pareto_rank[feasible].float().mean() if feasible.any() else torch.tensor(999.0, device=score.device)
            diagnostics["dominated_count_mean"] = dominated_count[feasible].float().mean() if feasible.any() else torch.tensor(0.0, device=score.device)
            diagnostics["front0_count"] = torch.tensor(float(((pareto_rank == 0) & feasible).sum().item()), device=score.device)
            diagnostics["front1_count"] = torch.tensor(float(((pareto_rank == 1) & feasible).sum().item()), device=score.device)

        if "num_F" in scoring.raw_properties and len(top_candidates) > 0:
            ci = torch.tensor(top_candidates, dtype=torch.long, device=score.device)
            num_f = scoring.raw_properties["num_F"].index_select(0, ci)
            diagnostics["top_candidate_raw_num_F_mean"] = num_f.mean()
            diagnostics["top_candidate_frac_num_F_eq_1"] = (num_f == 1).float().mean()

        if "sim_ranolazine_AP" in scoring.raw_properties and len(top_candidates) > 0:
            ci = torch.tensor(top_candidates, dtype=torch.long, device=score.device)
            sim = scoring.raw_properties["sim_ranolazine_AP"].index_select(0, ci)
            diagnostics["top_candidate_raw_sim_mean"] = sim.mean()
            diagnostics["top_candidate_raw_sim_max"] = sim.max()

        reason_keys = ["invalid", "severe", "component_floor", "dominated", "low_score"]
        for rk in reason_keys:
            cnt = sum(1 for r in bottom_reasons if r == rk)
            diagnostics[f"bottom_reason_{rk}_frac"] = torch.tensor(float(cnt) / max(1, b), device=score.device)

        return PartitionResult(
            top_mask=top_mask,
            bottom_mask=bottom_mask,
            selected_mask=selected_mask,
            top_weights=torch.ones_like(score),
            bottom_weights=torch.ones_like(score),
            pareto_rank=pareto_rank,
            dominated_count=dominated_count,
            bottom_reasons=bottom_reasons,
            top_reasons=top_reasons,
            diagnostics=diagnostics,
        )


def build_partition_selector(partition_config):
    return PartitionSelector(partition_config or {})
