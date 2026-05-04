from dataclasses import dataclass
import math
import torch
from .diversity import select_fingerprint_diverse, select_scaffold_diverse


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
    diagnostics: dict


def non_dominated_sort(objs, feasible):
    b = objs.shape[0]
    rank = torch.full((b,), 999, dtype=torch.long, device=objs.device)
    dom_count = torch.zeros((b,), dtype=torch.long, device=objs.device)
    idx = torch.where(feasible)[0]
    for i in idx:
        c = 0
        for j in idx:
            if i == j: continue
            ge = torch.all(objs[j] >= objs[i])
            gt = torch.any(objs[j] > objs[i])
            if ge and gt: c += 1
        dom_count[i] = c
        rank[i] = c
    return rank, dom_count


class PartitionSelector:
    def __init__(self, cfg): self.cfg = cfg
    def select(self, scoring):
        score = scoring.score
        b = score.numel()
        top_n = max(1, math.ceil(b * self.cfg.get("top_ratio", 0.25)))
        bot_n = max(1, math.ceil(b * self.cfg.get("bottom_ratio", 0.25)))
        mode = self.cfg.get("mode", "scalar_top_bottom")
        top = torch.zeros(b, dtype=torch.bool, device=score.device)
        bottom = torch.zeros_like(top)
        reasons = ["middle"] * b
        pareto_rank = dominated = None
        if mode == "scalar_top_bottom":
            top[torch.topk(score, k=min(top_n,b), largest=True).indices] = True
            cand = torch.topk(score, k=min(bot_n,b), largest=False).indices
            bottom[cand] = ~top[cand]
        else:
            comp = torch.stack([v for _, v in scoring.component_scores.items()], dim=1)
            pareto_rank, dominated = non_dominated_sort(comp, scoring.feasible)
            candidates = torch.where(scoring.feasible & (pareto_rank <= self.cfg.get("pareto_rank_max", 1)))[0].tolist()
            if len(candidates) < top_n:
                feasible_idx = torch.where(scoring.feasible)[0]
                q = torch.quantile(score[feasible_idx], self.cfg.get("top_candidate_quantile", 0.7)) if feasible_idx.numel() else torch.tensor(1.0, device=score.device)
                fill = [int(i) for i in feasible_idx.tolist() if score[i] >= q and i not in candidates]
                candidates.extend(fill)
            candidates = sorted(set(candidates), key=lambda i: float(score[i]), reverse=True)
            if self.cfg.get("diversity_mode", "none") == "scaffold":
                picked = select_scaffold_diverse(candidates, score.detach().cpu(), scoring.scaffolds, top_n)
            elif self.cfg.get("diversity_mode", "none") == "fingerprint":
                picked = select_fingerprint_diverse(candidates, score.detach().cpu(), scoring.fps, pareto_rank.detach().cpu().tolist(), top_n)
            else:
                picked = candidates[:top_n]
            top[picked] = True
            inv = (~scoring.feasible) | (~scoring.valid) | (~scoring.connected)
            sev = scoring.severe_violation
            lowq = torch.quantile(score, self.cfg.get("low_score_quantile", 0.25))
            dom = (pareto_rank >= int(torch.quantile(pareto_rank[scoring.feasible].float(), self.cfg.get("bad_rank_quantile", 0.75)).item())) if scoring.feasible.any() else torch.zeros_like(top)
            priorities = [("invalid", inv), ("severe", sev), ("dominated", dom), ("low_score", score <= lowq)]
            for name, m in priorities:
                for i in torch.where(m & (~top) & (~bottom))[0].tolist():
                    if bottom.sum() < bot_n:
                        bottom[i] = True
                        reasons[i] = name
        top_reasons = ["top" if bool(top[i]) else "middle" for i in range(b)]
        selected = (top | bottom).float()
        return PartitionResult(top, bottom, selected, torch.ones_like(score), torch.ones_like(score), pareto_rank, dominated, reasons, top_reasons, {})


def build_partition_selector(partition_config):
    return PartitionSelector(partition_config or {})
