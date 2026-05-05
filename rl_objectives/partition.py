from dataclasses import dataclass
import math
import torch
from .desirability import tchebycheff_score
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
    diagnostics: dict[str, torch.Tensor]

def non_dominated_sort(objs, feasible):
    b = objs.shape[0]; rank=torch.full((b,),999,dtype=torch.long,device=objs.device); dom=torch.zeros((b,),dtype=torch.long,device=objs.device)
    idx=torch.where(feasible)[0]
    for i in idx:
        c=0
        for j in idx:
            if i==j: continue
            if torch.all(objs[j]>=objs[i]) and torch.any(objs[j]>objs[i]): c+=1
        dom[i]=c; rank[i]=c
    return rank,dom

class PartitionSelector:
    def __init__(self,cfg): self.cfg=cfg
    def _top_score(self, scoring):
        base=scoring.score.clone(); mode=self.cfg.get("top_selection_score_mode","score")
        names=list(scoring.component_scores.keys())
        comp_stack=torch.stack([scoring.component_scores[k] for k in names],dim=1)
        min_comp=comp_stack.min(dim=1).values
        enable_bal = self.cfg.get("enable_component_balanced_top", False)
        if enable_bal and mode in ("component_balanced","score_plus_min_component"):
            for k,w in self.cfg.get("top_selection_component_weights",{}).items():
                if k in scoring.component_scores: base = base + float(w)*scoring.component_scores[k]
        if self.cfg.get("enable_min_component_bonus", False) and (self.cfg.get("use_min_component_bonus",False) or mode in ("component_balanced","score_plus_min_component")):
            base = base + float(self.cfg.get("min_component_weight",1.0))*min_comp
        if mode=="tchebycheff":
            base = tchebycheff_score(scoring.component_scores, self.cfg.get("top_selection_component_weights"))
        return base
    def select(self, scoring):
        score=scoring.score; b=score.numel(); top_n=max(1,math.ceil(b*self.cfg.get("top_ratio",0.25))); bot_n=max(1,math.ceil(b*self.cfg.get("bottom_ratio",0.25)))
        top=torch.zeros(b,dtype=torch.bool,device=score.device); bottom=torch.zeros_like(top); breasons=["middle"]*b; pareto_rank=dom=None; diag={}
        mode=self.cfg.get("mode","scalar_top_bottom")
        if mode=="scalar_top_bottom":
            sel=torch.topk(score,k=min(top_n,b),largest=True).indices; top[sel]=True
            low=torch.topk(score,k=min(bot_n,b),largest=False).indices; bottom[low]=~top[low]
        else:
            pnames=scoring.metadata.get("pareto_component_names") or list(scoring.component_scores.keys())
            comp=torch.stack([scoring.component_scores[k] for k in pnames if k in scoring.component_scores],dim=1)
            pareto_rank,dom=non_dominated_sort(comp,scoring.feasible)
            top_score=self._top_score(scoring)
            cand=torch.where(scoring.feasible & (pareto_rank<=self.cfg.get("pareto_rank_max",1)))[0].tolist()
            if len(cand)<top_n:
                fidx=torch.where(scoring.feasible)[0]; q=torch.quantile(top_score[fidx], self.cfg.get("top_candidate_quantile",0.7)) if fidx.numel() else torch.tensor(1.0,device=score.device)
                cand += [int(i) for i in fidx.tolist() if top_score[i]>=q and i not in cand]
            cand=sorted(set(cand), key=lambda i: float(top_score[i]), reverse=True)
            picked = select_scaffold_diverse(cand, top_score.detach().cpu(), scoring.scaffolds, top_n) if self.cfg.get("diversity_mode")=="scaffold" else (select_fingerprint_diverse(cand, top_score.detach().cpu(), scoring.fps, pareto_rank.detach().cpu().tolist(), top_n) if self.cfg.get("diversity_mode")=="fingerprint" else cand[:top_n])
            top[picked]=True
            inv=(~scoring.feasible)|(~scoring.valid)|(~scoring.connected); sev=scoring.severe_violation
            lowq=torch.quantile(score,self.cfg.get("low_score_quantile",0.25)); low=score<=lowq
            domm=(pareto_rank>=int(torch.quantile(pareto_rank[scoring.feasible].float(),self.cfg.get("bad_rank_quantile",0.75)).item())) if scoring.feasible.any() else torch.zeros_like(top)
            floor=torch.zeros_like(top)
            if self.cfg.get("enable_component_floor_bottom", False):
                for k,v in self.cfg.get("bottom_component_floor",{}).items():
                    if k in scoring.component_scores:
                        m=scoring.component_scores[k] < float(v); floor |= m
            pmap={"invalid":inv,"severe":sev,"component_floor":floor,"dominated":domm,"low_score":low}
            overlap = top & bottom
            for pname in self.cfg.get("bottom_priority",["invalid","severe","dominated","low_score"]):
                m=pmap.get(pname,torch.zeros_like(top))
                for i in torch.where(m & (~top) & (~bottom))[0].tolist():
                    if int(bottom.sum())<bot_n:
                        bottom[i]=True; breasons[i]="missing_F" if (pname=="component_floor" and "num_F" in self.cfg.get("bottom_component_floor",{})) else pname
            diag["top_candidate_count"]=torch.tensor(float(len(cand)),device=score.device)
            if "num_F" in scoring.raw_properties:
                ci=torch.tensor(cand,dtype=torch.long,device=score.device) if len(cand) else torch.zeros(0,dtype=torch.long,device=score.device)
                if ci.numel()>0:
                    diag["top_candidate_raw_num_F_mean"]=scoring.raw_properties["num_F"].index_select(0,ci).mean()
                    diag["top_candidate_frac_num_F_eq_1"]= (scoring.raw_properties["num_F"].index_select(0,ci)==1).float().mean()
            if "sim_ranolazine_AP" in scoring.raw_properties and len(cand):
                v=scoring.raw_properties["sim_ranolazine_AP"].index_select(0,ci); diag["top_candidate_raw_sim_mean"]=v.mean(); diag["top_candidate_raw_sim_max"]=v.max()
        top_reasons=["top" if bool(top[i]) else "middle" for i in range(b)]
        return PartitionResult(top,bottom,(top|bottom).float(),torch.ones_like(score),torch.ones_like(score),pareto_rank,dom,breasons,top_reasons,diag)

def build_partition_selector(partition_config): return PartitionSelector(partition_config or {})
