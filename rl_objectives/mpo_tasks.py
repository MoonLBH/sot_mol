from dataclasses import dataclass
from typing import Any
import warnings
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, Crippen, rdMolDescriptors, AllChem
from rdkit.Chem.AtomPairs import Pairs
from .desirability import gaussian, max_gaussian, min_gaussian, thresholded, weighted_geometric_mean, weighted_linear_sum, tchebycheff_score
from .diversity import canonical_smiles, murcko_scaffold_smiles, morgan_fp

@dataclass
class ScoringResult:
    score: torch.Tensor
    component_scores: dict[str, torch.Tensor]
    raw_properties: dict[str, torch.Tensor]
    feasible: torch.BoolTensor
    severe_violation: torch.BoolTensor
    valid: torch.BoolTensor
    connected: torch.BoolTensor
    smiles: list[str | None]
    canonical_smiles: list[str | None]
    scaffolds: list[str | None]
    fps: list[Any]
    mols: list[Any]
    metadata: dict

class BaseObjective:
    name = "base"
    component_names = []
    pareto_component_names = []
    def __init__(self, cfg=None): self.cfg = cfg or {}
    def score_mols(self, mols, device=None, dtype=torch.float32): raise NotImplementedError

class QEDObjective(BaseObjective):
    name = "qed"; component_names=["qed"]
    def score_mols(self,mols,device=None,dtype=torch.float32):
        with torch.no_grad():
            vals=[]; valid=[]; conn=[]; cs=[]; sc=[]; fps=[]
            for m in mols:
                v=m is not None; c=v and len(Chem.GetMolFrags(m))==1
                vals.append(QED.qed(m) if v else 0.0); valid.append(v); conn.append(c)
                cs.append(canonical_smiles(m) if v else None); sc.append(murcko_scaffold_smiles(m) if v else None); fps.append(morgan_fp(m) if v else None)
            s=torch.tensor(vals,dtype=dtype,device=device); vb=torch.tensor(valid,dtype=torch.bool,device=device); cb=torch.tensor(conn,dtype=torch.bool,device=device)
            return ScoringResult(s,{"qed":s},{"QED":s},vb&cb,(~vb)|(~cb),vb,cb,cs,cs,sc,fps,mols,{"official_guacamol":False})

class MPOObjective(BaseObjective):
    _warned=False
    def __init__(self,name,target_smiles,components,cfg=None):
        super().__init__(cfg); self.name=name; self.component_defs=components; self.component_names=[c[0] for c in components]
        self.pareto_component_names=self.cfg.get("pareto_component_names", self.component_names)
        self.target = Chem.MolFromSmiles(target_smiles)
        self.target_ap=Pairs.GetAtomPairFingerprint(self.target)
        self.ref_canonical=canonical_smiles(self.target)
        self.ap_self=DataStructs.TanimotoSimilarity(self.target_ap, self.target_ap)
    def _official_scores(self, smiles):
        if not self.cfg.get("use_official_guacamol", True): return None
        try:
            import guacamol.standard_benchmarks as sb
            fn = getattr(sb, self.name.lower(), None)
            if fn is None: return None
            obj = fn().objective
            if hasattr(obj, "score_list"):
                out = obj.score_list(smiles)
                return out if isinstance(out, list) else list(out)
            return [obj.score(s) for s in smiles]
        except Exception:
            if not MPOObjective._warned:
                warnings.warn("Using local fallback scorer (official GuacaMol unavailable/API mismatch).")
                MPOObjective._warned=True
            return None
    def score_mols(self,mols,device=None,dtype=torch.float32):
        with torch.no_grad():
            valid=[]; conn=[]; cs=[]; scaf=[]; fps=[]
            raws={"sim_ranolazine_AP":[],"logP":[],"TPSA":[],"num_F":[]}
            comps={k:[] for k in self.component_names}
            for m in mols:
                v=m is not None; c=v and len(Chem.GetMolFrags(m))==1; valid.append(v); conn.append(c)
                cs.append(canonical_smiles(m) if v else None); scaf.append(murcko_scaffold_smiles(m) if v else None); fps.append(morgan_fp(m) if v else None)
                if not v:
                    for k in raws: raws[k].append(0.0)
                    for k in comps: comps[k].append(0.0)
                    continue
                ap = Pairs.GetAtomPairFingerprint(m)
                sim_ap = DataStructs.TanimotoSimilarity(ap, self.target_ap)
                logp = Crippen.MolLogP(m); tpsa = rdMolDescriptors.CalcTPSA(m); nF = float(sum(1 for a in m.GetAtoms() if a.GetSymbol()=="F"))
                raws["sim_ranolazine_AP"].append(sim_ap); raws["logP"].append(logp); raws["TPSA"].append(tpsa); raws["num_F"].append(nF)
                local={"sim_AP":sim_ap,"logP":logp,"TPSA":tpsa,"num_F":nF,"formula":rdMolDescriptors.CalcMolFormula(m)}
                for n, fn in self.component_defs: comps[n].append(float(fn(local)))
            comp_t={k:torch.tensor(v,dtype=dtype,device=device).clamp(0,1) for k,v in comps.items()}
            raw_t={k:torch.tensor(v,dtype=dtype,device=device) for k,v in raws.items()}
            geo=weighted_geometric_mean(comp_t,self.cfg.get("component_weights"))
            lin=weighted_linear_sum(comp_t,self.cfg.get("component_weights"))
            tch=tchebycheff_score(comp_t,self.cfg.get("component_weights"))
            agg=self.cfg.get("aggregate","geometric")
            score={"geometric":geo,"linear":lin,"tchebycheff":tch}.get(agg,geo)
            official=self._official_scores([s or "" for s in cs]); off=False
            if agg=="official" and official is not None:
                score=torch.tensor(official,dtype=dtype,device=device).clamp(0,1); off=True
            vb=torch.tensor(valid,dtype=torch.bool,device=device); cb=torch.tensor(conn,dtype=torch.bool,device=device)
            metadata={"official_guacamol":off,"using_local_fallback":not off,"reference_canonical_smiles":self.ref_canonical,"ranolazine_AP_self_similarity":float(self.ap_self),"official_score":torch.tensor(official,dtype=dtype,device=device).clamp(0,1) if official is not None else None,"geometric_score":geo,"linear_score":lin,"tchebycheff_score":tch,"min_component_score":torch.stack([comp_t[k] for k in self.pareto_component_names if k in comp_t],dim=1).min(dim=1).values}
            return ScoringResult(score,comp_t,raw_t,vb&cb,(~vb)|(~cb),vb,cb,cs,cs,scaf,fps,mols,metadata)

def build_objective(name, objective_config=None):
    cfg=objective_config or {}; n=(name or "qed").lower()
    if n=="qed": return QEDObjective(cfg)
    tasks={"ranolazine_mpo":("Ranolazine_MPO","COc1ccc2nc(S(N)(=O)=O)sc2c1CCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1",[("sim_ranolazine_AP",lambda p: thresholded(torch.tensor(p["sim_AP"]),0.7)),("logP",lambda p:max_gaussian(torch.tensor(p["logP"]),7,1)),("TPSA",lambda p:max_gaussian(torch.tensor(p["TPSA"]),95,20)),("num_F",lambda p:gaussian(torch.tensor(p["num_F"]),1,1))]),
"osimertinib_mpo":("Osimertinib_MPO","COc1cc(Nc2ncnc3cc(OCCCN4CCOCC4)c(OC)c23)ccc1N(C)C",[("sim_osimertinib_FCFC4",lambda p: thresholded(torch.tensor(p.get("sim_ECFC4",0.0)),0.8)),("sim_osimertinib_ECFC6",lambda p:min_gaussian(torch.tensor(p.get("sim_ECFC6",0.0)),0.85,2)),("TPSA",lambda p:max_gaussian(torch.tensor(p["TPSA"]),100,2)),("logP",lambda p:min_gaussian(torch.tensor(p["logP"]),1,2))])}
    if n in tasks: t=tasks[n]; return MPOObjective(t[0],t[1],t[2],cfg)
    raise ValueError(f"Unknown objective: {name}")
