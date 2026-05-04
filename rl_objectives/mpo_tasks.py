from dataclasses import dataclass
from typing import Any
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
    name="base"
    component_names=[]
    pareto_component_names=[]
    def __init__(self, cfg=None): self.cfg = cfg or {}

    def score_mols(self, mols, device=None, dtype=torch.float32): raise NotImplementedError

class QEDObjective(BaseObjective):
    name="qed"; component_names=["qed"]
    def score_mols(self,mols,device=None,dtype=torch.float32):
        with torch.no_grad():
            vals=[]; valid=[]; connected=[]; cs=[]; scaf=[]; fps=[]
            for m in mols:
                v = m is not None; valid.append(v)
                c = v and len(Chem.GetMolFrags(m))==1; connected.append(c)
                q = QED.qed(m) if v else 0.0; vals.append(q)
                cs.append(canonical_smiles(m) if v else None); scaf.append(murcko_scaffold_smiles(m) if v else None); fps.append(morgan_fp(m) if v else None)
            score = torch.tensor(vals, dtype=dtype, device=device)
            vb = torch.tensor(valid, dtype=torch.bool, device=device); cb=torch.tensor(connected,dtype=torch.bool,device=device)
            return ScoringResult(score,{"qed":score}, {"QED":score}, vb&cb, (~vb)|(~cb), vb, cb, cs, cs, scaf, fps, mols, {"official_guacamol":False})

class MPOObjective(BaseObjective):
    def __init__(self,name,target_smiles,components,cfg=None):
        super().__init__(cfg); self.name=name; self.component_defs=components; self.component_names=[c[0] for c in components]
        self.pareto_component_names=self.cfg.get("pareto_component_names", self.component_names)
        tm=Chem.MolFromSmiles(target_smiles); self.target_ecfp4=AllChem.GetMorganFingerprintAsBitVect(tm,2,2048); self.target_ecfp6=AllChem.GetMorganFingerprintAsBitVect(tm,3,2048); self.target_ap=Pairs.GetAtomPairFingerprint(tm)
    def _official_scores(self, smiles):
        if not self.cfg.get("use_official_guacamol", True): return None
        try:
            import guacamol.standard_benchmarks as sb
            fn = getattr(sb, self.name.lower()) if hasattr(sb, self.name.lower()) else None
            if fn is None: return None
            bm = fn(); obj = bm.objective
            if hasattr(obj, "score_list"): return [obj.score_list(smiles)][0] if isinstance(obj.score_list(smiles), list) else obj.score_list(smiles)
            return [obj.score(s) for s in smiles]
        except Exception:
            return None
    def score_mols(self,mols,device=None,dtype=torch.float32):
        with torch.no_grad():
            valid=[]; conn=[]; cs=[]; scaf=[]; fps=[]; raws={"logP":[],"TPSA":[],"num_F":[]}; comps={k:[] for k in self.component_names}
            for m in mols:
                v = m is not None; valid.append(v); c = v and len(Chem.GetMolFrags(m))==1; conn.append(c)
                cs.append(canonical_smiles(m) if v else None); scaf.append(murcko_scaffold_smiles(m) if v else None); fps.append(morgan_fp(m) if v else None)
                if not v:
                    raws["logP"].append(0.0); raws["TPSA"].append(0.0); raws["num_F"].append(0.0)
                    for k in comps: comps[k].append(0.0)
                    continue
                logp=Crippen.MolLogP(m); tpsa=rdMolDescriptors.CalcTPSA(m); nF=sum(1 for a in m.GetAtoms() if a.GetSymbol()=="F")
                raws["logP"].append(logp); raws["TPSA"].append(tpsa); raws["num_F"].append(float(nF))
                ecfp4=AllChem.GetMorganFingerprintAsBitVect(m,2,2048); ecfp6=AllChem.GetMorganFingerprintAsBitVect(m,3,2048); ap=Pairs.GetAtomPairFingerprint(m)
                local={"sim_AP":DataStructs.TanimotoSimilarity(ap,self.target_ap),"sim_ECFC4":DataStructs.TanimotoSimilarity(ecfp4,self.target_ecfp4),"sim_ECFC6":DataStructs.TanimotoSimilarity(ecfp6,self.target_ecfp6),"logP":logp,"TPSA":tpsa,"num_F":float(nF),"formula":rdMolDescriptors.CalcMolFormula(m)}
                for n,fn in self.component_defs: comps[n].append(float(fn(local)))
            comp_t={k:torch.tensor(v,dtype=dtype,device=device).clamp(0,1) for k,v in comps.items()}
            raw_t={k:torch.tensor(v,dtype=dtype,device=device) for k,v in raws.items()}
            agg=self.cfg.get("aggregate","geometric")
            if agg=="linear": score=weighted_linear_sum(comp_t, self.cfg.get("component_weights"))
            elif agg=="tchebycheff": score=tchebycheff_score(comp_t, self.cfg.get("component_weights"))
            else: score=weighted_geometric_mean(comp_t, self.cfg.get("component_weights"))
            official=self._official_scores([s or "" for s in cs])
            if self.cfg.get("aggregate") == "official" and official is not None:
                score=torch.tensor(official,dtype=dtype,device=device).clamp(0,1); off=True
            else: off=False
            vb=torch.tensor(valid,dtype=torch.bool,device=device); cb=torch.tensor(conn,dtype=torch.bool,device=device)
            return ScoringResult(score, comp_t, raw_t, vb&cb, (~vb)|(~cb), vb, cb, cs, cs, scaf, fps, mols, {"official_guacamol":off})

def build_objective(name, objective_config=None):
    cfg=objective_config or {}
    n=(name or "qed").lower()
    if n=="qed": return QEDObjective(cfg)
    tasks={
        "ranolazine_mpo":("Ranolazine_MPO","COc1ccc2nc(S(N)(=O)=O)sc2c1CCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1",[("sim_ranolazine_AP",lambda p: thresholded(torch.tensor(p["sim_AP"]),0.7)),("logP",lambda p:max_gaussian(torch.tensor(p["logP"]),7,1)),("TPSA",lambda p:max_gaussian(torch.tensor(p["TPSA"]),95,20)),("num_F",lambda p:gaussian(torch.tensor(p["num_F"]),1,1))]),
        "osimertinib_mpo":("Osimertinib_MPO","COc1cc(Nc2ncnc3cc(OCCCN4CCOCC4)c(OC)c23)ccc1N(C)C",[("sim_osimertinib_FCFC4",lambda p: thresholded(torch.tensor(p["sim_ECFC4"]),0.8)),("sim_osimertinib_ECFC6",lambda p:min_gaussian(torch.tensor(p["sim_ECFC6"]),0.85,2)),("TPSA",lambda p:max_gaussian(torch.tensor(p["TPSA"]),100,2)),("logP",lambda p:min_gaussian(torch.tensor(p["logP"]),1,2))]),
        "fexofenadine_mpo":("Fexofenadine_MPO","CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1",[("sim_fexofenadine_AP",lambda p:thresholded(torch.tensor(p["sim_AP"]),0.8)),("TPSA",lambda p:max_gaussian(torch.tensor(p["TPSA"]),90,2)),("logP",lambda p:min_gaussian(torch.tensor(p["logP"]),4,2))]),
        "sitagliptin_mpo":("Sitagliptin_MPO","N[C@@H](CC1=CC=CC=C1)C(=O)N2CCN(CC2)C(=O)C(F)(F)F",[("sim_sitagliptin_ECFC4",lambda p:gaussian(torch.tensor(p["sim_ECFC4"]),0,0.1)),("logP",lambda p:gaussian(torch.tensor(p["logP"]),2.0165,0.2)),("TPSA",lambda p:gaussian(torch.tensor(p["TPSA"]),77.04,5)),("isomer_C16H15F6N5O",lambda p:torch.tensor(1.0 if p["formula"]=="C16H15F6N5O" else 0.0))]),
    }
    if n in tasks:
        t=tasks[n]; return MPOObjective(t[0],t[1],t[2],cfg)
    raise ValueError(f"Unknown objective: {name}")
