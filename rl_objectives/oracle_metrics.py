import argparse, csv, json
from pathlib import Path
import pandas as pd

class OracleLogger:
    def __init__(self, path, task_name, enabled=True, novelty_reference_path=None):
        self.enabled=enabled; self.path=Path(path) if path else None; self.task_name=task_name; self.oracle_call_id=0
        self.novel=set()
        if novelty_reference_path and Path(novelty_reference_path).exists():
            self.novel={x.strip() for x in Path(novelty_reference_path).read_text().splitlines() if x.strip()}
        self._header_written=False
    def log_batch(self, step, source, scoring_result, partition_result=None):
        if not self.enabled or self.path is None: return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows=[]
        b=scoring_result.score.numel()
        for i in range(b):
            self.oracle_call_id += 1
            row={"step":step,"source":source,"task":self.task_name,"oracle_call_id":self.oracle_call_id,"smiles":scoring_result.smiles[i],"canonical_smiles":scoring_result.canonical_smiles[i],"valid":bool(scoring_result.valid[i]),"connected":bool(scoring_result.connected[i]),"score":float(scoring_result.score[i]),"feasible":bool(scoring_result.feasible[i]),"severe_violation":bool(scoring_result.severe_violation[i]),"scaffold":scoring_result.scaffolds[i]}
            for k,v in scoring_result.component_scores.items(): row[f"component_{k}"]=float(v[i])
            for k,v in scoring_result.raw_properties.items(): row[f"raw_{k}"]=float(v[i])
            if partition_result is not None:
                row["bottom_reason"]=partition_result.bottom_reasons[i]; row["top_selected"]=bool(partition_result.top_mask[i]); row["bottom_selected"]=bool(partition_result.bottom_mask[i])
            if self.novel:
                s=row["canonical_smiles"]; row["is_novel"]=(s is not None and s not in self.novel)
            rows.append(row)
        df=pd.DataFrame(rows); mode='a' if self.path.exists() else 'w'; df.to_csv(self.path, mode=mode, header=not self.path.exists(), index=False)

def load_oracle_history(csv_path): return pd.read_csv(csv_path)

def compute_topk(history, k=100):
    s=history["score"].sort_values(ascending=False)
    return {"top1":float(s.iloc[0]) if len(s) else 0.0, "top10_mean":float(s.head(10).mean()) if len(s) else 0.0, "top100_mean":float(s.head(k).mean()) if len(s) else 0.0}

def compute_auc_top10(history, budget_col="oracle_call_id"):
    h=history.sort_values(budget_col); vals=[]
    for t in range(1,len(h)+1): vals.append(h.iloc[:t]["score"].sort_values(ascending=False).head(10).mean())
    return float(sum(vals)/max(len(vals),1))

def compute_validity_uniqueness_novelty(history):
    total=len(history); valid=history[history.valid==True]
    uniq=valid.canonical_smiles.dropna().unique()
    novelty=float(history.get("is_novel", pd.Series([False]*total)).mean()) if "is_novel" in history.columns else None
    return {"validity":len(valid)/max(total,1),"uniqueness":len(uniq)/max(len(valid),1),"novelty":novelty}

def compute_unique_scaffolds(history):
    s=history.scaffold.dropna().unique(); return {"n_unique_scaffolds":len(s),"ratio":len(s)/max(len(history),1)}

def compute_property_distribution(history, prop_names=("SA","logP","TPSA")):
    out={}
    for p in prop_names:
        col=f"raw_{p}" if f"raw_{p}" in history.columns else p
        if col in history.columns:
            v=history[col].dropna(); out[p]={"mean":float(v.mean()),"median":float(v.median()),"q10":float(v.quantile(0.1)),"q90":float(v.quantile(0.9))}
    return out

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True); args=ap.parse_args()
    h=load_oracle_history(args.csv)
    m={"topk":compute_topk(h),"auc_top10":compute_auc_top10(h),"vun":compute_validity_uniqueness_novelty(h),"scaffold":compute_unique_scaffolds(h),"props":compute_property_distribution(h)}
    Path(args.out).write_text(json.dumps(m, indent=2))
