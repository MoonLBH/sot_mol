from __future__ import annotations
import argparse
import csv
import gc
import json
import os
from pathlib import Path

import lightning as L
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem.AtomPairs import Pairs

from sot_mol.comparm import GP, Update_PARAMS
from sot_mol.models.rl_lfpo_interface import MolGen_LFPOModel
from sot_mol.rl_objectives.mpo_tasks import build_objective

RANOLAZINE_SMILES = "COc1ccc2nc(S(N)(=O)=O)sc2c1CCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1"


def parse_conditions(s: str) -> list[int]:
    out = [int(x.strip()) for x in s.split(",") if x.strip()]
    for c in out:
        if c not in (1, 2, 3, 4):
            raise ValueError(f"Unsupported condition id: {c}")
    return out


def write_sdf(mols, rows, path: Path):
    writer = Chem.SDWriter(str(path))
    for mol, row in zip(mols, rows):
        if mol is None:
            continue
        m = Chem.Mol(mol)
        for k in [
            "idx", "canonical_smiles", "raw_sim_ranolazine_AP", "raw_logP", "raw_TPSA", "raw_num_F",
            "cond1_sim_pass", "cond2_logp_pass", "cond3_tpsa_pass", "cond4_num_f_pass", "selected_conditions_pass",
        ]:
            m.SetProp(k, str(row[k]))
        writer.write(m)
    writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="rl.json")
    ap.add_argument("--load_ckpt", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--gen_chunk_size", type=int, default=100)
    ap.add_argument("--score_device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--output_dir", type=str, default="prior_profile/Ranolazine_MPO")
    ap.add_argument("--conditions", type=str, default="1,2,3,4")
    ap.add_argument("--sim_threshold", type=float, default=0.7)
    ap.add_argument("--logp_threshold", type=float, default=7.0)
    ap.add_argument("--tpsa_threshold", type=float, default=95.0)
    ap.add_argument("--num_f_target", type=int, default=1)
    ap.add_argument("--sdf_name", type=str, default="generated_all.sdf")
    ap.add_argument("--matched_sdf_name", type=str, default="generated_matched.sdf")
    ap.add_argument("--csv_name", type=str, default="generated_metrics.csv")
    ap.add_argument("--summary_name", type=str, default="summary.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    gp = Update_PARAMS(GP, str(config_path))
    os.environ["CUDA_VISIBLE_DEVICES"] = gp.CUDA_VISIBLE_DEVICES

    L.seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = MolGen_LFPOModel(
        d_model=gp.D_MODEL, atom_tokens=gp.TOKENS, n_bond_types=gp.N_BOND_TYPES,
        coord_std=gp.COORDS_STD_DEV, scale_ot=gp.SCALE_OT, self_cond=True,
        coord_noise_std=0.2, formulation="endpoint", eval_3D_props=False, ot_bond_weight=1,
        objective_name="Ranolazine_MPO",
    )

    lm = model.create_lightning_module(load_ckpt=args.load_ckpt)
    lm.eval().to(device)

    datasets_dir = Path(__file__).resolve().parent.parent / "datasets"
    model.data_module = model.data_module = __import__("sot_mol.data.datamodule", fromlist=["MGDataModule"]).MGDataModule(
        model.vocab, model.n_bond_types,
        train_datafile=datasets_dir / "train.smol", val_datafile=datasets_dir / "val.smol", test_datafile=datasets_dir / "test.smol",
        max_atoms=model.max_atoms, coord_std=model.coord_std, scale_ot=model.scale_ot,
        scale_ot_factor=0.2, batchsize=args.batch_size, mini_batchsize=1,
        with_Hs=model.with_Hs, ot_geo_weight=model.ot_geo_weight, ot_type_weight=model.ot_type_weight, ot_bond_weight=model.ot_bond_weight,
    )
    objective = build_objective("Ranolazine_MPO", {"aggregate": "geometric"})
    score_device = torch.device(args.score_device if (args.score_device == "cpu" or torch.cuda.is_available()) else "cpu")

    ref = Chem.MolFromSmiles(RANOLAZINE_SMILES)
    ref_ap = Pairs.GetAtomPairFingerprint(ref)
    self_sim = DataStructs.TanimotoSimilarity(ref_ap, ref_ap)
    if abs(float(self_sim) - 1.0) > 1e-6:
        print(f"[WARN] Ranolazine AP self similarity is {self_sim}, expected 1.0")

    selected_conditions = parse_conditions(args.conditions)
    rows = []
    valid_mols = []
    matched_mols = []
    matched_rows = []
    generated_count = 0
    chunk_idx = 0
    while generated_count < args.num_samples:
        chunk_idx += 1
        remaining = args.num_samples - generated_count
        cur_n = min(args.gen_chunk_size, remaining)
        try:
            model.data_module.testset.sample(cur_n)
            with torch.inference_mode():
                cur_mols, _ = model.generate_molecules(lm, model.data_module, model.max_steps, stabilities=False)
            cur_mols = cur_mols[:cur_n]
            scoring = objective.score_mols(cur_mols, device=score_device, dtype=torch.float32)
        except torch.OutOfMemoryError:
            print("[OOM] CUDA out of memory during generation/scoring.")
            print("Please reduce --batch_size or --gen_chunk_size and retry.")
            raise SystemExit(1)

        chunk_valid = 0
        chunk_selected = 0
        for local_i in range(cur_n):
            global_idx = generated_count + local_i
            valid = bool(scoring.valid[local_i].item())
            connected = bool(scoring.connected[local_i].item())
            cs = scoring.canonical_smiles[local_i]
            rs = float(scoring.raw_properties["sim_ranolazine_AP"][local_i].item()) if valid else float("nan")
            lp = float(scoring.raw_properties["logP"][local_i].item()) if valid else float("nan")
            tp = float(scoring.raw_properties["TPSA"][local_i].item()) if valid else float("nan")
            nf = float(scoring.raw_properties["num_F"][local_i].item()) if valid else float("nan")
            if valid:
                chunk_valid += 1
            cond_map = {}
            c1 = valid and (rs >= args.sim_threshold)
            c2 = valid and (lp >= args.logp_threshold)
            c3 = valid and (tp >= args.tpsa_threshold)
            c4 = valid and (nf == float(args.num_f_target))
            cond_map = {1: c1, 2: c2, 3: c3, 4: c4}
            selected_pass = valid and all(cond_map[c] for c in selected_conditions)
            if selected_pass:
                chunk_selected += 1
            row = {
                "idx": global_idx, "valid": valid, "connected": connected,
                "smiles": Chem.MolToSmiles(cur_mols[local_i]) if cur_mols[local_i] is not None else "",
                "canonical_smiles": cs or "",
                "raw_sim_ranolazine_AP": rs, "raw_logP": lp, "raw_TPSA": tp, "raw_num_F": nf,
                "cond1_sim_pass": c1, "cond2_logp_pass": c2, "cond3_tpsa_pass": c3, "cond4_num_f_pass": c4,
                "selected_conditions_pass": selected_pass,
            }
            rows.append(row)
            if valid and cur_mols[local_i] is not None:
                valid_mols.append(cur_mols[local_i])
            if selected_pass and cur_mols[local_i] is not None:
                matched_mols.append(cur_mols[local_i])
                matched_rows.append(row)

        generated_count += cur_n
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            reserv = torch.cuda.memory_reserved() / (1024 ** 2)
            free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / (1024 ** 2)
            print(f"[chunk {chunk_idx}] generated {generated_count} / {args.num_samples}, valid {chunk_valid}, selected {chunk_selected}, cuda allocated/reserved/free(MB)= {alloc:.1f}/{reserv:.1f}/{free:.1f}")
        else:
            print(f"[chunk {chunk_idx}] generated {generated_count} / {args.num_samples}, valid {chunk_valid}, selected {chunk_selected}, cuda n/a")

        del cur_mols, scoring
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    csv_path = outdir / args.csv_name
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_sdf(valid_mols, [r for r in rows if r["valid"]], outdir / args.sdf_name)
    write_sdf(matched_mols, matched_rows, outdir / args.matched_sdf_name)

    valid_rows = [r for r in rows if r["valid"]]
    sim_vals = np.array([r["raw_sim_ranolazine_AP"] for r in valid_rows], dtype=float) if valid_rows else np.array([])
    sim_sorted = np.sort(sim_vals)[::-1] if sim_vals.size else np.array([])

    def top_mean(n):
        if sim_sorted.size == 0:
            return 0.0
        return float(sim_sorted[: min(n, sim_sorted.size)].mean())

    condition_counts = {
        "1": int(sum(r["cond1_sim_pass"] for r in rows)),
        "2": int(sum(r["cond2_logp_pass"] for r in rows)),
        "3": int(sum(r["cond3_tpsa_pass"] for r in rows)),
        "4": int(sum(r["cond4_num_f_pass"] for r in rows)),
    }
    num_valid = len(valid_rows)
    summary = {
        "num_samples": args.num_samples,
        "num_valid": num_valid,
        "validity": float(num_valid / max(1, args.num_samples)),
        "num_connected": int(sum(r["connected"] for r in rows)),
        "connected_validity": float(sum(r["connected"] for r in rows) / max(1, num_valid)),
        "thresholds": {
            "sim_threshold": args.sim_threshold,
            "logp_threshold": args.logp_threshold,
            "tpsa_threshold": args.tpsa_threshold,
            "num_f_target": args.num_f_target,
        },
        "selected_conditions": selected_conditions,
        "condition_counts": condition_counts,
        "condition_fracs_valid": {k: float(v / max(1, num_valid)) for k, v in condition_counts.items()},
        "selected_conditions_count": int(sum(r["selected_conditions_pass"] for r in rows)),
        "selected_conditions_frac_valid": float(sum(r["selected_conditions_pass"] for r in rows) / max(1, num_valid)),
        "similarity_stats": {
            "mean": float(sim_vals.mean()) if sim_vals.size else 0.0,
            "max": float(sim_vals.max()) if sim_vals.size else 0.0,
            "top10_mean": top_mean(10),
            "top100_mean": top_mean(100),
            "q90": float(np.quantile(sim_vals, 0.9)) if sim_vals.size else 0.0,
            "q99": float(np.quantile(sim_vals, 0.99)) if sim_vals.size else 0.0,
        },
    }

    summary_path = outdir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Generated: {args.num_samples}")
    print(f"Valid: {num_valid} / {args.num_samples}")
    print(f"Connected: {summary['num_connected']} / {max(1, num_valid)}")
    print(f"Condition 1 sim >= threshold: {condition_counts['1']}")
    print(f"Condition 2 logP >= threshold: {condition_counts['2']}")
    print(f"Condition 3 TPSA >= threshold: {condition_counts['3']}")
    print(f"Condition 4 num_F == target: {condition_counts['4']}")
    print(f"Selected conditions {selected_conditions}: {summary['selected_conditions_count']}")
    print("Similarity mean/max/top10/top100:", summary["similarity_stats"]["mean"], summary["similarity_stats"]["max"], summary["similarity_stats"]["top10_mean"], summary["similarity_stats"]["top100_mean"])
    print("Saved:")
    print(f"- all SDF: {outdir / args.sdf_name}")
    print(f"- matched SDF: {outdir / args.matched_sdf_name}")
    print(f"- CSV: {csv_path}")
    print(f"- summary: {summary_path}")


if __name__ == "__main__":
    main()
