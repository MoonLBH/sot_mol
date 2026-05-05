from collections import defaultdict
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


def canonical_smiles(mol):
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def murcko_scaffold_smiles(mol):
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


def morgan_fp(mol, radius=2, n_bits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol is not None else None


def tanimoto_distance(fp1, fp2):
    if fp1 is None or fp2 is None:
        return 1.0
    return 1.0 - DataStructs.TanimotoSimilarity(fp1, fp2)


def crowding_distance(objectives, front_indices):
    d = {i: 0.0 for i in front_indices}
    if len(front_indices) <= 2:
        return {i: float("inf") for i in front_indices}
    arr = objectives[front_indices]
    for k in range(arr.shape[1]):
        order = np.argsort(arr[:, k])
        d[front_indices[order[0]]] = float("inf")
        d[front_indices[order[-1]]] = float("inf")
        span = max(arr[order[-1], k] - arr[order[0], k], 1e-8)
        for j in range(1, len(order) - 1):
            i = front_indices[order[j]]
            if np.isfinite(d[i]):
                d[i] += (arr[order[j + 1], k] - arr[order[j - 1], k]) / span
    return d


def select_scaffold_diverse(candidate_indices, scores, scaffolds, k):
    groups = defaultdict(list)
    for i in candidate_indices:
        groups[scaffolds[i]].append(i)
    picks = []
    for _, idxs in groups.items():
        picks.append(max(idxs, key=lambda i: float(scores[i])))
    picks = sorted(picks, key=lambda i: float(scores[i]), reverse=True)
    if len(picks) < k:
        remain = [i for i in sorted(candidate_indices, key=lambda i: float(scores[i]), reverse=True) if i not in picks]
        picks.extend(remain[: (k - len(picks))])
    return picks[:k]


def select_fingerprint_diverse(candidate_indices, scores, fps, pareto_rank, k, alpha=1.0, gamma=0.2, rank_weight=0.1):
    ordered = sorted(candidate_indices, key=lambda i: (pareto_rank[i], -float(scores[i])))
    if not ordered:
        return []
    selected = [ordered[0]]
    pool = set(ordered[1:])
    while pool and len(selected) < k:
        best_i, best_v = None, -1e9
        for i in list(pool):
            min_d = min(tanimoto_distance(fps[i], fps[j]) for j in selected)
            v = alpha * float(scores[i]) - rank_weight * float(pareto_rank[i]) + gamma * min_d
            if v > best_v:
                best_i, best_v = i, v
        selected.append(best_i)
        pool.remove(best_i)
    return selected
