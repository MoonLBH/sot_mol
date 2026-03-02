import os
from concurrent.futures import ProcessPoolExecutor

import torch
from rdkit import Chem
from torchmetrics import Metric

from .rdkit import mol_is_valid,smiles_from_mol,optimise_mol,calc_energy,conf_distance 
from ..comparm import GP



def calc_atom_stabilities(mol):
    stabilities = []

    for atom in mol.GetAtoms():
        atom_type = atom.GetSymbol()
        valence = atom.GetExplicitValence()
        charge = atom.GetFormalCharge()

        if atom_type not in GP.ALLOWED_VALENCIES:
            stabilities.append(False)
            continue

        allowed = GP.ALLOWED_VALENCIES[atom_type]
        atom_stable = _is_valid_valence(valence, allowed, charge)
        stabilities.append(atom_stable)

    return stabilities


def _is_valid_valence(valence, allowed, charge):
    if isinstance(allowed, int):
        valid = allowed == valence

    elif isinstance(allowed, list):
        valid = valence in allowed

    elif isinstance(allowed, dict):
        allowed = allowed.get(charge)
        if allowed is None:
            return False

        valid = _is_valid_valence(valence, allowed, charge)

    return valid


def _is_valid_float(num):
    return num not in [None, float("inf"), float("-inf"), float("nan")]


class GenerativeMetric(Metric):
    # TODO add metric attributes - see torchmetrics doc

    def __init__(self, **kwargs):
        # Pass extra kwargs (defined in Metric class) to parent
        super().__init__(**kwargs)

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


class PairMetric(Metric):
    def __init__(self, **kwargs):
        # Pass extra kwargs (defined in Metric class) to parent
        super().__init__(**kwargs)

    def update(self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


class AtomStability(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("atom_stable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, stabilities: list[list[bool]]) -> None:
        all_atom_stables = [atom_stable for atom_stbs in stabilities for atom_stable in atom_stbs]
        self.atom_stable += sum(all_atom_stables)
        self.total += len(all_atom_stables)

    def compute(self) -> torch.Tensor:
        return self.atom_stable.float() / self.total


class MoleculeStability(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("mol_stable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, stabilities: list[list[bool]]) -> None:
        mol_stables = [sum(atom_stbs) == len(atom_stbs) for atom_stbs in stabilities]
        self.mol_stable += sum(mol_stables)
        self.total += len(mol_stables)

    def compute(self) -> torch.Tensor:
        return self.mol_stable.float() / self.total


class Validity(GenerativeMetric):
    def __init__(self, connected=False, **kwargs):
        super().__init__(**kwargs)
        self.connected = connected

        self.add_state("valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        is_valid = [mol_is_valid(mol, connected=self.connected) for mol in mols if mol is not None]
        self.valid += sum(is_valid)
        self.total += len(mols)

    def compute(self) -> torch.Tensor:
        return self.valid.float() / self.total


# TODO I don't think this will work with DDP
class Uniqueness(GenerativeMetric):
    """Note: only tracks uniqueness of molecules which can be converted into SMILES"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valid_smiles = []

    def reset(self):
        self.valid_smiles = []

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        smiles = [smiles_from_mol(mol, canonical=True) for mol in mols if mol is not None]
        valid_smiles = [smi for smi in smiles if smi is not None]
        self.valid_smiles.extend(valid_smiles)

    def compute(self) -> torch.Tensor:
        num_unique = len(set(self.valid_smiles))
        uniqueness = torch.tensor(num_unique) / len(self.valid_smiles)
        return uniqueness


class Novelty(GenerativeMetric):
    def __init__(self, existing_mols: list[Chem.rdchem.Mol], **kwargs):
        super().__init__(**kwargs)

        n_workers = min(8, len(os.sched_getaffinity(0)))
        executor = ProcessPoolExecutor(max_workers=n_workers)

        futures = [executor.submit(smiles_from_mol, mol, canonical=True) for mol in existing_mols]
        smiles = [future.result() for future in futures]
        smiles = [smi for smi in smiles if smi is not None]

        executor.shutdown()

        self.smiles = set(smiles)

        self.add_state("novel", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        smiles = [smiles_from_mol(mol, canonical=True) for mol in mols if mol is not None]
        valid_smiles = [smi for smi in smiles if smi is not None]
        novel = [smi not in self.smiles for smi in valid_smiles]

        self.novel += sum(novel)
        self.total += len(novel)

    def compute(self) -> torch.Tensor:
        return self.novel.float() / self.total


class EnergyValidity(GenerativeMetric):
    def __init__(self, optimise=False, **kwargs):
        super().__init__(**kwargs)

        self.optimise = optimise

        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        num_mols = len(mols)

        if self.optimise:
            mols = [optimise_mol(mol) for mol in mols if mol is not None]

        energies = [calc_energy(mol) for mol in mols if mol is not None]
        valid_energies = [energy for energy in energies if _is_valid_float(energy)]

        self.n_valid += len(valid_energies)
        self.total += num_mols

    def compute(self) -> torch.Tensor:
        return self.n_valid.float() / self.total


class AverageEnergy(GenerativeMetric):
    """Average energy for molecules for which energy can be calculated

    Note that the energy cannot be calculated for some molecules (specifically invalid ones) and the pose optimisation
    is not guaranteed to succeed. Molecules for which the energy cannot be calculated do not count towards the metric.

    This metric doesn't require that input molecules have been sanitised by RDKit, however, it is usually a good idea
    to do this anyway to ensure that all of the required molecular and atom properties are calculated and stored.
    """

    def __init__(self, optimise=False, per_atom=False, **kwargs):
        super().__init__(**kwargs)

        self.optimise = optimise
        self.per_atom = per_atom

        self.add_state("energy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid_energies", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        if self.optimise:
            mols = [optimise_mol(mol) for mol in mols if mol is not None]

        energies = [calc_energy(mol, per_atom=self.per_atom) for mol in mols if mol is not None]
        valid_energies = [energy for energy in energies if _is_valid_float(energy)]

        self.energy += sum(valid_energies)
        self.n_valid_energies += len(valid_energies)

    def compute(self) -> torch.Tensor:
        return self.energy / self.n_valid_energies


class AverageStrainEnergy(GenerativeMetric):
    """
    The strain energy is the energy difference between a molecule's pose and its optimised pose. Estimated using RDKit.
    Only calculated when all of the following are true:
    1. The molecule is valid and an energy can be calculated
    2. The pose optimisation succeeds
    3. The energy can be calculated for the optimised pose

    Note that molecules which do not meet these criteria will not count towards the metric and can therefore give
    unexpected results. Use the EnergyValidity metric with the optimise flag set to True to track the proportion of
    molecules for which this metric can be calculated.

    This metric doesn't require that input molecules have been sanitised by RDKit, however, it is usually a good idea
    to do this anyway to ensure that all of the required molecular and atom properties are calculated and stored.
    """

    def __init__(self, per_atom=False, **kwargs):
        super().__init__(**kwargs)

        self.per_atom = per_atom

        self.add_state("total_energy_diff", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        opt_mols = [(idx, optimise_mol(mol)) for idx, mol in list(enumerate(mols)) if mol is not None]
        energies = [(idx, calc_energy(mol, per_atom=self.per_atom)) for idx, mol in opt_mols if mol is not None]
        valids = [(idx, energy) for idx, energy in energies if energy is not None]

        if len(valids) == 0:
            return

        valid_indices, valid_energies = tuple(zip(*valids))
        original_energies = [calc_energy(mols[idx], per_atom=self.per_atom) for idx in valid_indices]
        energy_diffs = [orig - opt for orig, opt in zip(original_energies, valid_energies)]

        self.total_energy_diff += sum(energy_diffs)
        self.n_valid += len(energy_diffs)

    def compute(self) -> torch.Tensor:
        return self.total_energy_diff / self.n_valid


class AverageOptRmsd(GenerativeMetric):
    """
    Average RMSD between a molecule and its optimised pose. Only calculated when all of the following are true:
    1. The molecule is valid
    2. The pose optimisation succeeds

    Note that molecules which do not meet these criteria will not count towards the metric and can therefore give
    unexpected results.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("total_rmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        valids = [(idx, optimise_mol(mol)) for idx, mol in list(enumerate(mols)) if mol is not None]
        valids = [(idx, mol) for idx, mol in valids if mol is not None]

        if len(valids) == 0:
            return

        valid_indices, opt_mols = tuple(zip(*valids))
        original_mols = [mols[idx] for idx in valid_indices]
        rmsds = [conf_distance(mol1, mol2) for mol1, mol2 in zip(original_mols, opt_mols)]

        self.total_rmsd += sum(rmsds)
        self.n_valid += len(rmsds)

    def compute(self) -> torch.Tensor:
        return self.total_rmsd / self.n_valid


class MolecularAccuracy(PairMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("n_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]) -> None:
        predicted_smiles = [smiles_from_mol(pred, canonical=True) for pred in predicted]
        actual_smiles = [smiles_from_mol(act, canonical=True) for act in actual]
        matches = [pred == act for pred, act in zip(predicted_smiles, actual_smiles) if act is not None]

        self.n_correct += sum(matches)
        self.total += len(matches)

    def compute(self) -> torch.Tensor:
        return self.n_correct.float() / self.total


class MolecularPairRMSD(PairMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("total_rmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]) -> None:
        valid_pairs = [(pred, act) for pred, act in zip(predicted, actual) if pred is not None and act is not None]
        rmsds = [conf_distance(pred, act) for pred, act in valid_pairs]
        rmsds = [rmsd for rmsd in rmsds if rmsd is not None]

        self.total_rmsd += sum(rmsds)
        self.n_valid += len(rmsds)

    def compute(self) -> torch.tensor:
        return self.total_rmsd / self.n_valid

class MaxRingSize(GenerativeMetric):
    """Track maximum ring size in generated molecules.
    
    Useful for monitoring large ring formation during training.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.add_state("total_max_ring", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        for mol in mols:
            if mol is None:
                continue
            try:
                ring_info = mol.GetRingInfo()
                ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
                max_ring = max(ring_sizes) if ring_sizes else 0
                self.total_max_ring += max_ring
                self.n_valid += 1
            except Exception:
                continue
    
    def compute(self) -> torch.Tensor:
        if self.n_valid == 0:
            return torch.tensor(0.0)
        return self.total_max_ring / self.n_valid


class LargeRingRatio(GenerativeMetric):
    """Track ratio of molecules with rings larger than threshold.
    
    Args:
        max_allowed_size: Maximum allowed ring size (default 6)
    """
    
    def __init__(self, max_allowed_size: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.max_allowed_size = max_allowed_size
        
        self.add_state("n_large_ring", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        for mol in mols:
            if mol is None:
                continue
            self.total += 1
            try:
                ring_info = mol.GetRingInfo()
                ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
                max_ring = max(ring_sizes) if ring_sizes else 0
                if max_ring > self.max_allowed_size:
                    self.n_large_ring += 1
            except Exception:
                continue
    
    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0)
        return self.n_large_ring.float() / self.total


class ConnectedRatio(GenerativeMetric):
    """Track ratio of fully connected molecules (no disconnected fragments).
    
    This is similar to Validity(connected=True) but specifically tracks connectivity.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.add_state("n_connected", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        for mol in mols:
            if mol is None:
                continue
            self.total += 1
            try:
                frags = Chem.GetMolFrags(mol)
                if len(frags) == 1:
                    self.n_connected += 1
            except Exception:
                continue
    
    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0)
        return self.n_connected.float() / self.total


class AverageFragments(GenerativeMetric):
    """Track average number of disconnected fragments per molecule.
    
    Ideal value is 1.0 (fully connected molecules).
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.add_state("total_frags", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        for mol in mols:
            if mol is None:
                continue
            try:
                frags = Chem.GetMolFrags(mol)
                self.total_frags += len(frags)
                self.n_valid += 1
            except Exception:
                continue
    
    def compute(self) -> torch.Tensor:
        if self.n_valid == 0:
            return torch.tensor(0.0)
        return self.total_frags / self.n_valid


STABILITY_METRICS = {
            "atom-stability": AtomStability(),
            "molecule-stability": MoleculeStability(),

        }

GEN_METRICS = {
            "validity": Validity(),
            "fc-validity": Validity(connected=True),
            "uniqueness": Uniqueness(),
            "energy-validity": EnergyValidity(),
            "opt-energy-validity": EnergyValidity(optimise=True),
            "energy": AverageEnergy(),
            "energy-per-atom": AverageEnergy(per_atom=True),
            "strain": AverageStrainEnergy(),
            "strain-per-atom": AverageStrainEnergy(per_atom=True),
            "opt-rmsd": AverageOptRmsd(),
            "MaxRingSize": MaxRingSize(),
            "LargeRingRatio": LargeRingRatio(),
            "ConnectedRatio": ConnectedRatio(),
            "AverageFragments": AverageFragments()
        }
PAIRWISE_METRICS = {
                "mol-accuracy": MolecularAccuracy(),
                "pair-rmsd": MolecularPairRMSD()
}
