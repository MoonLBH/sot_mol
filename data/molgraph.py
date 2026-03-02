import rdkit, pickle, math 
import torch 
from rdkit import Chem,RDConfig
from rdkit.Chem import ChemicalFeatures,AllChem
import numpy as np
from ..util.functional import rotate,adj_from_edges,one_hot_encode_tensor
from ..util.rdkit import PT,mol_from_atoms , get_pharmcore_informations
from ..comparm import * 
import os 
import  itertools 

def flatten(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

class MolGraph:
    """
    Class to represent a molecule as a graph
    """

    def __init__(self, coords, atomics, bond_indices=None, bond_types=None, charges=None,str_id=None,):

        if len(atomics.shape)==1:
            self.atomics=self.one_hot_atomics(atomics)
        else:
            self.atomics=atomics 

        self.coords=coords
        self.natoms=len(atomics)
        self.bond_indices = torch.tensor([[]]*2).T if bond_indices is None else bond_indices
        bond_types = torch.tensor([1]*self.bond_indices.shape[0]) if bond_types is None else bond_types
        
        if len(bond_types.shape)==1:
            self.bond_types=self.one_hot_bond_types(bond_types)
        else:
            self.bond_types=bond_types

        self.nbonds=len(self.bond_indices)
        self.charges = torch.tensor([0]*self.natoms) if charges is None else charges    
        self.str_id=str_id
        
        self.flag_3D = 1 if np.std(self.coords.numpy()) >0 else 0

        return 
    
    @staticmethod
    def from_bytes(data: bytes):
        obj = pickle.loads(data)
        if obj.get("bond_types") is not None:
            bond_indices = obj["bond_indices"]
            bond_types = obj["bond_types"]
        else:
            bonds= obj["bonds"]
            bond_indices = bonds[:,:2]
            bond_types = bonds[:,2]

        mol=MolGraph(
            coords=obj["coords"],
            atomics=obj["atomics"],
            bond_indices=bond_indices,
            bond_types=bond_types,
            charges=obj["charges"],
            str_id=obj["id"],
        )
        return mol
    
    @staticmethod
    def from_rdkit(mol: Chem.rdchem.Mol, type="atom_based"):
        """
        Convert an RDKit molecule to a MolGraph object.
        type="atom_based" or "group_based"
        """

        if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
            raise RuntimeError("The default conformer must have 3D coordinates")

        conf = mol.GetConformer()

        smiles= Chem.MolToSmiles(mol)
        atomics = []
        coords = []
        charges = []

        for atom in mol.GetAtoms():
            atomics.append(atom.GetAtomicNum())
            charges.append(atom.GetFormalCharge())
        
        coords = np.array(conf.GetPositions())

        atomics = torch.tensor(atomics)
        coords = torch.tensor(coords)
        charges = torch.tensor(charges)
        
        # Create bond indices and types
        bond_indices = []
        bond_types = []
        for bond in mol.GetBonds():
            bond_indices.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            bond_type = GP.BOND_IDX_MAP.get(bond.GetBondType())
            if bond_type is None:
                raise ValueError(f"Unknown bond type: {bond.GetBondType()}")
            bond_types.append(bond_type)
        
        bond_indices = torch.tensor(bond_indices)
        bond_types = torch.tensor(bond_types)

        return MolGraph(coords=coords, atomics=atomics, bond_indices=bond_indices, bond_types=bond_types, charges=charges, str_id=smiles)

    def _copy_with(self, coords=None, atomics=None, bond_indices=None, bond_types=None, charges=None):

        coords = self.coords if coords is None else coords
        atomics = self.atomics if atomics is None else atomics
        bond_indices = self.bond_indices if bond_indices is None else bond_indices
        bond_types = self.bond_types if bond_types is None else bond_types
        charges = self.charges if charges is None else charges    
        obj = MolGraph(
            coords,
            atomics,
            bond_indices=bond_indices,
            bond_types=bond_types,
            charges=charges,
            str_id=self.str_id,
        )

        return obj
    
    def zero_com(self):
        shifted = self.coords - self.com.unsqueeze(0)
        return self._copy_with(coords=shifted)

    def rotate(self, rotation):
        rotated = rotate(self.coords, rotation)
        return self._copy_with(coords=rotated)

    def shift(self, shift):
        shift_tensor = torch.tensor(shift).view(1, -1)
        shifted = self.coords + shift_tensor
        return self._copy_with(coords=shifted)

    def scale(self, scale: float):
        scaled = self.coords * scale
        return self._copy_with(coords=scaled)

    def zero_coords(self):
        zeros = torch.zeros_like(self.coords)
        return self._copy_with(coords=zeros)
    
    @property
    def com(self):
        return self.coords.mean(dim=0)
    
    @property
    def adjacency(self):
        bond_indices = self.bond_indices
        bond_types = self.bond_types
        return adj_from_edges(bond_indices,bond_types,self.natoms,symmetric=True)
    
    @property
    def mask(self):
        mask = torch.ones(self.natoms).long()
        return mask
    
    @staticmethod
    def one_hot_atomics(atomics):
        atomic_nums = [int(atomic) for atomic in atomics.tolist()]
        tokens = [PT.symbol_from_atomic(atomic) for atomic in atomic_nums]
        one_hot_atomics = torch.tensor(GP.VOCAB.indices_from_tokens(tokens, one_hot=True))
        return one_hot_atomics

    @staticmethod
    def one_hot_bond_types(bond_types):
        one_hot_bond_types=one_hot_encode_tensor(bond_types, GP.N_BOND_TYPES)
        return one_hot_bond_types

    @staticmethod
    def one_hot_charges(charges):
        charge_idx = [GP.CHARGE_IDX_MAP[charge] for charge in charges.tolist()]
        charge_idx = torch.tensor(charge_idx)
        n_charges = len(GP.CHARGE_IDX_MAP.keys())
        charges_onehot = one_hot_encode_tensor(charge_idx, n_charges)
        return charges_onehot

    def permute(self,perm_indices):
        #print (self.coords.shape,perm_indices)
        perm_indices=torch.tensor(perm_indices)
        #print (perm_indices)
        coords = self.coords[perm_indices]
        atomics = self.atomics[perm_indices]
        charges = self.charges[perm_indices]

        # Relabel bond from and to indices with new indices
        from_idxs = self.bond_indices[:, 0].clone()
        to_idxs = self.bond_indices[:, 1].clone()
        curr_indices = torch.arange(perm_indices.size(0))

        old_from, new_from = torch.nonzero(from_idxs.unsqueeze(1) == curr_indices, as_tuple=True)
        old_to, new_to = torch.nonzero(to_idxs.unsqueeze(1) == curr_indices, as_tuple=True)

        from_idxs[old_from] = perm_indices[new_from]
        to_idxs[old_to] = perm_indices[new_to]

        # Remove bonds whose indices do not appear in new indices list
        bond_idxs = torch.cat((from_idxs.unsqueeze(-1), to_idxs.unsqueeze(-1)), dim=-1)
        mask = bond_idxs.unsqueeze(-1) == perm_indices.view(1, 1, -1)
        mask = ~(~mask.any(dim=-1)).any(dim=-1)
        bond_indices = bond_idxs[mask]
        bond_types = self.bond_types[mask]

        mol_copy = self._copy_with(
            coords=coords, atomics=atomics, bond_indices=bond_indices, bond_types=bond_types, charges=charges
        )
        return mol_copy
    
    def to_rdkit(self,sanitise=False):
        if len(self.atomics.size()) == 2:
            vocab_indices = torch.argmax(self.atomics, dim=1).tolist()
            tokens = GP.VOCAB.tokens_from_indices(vocab_indices)

        else:
            atomics = self.atomics.tolist()
            tokens = [PT.symbol_from_atomic(a) for a in atomics]

        coords = self.coords.numpy()        
        
        bond_types=torch.argmax(self.bond_types, dim=1).reshape(-1,1)

        bonds = torch.cat((self.bond_indices,bond_types),dim=-1).numpy()
        charges = self.charges.numpy()
        
        mol = mol_from_atoms(coords, tokens, bonds, charges, sanitise=sanitise)
        return mol
    
    def to_bytes(self):
        vocab_indices = torch.argmax(self.atomics, dim=1).tolist()
        tokens = GP.VOCAB.tokens_from_indices(vocab_indices)
        atomics = torch.tensor([PT.atomic_from_symbol(t) for t in tokens])
        bond_types= torch.argmax(self.bond_types, dim=1)

        datadict={"coords":self.coords,
               "atomics":atomics,
               "bond_indices":self.bond_indices,
               "bond_types":bond_types,
               "charges":self.charges,
               "id":self.str_id,
               }
        data = pickle.dumps(datadict)
        return data 

class MolGraphList:
    def __init__(self, mols: list[MolGraph],max_atoms=None):
        self.mols = mols
        
        if max_atoms is None:
            self.max_atoms = max([mol.natoms for mol in mols])
        else:
            self.max_atoms = max_atoms

        self._coords=None
        self._atomics=None
        self._bond_types=None
        self._bond_indices=None
        self._charges=None
        self._masks=None 
        self._natoms=None
        self._nbonds=None
        self._nmols=len(mols)
        self._flag_3D=None 

    def __len__(self):
        return len(self.mols)

    @property
    def coords(self):
        if self._coords is None:
            self._coords=pad_tensors([mol.coords for mol in self.mols],self.max_atoms,0)
        return self._coords
    
    @property
    def atomics(self):
        if self._atomics is None:
            self._atomics=pad_tensors([mol.atomics for mol in self.mols],self.max_atoms,0)
        
        return self._atomics

    @property
    def bond_types(self):
        if self._bond_types is None:
            self._bond_types=pad_tensors([mol.bond_types for mol in self.mols],self.max_atoms**2,0)
        
        return self._bond_types

    @property
    def bond_indices(self):
        if self._bond_indices is None:
            self._bond_indices=pad_tensors([mol.bond_indices for mol in self.mols],self.max_atoms**2,0)
        return self._bond_indices    
        
    @property
    def charges(self):
        if self._charges is None:
            self._charges=pad_tensors([mol.one_hot_charges(mol.charges) for mol in self.mols],self.max_atoms,0)
        return self._charges 

    @property
    def masks(self):
        if self._masks is None:
            self._masks=pad_tensors([mol.mask for mol in self.mols],self.max_atoms,0)
        
        return self._masks
    
    @property
    def natoms(self):
        if self._natoms is None:
            self._natoms=torch.tensor([[mol.natoms] for mol in self.mols]).long()

        return self._natoms
    
    @property
    def nbonds(self):
        if self._nbonds is None:
            self._nbonds=pad_tensors([mol.nbonds for mol in self.mols],self.max_atoms**2,0)
        return self._nbonds
    
    @property
    def adjacencies(self):        
        adjs=torch.stack([adj_from_edges(mol.bond_indices,mol.bond_types,self.max_atoms,symmetric=True) for mol in self.mols])
        return adjs

    @property
    def flag_3D(self):
        flags = torch.tensor([mol.flag_3D for mol in self.mols]).float()
        return flags

def pad_tensors(tensors, pad_length, pad_value=0):
    if len(tensors) == 0:
        return torch.empty(0)
    shape = list(tensors[0].shape)
    shape[0] = pad_length
    padded = []
    for t in tensors:
        pad_size = pad_length - t.shape[0]
        if pad_size > 0:
            # 1D
            if t.dim() == 1:
                pad = torch.full((pad_size,), pad_value, dtype=t.dtype, device=t.device)
                t_padded = torch.cat([t, pad], dim=0)
            # 2D
            elif t.dim() == 2:
                pad = torch.full((pad_size, t.shape[1]), pad_value, dtype=t.dtype, device=t.device)
                t_padded = torch.cat([t, pad], dim=0)
            else:
                raise ValueError("Only support 1D or 2D tensors")
        else:
            t_padded = t[:pad_length]
        padded.append(t_padded)
    return torch.stack(padded, dim=0)
