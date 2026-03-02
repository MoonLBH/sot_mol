import torch 
import numpy as np
from ..util.functional import one_hot_encode_tensor,inter_distances
from .molgraph import MolGraph, MolGraphList
import math 
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

class MGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        MGs,
        vocab,
        n_bond_types,
        max_atoms=50,
        coord_std=1.0,
        scale_ot=False,
        scale_ot_factor=0.2,
        mini_batch_size=64,
        mode="train",
        ot_geo_weight=1.0,
        ot_type_weight=1.0,
        ot_bond_weight=1.0,
    ):
        
        super().__init__()
        self.MGs = MGs
        self.vocab = vocab
        self.n_bond_types = n_bond_types
        self.coord_std = coord_std
        self.max_atoms = max_atoms
        self.coord_dist=torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
        
        self.scale_ot=scale_ot
        self.scale_ot_factor = scale_ot_factor
        self.mini_batch_size=mini_batch_size
        self.mode=mode
        self.ot_geo_weight = ot_geo_weight
        self.ot_type_weight = ot_type_weight
        self.ot_bond_weight = ot_bond_weight

    def sample(self,n_samples):
        random_indices = np.random.choice(len(self.MGs), n_samples, replace=False)
        self.MGs= [self.MGs[i] for i in random_indices]
        return 
    
    def __len__(self):
        return math.ceil(len(self.MGs)/self.mini_batch_size)
    
    def _get_minibatch_of_molgraphs(self, item):
        mgs = self.MGs[item*self.mini_batch_size:(item+1)*self.mini_batch_size]

        if len(mgs) < self.mini_batch_size:
            mgs = mgs+self.MGs[:self.mini_batch_size-len(mgs)]
        
        flag_3Ds = torch.Tensor([mg.flag_3D for mg in mgs])
        
        mini_batch=[]        
        coms= [mg.com for mg in mgs]

        for mg in mgs:
            if mg.flag_3D == 1:
                if self.mode=="train":
                    rotation = tuple(np.random.rand(3) * np.pi * 2)
                else:
                    rotation=tuple(np.array([0,0,0]))
                mg = mg.scale(1/self.coord_std).rotate(rotation).zero_com()

            mini_batch.append(mg)
        
        return mini_batch,coms, flag_3Ds 
    
    def _sample_noise_mglist(self, flag_3Ds):

        noise_mglist=[self.sample_noise_mg(self.max_atoms) for _ in range(self.mini_batch_size)]

        noise_mini_batch=[]

        for mg, flag in zip(noise_mglist, flag_3Ds):
            if flag.item() == 0:
                mg = mg.zero_coords()
            else:
                if self.scale_ot:
                    mg=mg.scale(np.log(self.max_atoms+1)*self.scale_ot_factor).zero_com()
            noise_mini_batch.append(mg)
        
        return noise_mini_batch

    def __getitem__(self, item):
        mols,coms, flag_3Ds=self._get_minibatch_of_molgraphs(item)
        noise_mols=self._sample_noise_mglist(flag_3Ds=flag_3Ds)

        noise_mols,mols= self.OT(mols, noise_mols,flag_3Ds )

        real_mgl= MolGraphList(mols,max_atoms=self.max_atoms)

        noise_mgl=MolGraphList(noise_mols,max_atoms=self.max_atoms)

        data_batch={"real_atomics":real_mgl.atomics.float(),
                    "real_bonds":real_mgl.adjacencies.float(),
                    "real_charges":real_mgl.charges.long(),
                    "masks":real_mgl.masks.long(),
                    "noise_atomics":noise_mgl.atomics.float(),
                    "noise_bonds":noise_mgl.adjacencies.float(),
                    "natoms":real_mgl.natoms.long(),
                    "real_coords":real_mgl.coords.float(),
                    "noise_coords":noise_mgl.coords.float(),
                    "flag_3Ds":flag_3Ds.float(),
                    "coms":torch.stack(coms).float(),
                    }
    
        return data_batch

    def OT(self, mols, noise_mols, flag_3Ds):

        best_noise_mols= [
            self.mix_map(noise_mol, mol, use_coords=bool(flag.item()))
            for noise_mol, mol, flag in zip(noise_mols, mols, flag_3Ds)
        ]

        return  best_noise_mols, mols

    def sample_noise_mg(self, n_atoms):
        coords=self.coord_dist.sample((n_atoms,3))

        atomics_idx = torch.randint(0, self.vocab.size, (n_atoms,))
        
        atomics=one_hot_encode_tensor(atomics_idx,self.vocab.size)
        
        bond_indices=torch.ones((n_atoms,n_atoms)).nonzero()
        n_bonds=bond_indices.size(0)

        bond_types=torch.randint(0, self.n_bond_types, (n_bonds,))

        bond_types=one_hot_encode_tensor(bond_types,self.n_bond_types)
        noise_mg=MolGraph(coords,atomics,bond_indices=bond_indices,bond_types=bond_types)

        return noise_mg

    def mix_map(self,noise_mol,mol, use_coords=True):
        noise_mol=noise_mol.permute(list(range(mol.natoms)))

        cost = self._pairwise_cost_matrix(noise_mol, mol, use_coords=use_coords)

        _, noise_indices = linear_sum_assignment(cost.detach().cpu().numpy())

        noise_mol = noise_mol.permute(noise_indices.tolist())

        if use_coords:

            rotation, _ = Rotation.align_vectors(
                mol.coords.detach().cpu().numpy(),
                noise_mol.coords.detach().cpu().numpy(),
            )

            noise_mol=noise_mol.rotate(rotation)

        return noise_mol
    
    def _pairwise_cost_matrix(self, noise_mol, mol, use_coords=True):
        cost = torch.zeros((noise_mol.natoms, mol.natoms), dtype=torch.float32)

        if use_coords:
            coord_cost = inter_distances(mol.coords, noise_mol.coords, sqrd=True)
            weighted_coord = self.ot_geo_weight * coord_cost
            cost += weighted_coord

        type_cost = self._type_cost_matrix(noise_mol, mol)        
        weighted_type = self.ot_type_weight * type_cost
        cost += weighted_type

        if self.ot_bond_weight > 0:
            bond_feat_cost = self._bond_feat_cost_matrix(noise_mol, mol)
            cost += self.ot_bond_weight * bond_feat_cost

        return cost

    def _type_cost_matrix(self, noise_mol, mol):
        noise_types = noise_mol.atomics.float()
        mol_types = mol.atomics.float()
        sim = noise_types @ mol_types.T
        return 1.0 - sim

    def _normalize_probs(self, probs, dim):
        probs = probs.float().flatten()
        if probs.numel() != dim:
            raise ValueError(f"prob size mismatch: expect {dim}, got {probs.numel()}")
        probs = torch.clamp(probs, min=0)
        total = probs.sum()
        if total <= 0:
            return torch.ones(dim, dtype=torch.float32) / float(dim)
        return probs / total

    def _node_bond_features(self, mg):
        n = mg.natoms
        features = torch.zeros((n, self.n_bond_types + 1), dtype=torch.float32)

        if getattr(mg, "bond_indices", None) is None or getattr(mg, "bond_types", None) is None:
            return features

        if mg.bond_indices.numel() == 0:
            return features

        idx = mg.bond_indices.long()
        btypes = mg.bond_types.float()
        src = idx[:, 0]
        dst = idx[:, 1]

        features[:, :self.n_bond_types].index_add_(0, src, btypes)
        features[:, :self.n_bond_types].index_add_(0, dst, btypes)

        deg = features[:, :self.n_bond_types].sum(dim=-1, keepdim=True)
        features[:, :self.n_bond_types] = features[:, :self.n_bond_types] / (deg + 1e-6)

        max_deg = torch.clamp(deg.max(), min=1.0)
        features[:, -1:] = deg / max_deg

        return features

    def _bond_feat_cost_matrix(self, noise_mol, mol):
        noise_feat = self._node_bond_features(noise_mol)
        mol_feat = self._node_bond_features(mol)
        feat_dist = inter_distances(noise_feat, mol_feat, sqrd=True)
        return feat_dist
