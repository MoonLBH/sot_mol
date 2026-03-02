import os
from functools import partial
from pathlib import Path 
import lightning as L
import torch,pickle 
from torch.utils.data import DataLoader
from .molgraph import MolGraph 
from .datasets import MGDataset 
from tqdm import tqdm 
from rdkit import Chem
import random 

class MGDataModule(L.LightningDataModule):
    def __init__(
        self,
        vocab,
        n_bond_types,
        train_datafile=None,
        val_datafile=None,
        test_datafile=None,
        max_atoms=50,
        coord_std=1.0,
        scale_ot=False,
        scale_ot_factor=0.2,
        batchsize=64,
        mini_batchsize=4,
        with_Hs=True,
        ot_geo_weight=1.0,
        ot_type_weight=1.0,
        ot_bond_weight=1.0,
    ):
        
        super().__init__()
        self.vocab=vocab
        self.n_bond_types=n_bond_types
        
        self.train_datafile=train_datafile
        self.val_datafile=val_datafile
        self.test_datafile=test_datafile

        self.max_atoms=max_atoms
        self.coord_std=coord_std

        self.scale_ot=scale_ot
        self.scale_ot_factor=scale_ot_factor
        self.ot_geo_weight = ot_geo_weight
        self.ot_type_weight = ot_type_weight
        self.ot_bond_weight = ot_bond_weight
        self.batchsize=batchsize
        self.mini_batchsize=mini_batchsize
        self.num_workers=20
        self.with_Hs=with_Hs

    @staticmethod
    def load_mgs(filepath,with_Hs=True):
        data_file=filepath.read_bytes()
        datas=pickle.loads(data_file)
        mgs=[]
        failed_num=0
        for data in tqdm(datas):
            if with_Hs:
                mg=MolGraph.from_bytes(data)
                mgs.append(mg)
            else:
                try:
                    mol=MolGraph.from_bytes(data)
                    rdkitmol=mol.to_rdkit()
                    if rdkitmol is not None:
                        mol_noh=Chem.RemoveAllHs(rdkitmol)
                        mg=MolGraph.from_rdkit(mol_noh)
                        mgs.append(mg)
                except:
                    failed_num+=1
        print (f"Failed to convert {failed_num} molecules to MolGraph without Hs for smol data in {filepath}.")
        return mgs 

    def setup_dataset(self, mgs=None,fixed_time=None,mode="train"):

        dataset=MGDataset(
            MGs=mgs,
            vocab=self.vocab,
            n_bond_types=self.n_bond_types,
            max_atoms=self.max_atoms,
            coord_std=self.coord_std,
            scale_ot=self.scale_ot,
            scale_ot_factor=self.scale_ot_factor,
            mini_batch_size=self.mini_batchsize,
            mode=mode,
            ot_geo_weight=self.ot_geo_weight,
            ot_type_weight=self.ot_type_weight,
            ot_bond_weight=self.ot_bond_weight,
        )
        return dataset 

    def setup(self,stage=None,train_mgs=None,val_mgs=None,test_mgs=None):
        if stage == "fit" or stage is None:
            if train_mgs is None:
                assert self.train_datafile.exists(), f"Train data file {self.train_datafile} must be provided when train_mgs is None."
                train_mgs=self.load_mgs(self.train_datafile,with_Hs=self.with_Hs,)

            self.trainset=self.setup_dataset(mgs=train_mgs,mode="train")

            if val_mgs is None:
                val_mgs=self.load_mgs(self.val_datafile,with_Hs=self.with_Hs,)
            random.shuffle(val_mgs)
            self.valset=self.setup_dataset(mgs=val_mgs[:200],mode="train")

        if stage == "test" or stage is None:
            if test_mgs is None:
                test_mgs = self.load_mgs(self.test_datafile,with_Hs=self.with_Hs,)
            random.shuffle(test_mgs)
            self.testset=self.setup_dataset(mgs=test_mgs,mode="test")

    def create_dataloader(self,dataset,shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batchsize,
            num_workers=self.num_workers,
            shuffle=shuffle,  # Lightning 会自动关闭 shuffle 并用 DistributedSampler
            pin_memory=True,
            drop_last=False,
        ) 
    
    def train_dataloader(self):
        return self.create_dataloader(self.trainset, shuffle=True)
    
    def val_dataloader(self):
        return self.create_dataloader(self.valset)
    
    def test_dataloader(self):
        return self.create_dataloader(self.testset)
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # 假设 batch 是一个字典，递归地将所有 tensor 转到 device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, dict):
                batch[k] = self.transfer_batch_to_device(v, device, dataloader_idx)
            # 如果有 list/tuple 也可以递归处理
        return batch
    

