import json 
from rdkit import Chem 
from .util.tokeniser import Vocabulary
import numpy as np
import torch 

class GPARAMS():
    def __init__(self):
        self.DATASET="geom-drugs"
        self.ARCH = "semla"
        self.D_MODEL=384
        self.N_LAYERS = 12
        self.D_MESSAGE = 128
        self.D_EDGE = 128
        self.N_COORD_SETS = 64
        self.N_ATTN_HEADS = 32
        self.D_MESSAGE_HIDDEN = 128
        self.COORD_NORM = "length"
        self.SIZE_EMB = 64
        self.MAX_ATOMS = 256
        self.LR=0.0003
        self.BATCH_SIZE = 24
        self.ACC_BATCHES = 1
        self.GRADIENT_CLIP_VAL = 1.0

        self.LR_SCHEDULE = "constant"
        self.WARM_UP_STEPS = 10000
        self.N_VALIDATION_MOLS = 2000
        self.VAL_CHECK_EPOCHS = 10
        self.NUM_INFERENCE_STEPS = 100
        self.CAT_SAMPLING_NOISE_LEVEL = 1
        self.COORD_NOISE_STD_DEV = 0.2
        self.TYPE_DIST_TEMP = 1.0
        self.TIME_ALPHA = 2.0
        self.TIME_BETA = 1.0
        self.SPECIAL_TOKENS = ["<PAD>", "<MASK>"]
        self.CORE_ATOMS = ["H", "C", "N", "O", "F", "P", "S", "Cl"]
        self.OTHER_ATOMS = ["Br", "B", "Al", "Si", "As", "I", "Hg", "Bi"]
        self.EQUIVARIANT_OT=True
        self.BATCH_OT=False
        self.SCALE_OT=False
        self.COORDS_STD_DEV=2.407038688659668 # for drugs, 1.723299503326416 for qm9
        
        self.IDX_BOND_MAP = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}        
        self.BOND_IDX_MAP = {bond: idx for idx, bond in self.IDX_BOND_MAP.items()}
        self.IDX_CHARGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: -1, 5: -2, 6: -3}
        self.CHARGE_IDX_MAP = {charge: idx for idx, charge in self.IDX_CHARGE_MAP.items()}

        self.ALLOWED_VALENCIES = {
                                "H": {0: 1, 1: 0, -1: 0},
                                "C": {0: [3, 4], 1: 3, -1: 3},
                                "N": {0: [2, 3], 1: [2, 3, 4], -1: 2},  # In QM9, N+ seems to be present in the form NH+ and NH2+
                                "O": {0: 2, 1: 3, -1: 1},
                                "F": {0: 1, -1: 0},
                                "B": 3,
                                "Al": 3,
                                "Si": 4,
                                "P": {0: [3, 5], 1: 4},
                                "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
                                "Cl": 1,
                                "As": 3,
                                "Br": {0: 1, 1: 2},
                                "I": 1,
                                "Hg": [1, 2],
                                "Bi": [3, 5],
                                "Se": [2, 4, 6],
                                }
        
        self.COMPILER_CACHE_SIZE=128
        self.SELF_COND=True
        self.NO_EMA=False
        
        self.LOSS_WEIGHT={"type":0.2,"bond":1.0,"charge":1.0}
        self.LOG_STEPS=50

        self.ODE_SAMPLING_STRATEGY="log"
        self.RANDOM_SEED=12345
        self.NOISE_STRATEGY_TYPE="uniform-sample"
        self.CUDA_VISIBLE_DEVICES="0,1,2,3"
        self.SCALE_OT_FACTOR=0.2
        self.MINI_BATCH_SIZE=4
        self.MAX_STEPS=128
        self.WITH_HS=True 
        self.ATOM_PROBS=torch.Tensor([  
                                    7.5237e-14, 7.5237e-14, 4.3885e-01, 4.0728e-01, 6.5471e-02, 6.5678e-02,
                                    4.9734e-03, 1.0676e-04, 1.2374e-02, 4.1508e-03, 1.0878e-03, 1.0533e-06,
                                    7.5237e-14, 8.2761e-07, 7.5237e-14, 1.9938e-05, 7.5237e-14, 1.5047e-07
                                    ])
        self.BOND_PROBS=torch.Tensor([0.2000, 0.5439, 0.0359, 0.0008, 0.2194])
        return 
    
    def update(self):
        self.TOKENS = self.SPECIAL_TOKENS + self.CORE_ATOMS + self.OTHER_ATOMS
        self.N_BOND_TYPES = len(self.BOND_IDX_MAP.keys()) + 1
        self.VOCAB = Vocabulary(self.TOKENS)

        return 
    
            
def Loaddict2obj(dict,obj):
    objdict=obj.__dict__
    for i in dict.keys():
        if i not in objdict.keys():
            print ("%s not is not a standard setting option!"%i)
        objdict[i]=dict[i]
    obj.__dict__==objdict

def Update_PARAMS(obj,jsonfile):
    with open(jsonfile,'r') as f:
        jsondict=json.load(f)
        Loaddict2obj(jsondict,obj)

    obj.update()
    return obj

GP=GPARAMS()
