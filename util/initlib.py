"""Util file for Equinv scripts"""

import math
import resource
from pathlib import Path

import numpy as np
import torch
from openbabel import pybel
from rdkit import RDLogger
from torchmetrics import MetricCollection
from tqdm import tqdm

from .functional import one_hot_encode_tensor 
from ..comparm import GP
from .rdkit import PT


def disable_lib_stdout():
    pybel.ob.obErrorLog.StopLogging()
    RDLogger.DisableLog("rdApp.*")


# Need to ensure the limits are large enough when using OT since lots of preprocessing needs to be done on the batches
# OT seems to cause a problem when there are not enough allowed open FDs
def configure_fs(limit=4096):
    """
    Try to increase the limit on open file descriptors
    If not possible use a different strategy for sharing files in torch
    """

    n_file_resource = resource.RLIMIT_NOFILE
    soft_limit, hard_limit = resource.getrlimit(n_file_resource)

    print(f"Current limits (soft, hard): {(soft_limit, hard_limit)}")

    if limit > soft_limit:
        try:
            print(f"Attempting to increase open file limit to {limit}...")
            resource.setrlimit(n_file_resource, (limit, hard_limit))
            print("Limit changed successfully!")

        except Exception:
            print("Limit change unsuccessful. Using torch file_system file sharing strategy instead.")

            import torch.multiprocessing

            torch.multiprocessing.set_sharing_strategy("file_system")

    else:
        print("Open file limit already sufficiently large.")


# Applies the following transformations to a molecule:
# 1. Scales coordinate values by 1 / coord_std (so that they are standard normal)
# 2. Applies a random rotation to the coordinates
# 3. Removes the centre of mass of the molecule
# 4. Creates a one-hot vector for the atomic numbers of each atom
# 5. Creates a one-hot vector for the bond type for every possible bond
# 6. Encodes charges as non-negative numbers according to encoding map
def mol_transform(molecule, vocab, n_bonds, coord_std):
    rotation = tuple(np.random.rand(3) * np.pi * 2)
    molecule = molecule.scale(1.0 / coord_std).rotate(rotation).zero_com()

    atomic_nums = [int(atomic) for atomic in molecule.atomics.tolist()]
    tokens = [PT.symbol_from_atomic(atomic) for atomic in atomic_nums]
    one_hot_atomics = torch.tensor(vocab.indices_from_tokens(tokens, one_hot=True))

    bond_types = one_hot_encode_tensor(molecule.bond_types, n_bonds)

    charge_idxs = [GP.CHARGE_IDX_MAP[charge] for charge in molecule.charges.tolist()]
    charge_idxs = torch.tensor(charge_idxs)

    transformed = molecule._copy_with(atomics=one_hot_atomics, bond_types=bond_types, charges=charge_idxs)
    return transformed

# TODO support multi gpus
def calc_train_steps(dm, epochs, acc_batches):
    dm.setup("train")
    steps_per_epoch = math.ceil(len(dm.train_dataloader()) / acc_batches)
    return steps_per_epoch * epochs

