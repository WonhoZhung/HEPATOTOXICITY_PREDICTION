import torch
import torch.nn as nn
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem.AllChem import GetAdjacencyMatrix

import numpy as np


def data_to_sample(smiles):
    
    sample = dict()

    mol = Chem.MolFromSmiles(smiles)
    adj = GetAdjacencyMatrix(mol) + np.eye(mol.GetNumAtoms())
        
    x = [get_atom_feature(mol.GetAtomWithIdx(i)) for i in range(mol.GetNumAtoms())]
    x = np.array(x)
    edge = get_edge_feature(mol)
    
    sample['x'] = x
    sample['adj'] = adj
    sample['edge'] = edge

    return sample

def get_edge_feature(molecule, extra_features=True):
    n_bond_features = 5
    n_extra_bond_features = 6
    
    n_atoms = molecule.GetNumAtoms()
    E = np.zeros((\
            n_atoms, n_atoms, n_bond_features+n_extra_bond_features))
    for i in range(n_atoms):
        atom = molecule.GetAtomWithIdx(i)  # rdkit.Chem.Atom
        for j in range(n_atoms):
            e_ij = molecule.GetBondBetweenAtoms(i, j)  # rdkit.Chem.Bond
            if e_ij != None:
                e_ij = get_bond_feature(e_ij, extra_features) # ADDED edge feat; one-hot vector
                e_ij = list(map(lambda x: 1 if x == True else 0, e_ij)) # ADDED edge feat; one-hot vector
                E[i,j,:] = np.array(e_ij)
    return E #(N, N, 11)

def get_bond_feature(bond, include_extra=False):
    bt = bond.GetBondType()  # rdkit.Chem.BondType
    retval = [
      bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
      bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
      0 # no bond
      ]
    if include_extra:
        bs = bond.GetStereo()
        retval += [bs == Chem.rdchem.BondStereo.STEREONONE,
                   bs == Chem.rdchem.BondStereo.STEREOANY,
                   bs == Chem.rdchem.BondStereo.STEREOZ,
                   bs == Chem.rdchem.BondStereo.STEREOE,
                   bs == Chem.rdchem.BondStereo.STEREOCIS,
                   bs == Chem.rdchem.BondStereo.STEREOTRANS
                  ]
    return np.array(retval)

def get_atom_feature(atom):
    # atom = m.GetAtomWithIdx(atom_i)
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Na', 'Ni', 'Hg', 'Si', 'K', 'Ca', 'Fe', 'Al', 'B', 'Mn', 'Mg', 'Li', 'Cu', 'Se', 'X']) +
                    one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetFormalCharge(), [-2, -1, 0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED]) +
                    one_of_k_encoding_unk(atom.GetChiralTag(), [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.rdchem.ChiralType.CHI_OTHER]) +
                    [atom.IsInRingSize(i) for i in range(3, 10)] +
                    [atom.GetIsAromatic()])  # (24, 6, 6, 5, 7, 7, 4, 7, 1) --> total 67

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# class GraphDataset(Dataset):
# 
#     def __init__(self, data):
#         if len(data) == 2:
#             self.smiles = data[0]
#             self.label = data[1]
#         else:
#             self.smiles = torch.Tensor(data)
#             self.label = None
# 
#     def __len__(self):
#         return len(self.smiles)
# 
#     def __getitem__(self, idx):
#         keydata = self.smiles[idx]
#         sample = data_to_sample(keydata)
#         
#         if self.label is not None:
#             label = self.label[idx]
#             sample['label'] = label
# 
#         sample['smiles'] = keydata
#         return sample


def check_dimension(tensors):
    size = []
    for tensor in tensors:
        if isinstance(tensor, np.ndarray):
            size.append(tensor.shape)
        else:
            size.append(0)
    size = np.asarray(size)

    return np.max(size, 0)


def collate_tensor(tensor, max_tensor, batch_idx):
    if isinstance(tensor, np.ndarray):
        dims = tensor.shape
        max_dims = max_tensor.shape
        slice_list = tuple([slice(0, dim) for dim in dims])
        slice_list = [slice(batch_idx, batch_idx + 1), *slice_list]
        max_tensor[tuple(slice_list)] = tensor
    elif isinstance(tensor, str):
        max_tensor[batch_idx] = tensor
    else:
        max_tensor[batch_idx] = tensor

    return max_tensor


def tensor_collate_fn(batch):
    # batch_items = [it for e in batch for it in e.items() if 'key' != it[0]]
    batch_items = [it for e in batch for it in e.items()]
    dim_dict = dict()
    total_key, total_value = list(zip(*batch_items))
    batch_size = len(batch)
    n_element = int(len(batch_items) / batch_size)
    total_key = total_key[0:n_element]
    for i, k in enumerate(total_key):
        value_list = [v for j, v in enumerate(total_value) if j % n_element == i]
        if isinstance(value_list[0], np.ndarray):
            dim_dict[k] = np.zeros(np.array(
                [batch_size, *check_dimension(value_list)])
            )
        elif isinstance(value_list[0], str):
            dim_dict[k] = ["" for _ in range(batch_size)]
        else:
            dim_dict[k] = np.zeros((batch_size,))

    ret_dict = {}
    for j in range(batch_size):
        if batch[j] == None: continue
        keys = []
        for key, value in dim_dict.items():
            value = collate_tensor(batch[j][key], value, j)
            if not isinstance(value, list):
                value = torch.from_numpy(value).float()
            ret_dict[key] = value

    return ret_dict


if __name__ == "__main__":
    
    pass
