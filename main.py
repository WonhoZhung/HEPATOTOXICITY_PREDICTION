import os
import sys
import time
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import model.utils as utils

from model.model import GCNModel
from model.dataset import data_to_sample
from model.predict import predict
from tqdm import tqdm


parser = argparse.ArgumentParser()                                              
parser.add_argument("--ngpu", help="number of gpu", type=int, default=0)        
parser.add_argument("--n_hidden", help="n_hidden", type=int, default=128)
parser.add_argument("--n_layers", help="n_layer", type=int, default=4)
parser.add_argument("--model_saved", help="model_saved", type=str, \
        default="./best_model.pt")
parser.add_argument("--smiles_filename", help="smiles_filename", type=str, \
        default=None)
parser.add_argument("--result_filename", help="result_filename", type=str, \
        default=None)
parser.add_argument("--visualize", help="visualize", action="store_true")
args = parser.parse_args()


# Set GPU
cmd = utils.set_cuda_visible_device(args.ngpu)
os.environ['CUDA_VISIBLE_DEVICES'] = cmd[:-1]

# Model
model = GCNModel(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.initialize_model(model, device, load_save_file=args.model_saved)

# Read .smi File
with open(args.smiles_filename, 'r') as f: 
    smiles_list = [l.strip() for l in f.readlines()]
smiles_list = [s for s in smiles_list if '.' not in s]
test_data = (smiles_list, [1 for _ in range(len(smiles_list))])

save_pred, save_grad, save_smi = dict(), dict(), dict()

for idx in tqdm(range(len(test_data[0]))):

    smiles, _ = test_data[0][idx], test_data[1][idx]
    sample = data_to_sample(smiles)
    sample['idx'] = idx
    sample['smiles'] = smiles
    
    x, adj, edge, _, _ = sample.values()
    sample['x'] = torch.Tensor(x).unsqueeze(0)
    sample['edge'] = torch.Tensor(edge).unsqueeze(0)
    sample['adj'] = torch.Tensor(adj).unsqueeze(0)
    
    # ===================================================
    st = time.time()
    
    try: pred, true, grad = predict(model, sample, device)
    except Exception as e: print(smiles, e); continue

    et = time.time()
    # ===================================================

    save_pred[idx] = pred
    save_grad[idx] = grad
    save_smi[idx] = smiles

    # utils.view_grad(f"{idx}", smiles, grad)


save_pred = [(k, save_smi[k], v) for k, v in save_pred.items()]
# save_pred = sorted(save_pred, key=lambda x: x[2], reverse=False)

# Write Result
utils.write_result(args.result_filename, save_pred)

if args.visualize:
    for idx, smi, score in save_pred: 
        print(save_grad[idx])
        utils.view_grad(f"draw/{idx}", smi, save_grad[idx])
