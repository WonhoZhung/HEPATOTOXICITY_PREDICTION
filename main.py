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
        default="./model/best_model_230401.pt")
parser.add_argument("--smiles_filename", help="smiles_filename", type=str, \
        default=None)
parser.add_argument("--smiles", help="smiles", type=str, \
        default=None)
parser.add_argument("--result_filename", help="result_filename", type=str, \
        default=None)
parser.add_argument("--image_filename", help="image_filename", type=str, \
        default=None)
parser.add_argument("--grad_filename", help="grad_filename", type=str, \
        default=None)
parser.add_argument("--dropout", help="dropout", type=float, default=0.2)
parser.add_argument("--num_sample", type=int, default=30)
parser.add_argument("--verbose", help="verbose", action="store_true")
args = parser.parse_args()


# Set GPU
if args.ngpu > 0:
    cmd = utils.set_cuda_visible_device(args.ngpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = cmd[:-1]

# Model
model = GCNModel(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.initialize_model(model, device, load_save_file=args.model_saved)

# Read .smi File
if args.smiles_filename is not None:
    with open(args.smiles_filename, 'r') as f: 
        smiles_list = [l.strip() for l in f.readlines()]
    smiles_list = [s for s in smiles_list if '.' not in s]
elif args.smiles is not None:
    smiles_list = [args.smiles]
else:
    print("No input!!")
    exit()

save_pred, save_grad, save_smi = dict(), dict(), dict()

for idx, smi in tqdm(enumerate(smiles_list), total=len(smiles_list)):

    sample = data_to_sample(smi)
    sample['idx'] = idx
    sample['smiles'] = smi
    
    node, adj, edge, _, _ = sample.values()
    sample['node'] = torch.Tensor(node).unsqueeze(0)
    sample['edge'] = torch.Tensor(edge).unsqueeze(0)
    sample['adj'] = torch.Tensor(adj).unsqueeze(0)
    
    # ===================================================
    st = time.time()
    
    pred_list, grad_list = [], []
    
    for _ in range(args.num_sample):

        try: 
            pred, grad = predict(model, sample, device)
            pred_list.append(pred)
            grad_list.append(grad)
        except Exception as e: 
            print(smi, e); continue
    
    pred_list = np.array(pred_list)
    pred_mean = np.mean(pred_list)
    pred_std = np.std(pred_list)
    grad_list = np.array(grad_list).mean(0)

    et = time.time()
    # ===================================================

    save_pred[idx] = pred_mean, pred_std
    save_grad[idx] = grad
    save_smi[idx] = smi

    if args.verbose:
        print(f"{smi}: {pred_mean:.2f} ± {pred_std:.2f}")

if args.image_filename is not None and len(smiles_list) == 1:
    if ".png" not in args.image_filename:
        args.image_filename = args.image_filename + ".png"
    utils.view_grad(args.image_filename, smi, grad_list)

if args.grad_filename is not None and len(smiles_list) == 1:
    if ".npy" in args.grad_filename:
        args.grad_filename = args.grad_filename[:-4]
    np.save(args.grad_filename, grad_list)

save_pred = [(k, save_smi[k], v[0], v[1]) for k, v in save_pred.items()]

# Write Result
if args.result_filename is not None:
    utils.write_result(args.result_filename, save_pred)
