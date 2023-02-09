import os
import numpy as np

import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import AllChem


def dic_to_device(dic, device):
    for dic_key, dic_value in dic.items():
        if isinstance(dic_value, torch.Tensor):
            dic_value = dic_value.to(device)
            dic[dic_key] = dic_value
    
    return dic

def load_data(filename):
    data = dict()
    with open(filename) as f:
        lines = f.readlines()[1:]
        lines = [l.strip() for l in lines]
        for l in lines:
            l = l.split(',')
            if len(l)==5: l.append(-100.0)
            k = l[1]
            if k not in data.keys(): data[k] = []
            data[k].append((int(l[2]),int(l[3]),l[4],float(l[5])))

    return data
        
def set_cuda_visible_device(ngpus):
    import subprocess
    import os
    import numpy as np
    empty = []
    if ngpus>0:
        fn = f'/tmp/empty_gpu_check_{np.random.randint(0,10000000,1)[0]}'
        for i in range(4):
            os.system(f'nvidia-smi -i {i} | grep "No running" | wc -l > {fn}')
            with open(fn) as f:
                out = int(f.read())
            if int(out)==1:
                empty.append(i)
            if len(empty)==ngpus: break
        if len(empty)<ngpus:
            print ('avaliable gpus are less than required', len(empty), ngpus)
            exit(-1)
        os.system(f'rm -f {fn}')        
    
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','

    return cmd

def initialize_model(model, device, load_save_file=False):
    if load_save_file:
        if device.type=='cpu':
            save_file_dict = torch.load(load_save_file, map_location='cpu')
            new_save_file_dict = dict()
            for k in save_file_dict:
                new_key = k.replace("module.", "")
                new_save_file_dict[new_key] = save_file_dict[k]
            model.load_state_dict(new_save_file_dict, strict=False)
        else:
            save_file_dict = torch.load(load_save_file)
            new_save_file_dict = dict()
            for k in save_file_dict:
                new_key = k.replace("module.", "")
                new_save_file_dict[new_key] = save_file_dict[k]
            model.load_state_dict(new_save_file_dict, strict=False)
    else:
        for param in model.parameters():
            if not param.requires_grad: 
                continue
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(model)
    model.to(device)
    return model

def get_dataset_dataloader(train_data, test_data, 
                batch_size, num_workers):
    from torch.utils.data import DataLoader
    from dataset import GraphDataset, tensor_collate_fn
    train_dataset = GraphDataset(train_data)
    train_dataloader = DataLoader(train_dataset,
                                   batch_size,
                                   num_workers=num_workers,
                                   collate_fn=tensor_collate_fn,
                                   shuffle=True)

    test_dataset = GraphDataset(test_data)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size,
                                  num_workers=num_workers,
                                  collate_fn=tensor_collate_fn,
                                  shuffle=False)
    return train_dataset, train_dataloader, test_dataset, test_dataloader

def write_result(filename, pred):
    with open(filename, 'w')  as w:
        w.write(f"Index\tSmiles\tScore\tUncertainty\n")
        for idx, smi, score, unc in pred:
            w.write(f"{idx}\t{smi}\t{float(score):.3f}\t{float(unc):.3f}\n")
    return

def view_grad(filename, smiles, grad):
    import cairosvg
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw.SimilarityMaps import GetSimilarityMapFromWeights

    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolDraw2DCairo(400, 400)
    GetSimilarityMapFromWeights(mol, grad.tolist(), draw2d=img, contourLines=0)
    img.FinishDrawing()
    img.WriteDrawingText(filename)
    return 

def calc_precision(pred, true):
    from sklearn.metrics import precision_score
    pred = np.array(list(pred.values()))
    true = np.array(list(true.values()))
    return precision_score(true, pred.round())

def calc_recall(pred, true):
    from sklearn.metrics import recall_score
    pred = np.array(list(pred.values()))
    true = np.array(list(true.values()))
    return recall_score(true, pred.round())

def calc_auroc(pred, true):
    from sklearn.metrics import roc_auc_score
    pred = np.array(list(pred.values()))
    true = np.array(list(true.values()))
    return roc_auc_score(true, pred)

def calc_accuracy(pred, true):
    from sklearn.metrics import accuracy_score
    pred = np.array(list(pred.values()))
    true = np.array(list(true.values()))
    return accuracy_score(true, pred.round())


if __name__ == '__main__':
    
    pass
