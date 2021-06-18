import torch
from .utils import dic_to_device

def predict(model, sample, device):
    model.eval() 
    sample = dic_to_device(sample, device)
    smiles = sample['smiles']
    #true = sample['label']
    true = None
    x = sample['x']
    x.requires_grad = True
    pred = model(sample)

    pred.backward(retain_graph=True)
    grad = getattr(x, 'grad', None)
    grad = torch.sum(grad, -1).squeeze(0)
    print(grad)
    grad = torch.abs(grad / (1e-12+torch.norm(grad)))
    grad = 1 / (1+torch.exp(-10*(grad-0.5)))

    pred = pred.data.cpu().numpy()
    # true = true.data.cpu().numpy()

    return pred, true, grad
