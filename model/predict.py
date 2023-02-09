import torch
from .utils import dic_to_device

def predict(model, sample, device):
    sample = dic_to_device(sample, device)
    smiles = sample['smiles']
    true = None
    node = sample['node']
    node.requires_grad = True
    pred = model(sample)

    pred.backward(retain_graph=True)
    grad = getattr(node, 'grad', None)
    grad = torch.sum(grad, -1).squeeze(0)
    grad = torch.abs(grad / (1e-12+torch.norm(grad)))
    grad = 1 / (1 + torch.exp(-10 * (grad - 0.5)))

    pred = pred.data.cpu().numpy()
    grad = grad.data.cpu().numpy()

    return pred, grad
