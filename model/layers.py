import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
         
        if self.bias == None:
            return torch.einsum('ijk,ikl->ijl', \
                    adj, F.linear(input, self.weight)) 
        else:
            return torch.einsum('ijk,ikl->ijl', \
                    adj, F.linear(input, self.weight, self.bias))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ReadOut(nn.Module):
    def __init__(self, n_hidden, pooling=None):
        super(ReadOut, self).__init__()

        self.pooling = pooling
        self.d = nn.Linear(n_hidden, n_hidden, bias=False)

    def forward(self, x):
        return self.pooling(self.d(x))


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=1)


class SumPooling(nn.Module):
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, x):
        return torch.sum(x, dim=1)


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=1)
