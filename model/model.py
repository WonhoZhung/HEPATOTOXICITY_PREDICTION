import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution, ReadOut, SumPooling


class GCNModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.in_features = 67
        self.n_layers = args.n_layers
        self.n_hidden = args.n_hidden
        
        self.embedding = nn.Linear(self.in_features, self.n_hidden, bias=False)
        self.GCN = nn.ModuleList(
                [GraphConvolution(self.n_hidden, self.n_hidden) \
                        for _ in range(self.n_layers)]
        )
        self.readout = ReadOut(self.n_hidden, pooling=SumPooling())
        self.fc_layer_GCN = nn.Sequential(
                        nn.Linear(self.n_hidden, 1),
                        nn.Sigmoid()
        )
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, sample):
        x, adj = sample['node'], sample['adj']

        h = self.embedding(x)

        for i in range(self.n_layers):
            h_ori = h
            h = self.dropout(h)
            h = torch.sigmoid(self.GCN[i](h, adj))
            h = torch.relu(h+h_ori) # Residual

        g = self.readout(h)
        g = self.dropout(g)
        return self.fc_layer_GCN(g).squeeze(-1)


