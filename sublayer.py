import torch
from torch import nn


class LayerNorm(nn.Module):
    "Construct a layer normalization module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    '''
    A residual connection followed by a layer norm
    '''

    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.size = size
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if x.size(-1) == self.size:
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            return x + self.dropout(sublayer(self.norm(x.permute(0,2,1)).permute(0,2,1)))
