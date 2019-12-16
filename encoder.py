from torch import nn
import torch
from train_utils import clones, Matrix_Embedding
from sublayer import LayerNorm, SublayerOutput, MatrixNorm

class Encoder(nn.Module):
    '''
    Matposer Encoder

    It is a stack of N layers.
    '''
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class EncoderLayer(nn.Module):
    '''
    An encoder layer

    Made up of Interactor and a feed forward layer
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''
    def __init__(self, size, interactor, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.interactor = interactor
        self.feed_forward = feed_forward
        self.sublayer = SublayerOutput(size, dropout)
        self.size = size

    def forward(self, x):
        "Matposer Encoder"
        x = self.interactor(x)
        return self.sublayer(x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, output_size, d_model1, dropout=0.1):
        super(Decoder, self).__init__()
        self.out = nn.Sequential(
            nn.Linear(d_model1, d_model1),
            nn.ReLU(),
            nn.Linear(d_model1, output_size),
            #nn.Softmax(dim=-1)
        )
        self.weight = nn.Parameter(torch.empty((50, 300)).normal_(mean=0,std=0.0000001))
        self.bias = nn.Parameter(torch.empty((1, 300)).normal_(mean=0,std=0.0000001))
        self.dropout = nn.Dropout(dropout)
        self.norm = MatrixNorm(size=(50,300))

    def forward(self, x, matrix_embed):
        #print(matrix_embed)
        token = torch.matmul(x, matrix_embed)
        filter_token = token + torch.tanh(torch.matmul(x, self.weight) + self.bias)
        return self.dropout(self.out(filter_token))



