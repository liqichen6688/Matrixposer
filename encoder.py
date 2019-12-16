from torch import nn
import torch
from train_utils import clones, Matrix_Embedding
from sublayer import LayerNorm, SublayerOutput

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
        self.norm = LayerNorm(300)
        self.weightre = nn.Parameter(torch.empty((50, 300)).normal_(mean=0,std=0.0000001))
        self.biasre = nn.Parameter(torch.empty((1, 300)).normal_(mean=0, std=0.0000001))


    def forward(self, x, matrix_embed, past_state):
        #print(matrix_embed)
        token = self.norm(torch.matmul(x, matrix_embed))
        past_state_true = self.norm(torch.matmul(past_state, matrix_embed))
        past_state = torch.matmul(past_state, self.weightre) + self.biasre
        reattention = torch.sigmoid(torch.matmul(token, past_state_true.permute(0, 2, 1)))
        pre_state = torch.tanh(torch.matmul(reattention, past_state))
        concate_state = torch.concate((token, pre_state), 1)
        wholeattention = torch.sigmoid(torch.matmul(token, concate_state.permute(0, 2, 1)))
        filter_token = torch.matmul(wholeattention, concate_state)
        #filter_token = token + pre_state #+ torch.tanh(torch.matmul(x, self.weight) + self.bias)
        return self.dropout(self.out(filter_token))



