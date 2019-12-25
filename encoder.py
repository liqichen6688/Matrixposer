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
        self.weight = nn.Parameter(torch.empty((50, 300)).normal_(mean=0,std=0.0001))
        self.bias = nn.Parameter(torch.empty((1, 300)).normal_(mean=0,std=0.0001))
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(300)

        self.weightpasttoken = nn.Parameter(torch.empty((50, 300)).normal_(mean=0,std=0.0001))
        self.biaspasttoken = nn.Parameter(torch.empty((1, 300)).normal_(mean=0, std=0.0001))

        self.weightpastgate = nn.Parameter(torch.empty((50, 300)).normal_(mean=0,std=0.0001))
        self.biaspastgate = nn.Parameter(torch.empty((1, 300)).normal_(mean=0, std=0.0001))

        self.weightpaststate = nn.Parameter(torch.empty((50, 300)).normal_(mean=0,std=0.0001))
        self.biaspaststate = nn.Parameter(torch.empty((1, 300)).normal_(mean=0, std=0.0001))

        self.weightexpose = nn.Parameter(torch.empty((50, 300)).normal_(mean=0,std=0.0001))
        self.biasexpose = nn.Parameter(torch.empty((1, 300)).normal_(mean=0, std=0.0001))



    def forward(self, x, matrix_embed, past):
        #print(matrix_embed)
        token = self.norm(torch.matmul(x, matrix_embed))

        past_token = torch.tanh(torch.matmul(x, self.weightpasttoken) + self.biaspasttoken)
        past_vector = torch.tanh(torch.matmul(past, self.weightpastgate) + self.biaspastgate)
        past_gate = torch.sigmoid(torch.matmul(past_token, past_vector.permute(0, 2, 1)))

        past_state = torch.tanh(torch.matmul(past, self.weightpaststate) + self.biaspaststate)
        pre_state = torch.matmul(past_gate, past_state)

        expose_vector = torch.tanh(torch.matmul(x, self.weightexpose) + self.biasexpose)
        expose_gate = torch.sigmoid(torch.matmul(expose_vector, pre_state.permute(0, 2, 1)))

        filter_token = token + torch.matmul(expose_gate, pre_state)
        #filter_token = token + pre_state #+ torch.tanh(torch.matmul(x, self.weight) + self.bias)
        return self.dropout(self.out(filter_token))

class NewDecoder(nn.Module):
    def __init__(self, output_size, d_model1, dropout=0.1):
        super(NewDecoder, self).__init__()
        self.out = nn.Sequential(
            nn.Linear(d_model1, d_model1),
            nn.ReLU(),
            nn.Linear(d_model1, output_size),
            #nn.Softmax(dim=-1)
        )
        self.weight = nn.Parameter(torch.empty((300, 300)).normal_(mean=0,std=0.0001))
        self.bias = nn.Parameter(torch.empty((1, 300)).normal_(mean=0,std=0.0001))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = LayerNorm(300)
        self.norm2 = LayerNorm(300)

        self.weightretoken = nn.Parameter(torch.empty((300, 300)).normal_(mean=0,std=0.0001))
        self.biasretoken = nn.Parameter(torch.empty((1, 300)).normal_(mean=0,std=0.0001))

        self.weightpast = nn.Parameter(torch.empty((300, 300)).normal_(mean=0,std=0.0001))
        self.biaspast = nn.Parameter(torch.empty((1, 300)).normal_(mean=0, std=0.0001))

        #self.weightpaexpose = nn.Parameter(torch.empty((300, 300)).normal_(mean=0,std=0.0001))
        #self.biaspaexpose = nn.Parameter(torch.empty((1, 300)).normal_(mean=0, std=0.0001))

        self.weightpre = nn.Parameter(torch.empty((300, 300)).normal_(mean=0,std=0.0001))
        self.biaspre = nn.Parameter(torch.empty((1, 300)).normal_(mean=0, std=0.0001))

        #self.weightpreexpose = nn.Parameter(torch.empty((300, 300)).normal_(mean=0,std=0.0001))
        #self.biaspreexpose = nn.Parameter(torch.empty((1, 300)).normal_(mean=0, std=0.0001))




    def forward(self, x, matrix_embed, past):
        #print(matrix_embed)


        past_represent = torch.tanh(torch.matmul(past, self.weightretoken) + self.biasretoken)
        past_embeding = torch.matmul(past.permute(0, 2, 1), past_represent) / past.shape[1]

        past_token = torch.tanh(torch.matmul(x, self.weightpast) + self.biaspast)
        past = self.norm2(torch.matmul(past_token, past_embeding))


        #past_content = torch.tanh(torch.matmul(past_base, self.weightpast) + self.biaspast)
        #past_expose = torch.sigmoid(torch.matmul(past_base, self.weightpaexpose) + self.biaspaexpose)
        #past_token = past_content * past_expose

        present_token = torch.tanh(torch.matmul(x, self.weightpre) + self.biaspre)
        present = self.norm2(torch.matmul(present_token, matrix_embed))

        #present_base = self.norm1(torch.matmul(x, matrix_embed))
        #present_content = torch.tanh(torch.matmul(present_base, self.weightpre) + self.biaspre)
        #present_expose = torch.sigmoid(torch.matmul(past_base, self.weightpreexpose) + self.biaspreexpose)
        #present_token = present_content * present_expose


        filter_token = present + past
        #filter_token = token + pre_state #+ torch.tanh(torch.matmul(x, self.weight) + self.bias)
        return self.dropout(self.out(filter_token))



