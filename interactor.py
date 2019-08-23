import torch
from torch.autograd import Variable
from sublayer import SublayerOutput
import torch.nn.functional as F
import math
from torch import nn


class Column_wise_nn(nn.Module):
    def __init__(self, d_row, d_ff, dropout=0.1):
        '''
        initialize column-wise neural network
        :param d_row: input row number
        :param d_ff: middle size row number
        :param dropout: default None
        '''
        super(Column_wise_nn, self).__init__()
        self.w_1 = nn.Linear(d_row, d_ff)
        self.w_2 = nn.Linear(d_ff, d_ff)
        self.w_3 = nn.Linear(d_ff, d_row)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = x.permute(0,2,1)
        #d_k = x.size(-1)
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        output = F.relu(output)
        if self.dropout is not None:
            output = self.dropout(output)

        return output.permute(0,2,1)


class Row_wise_nn(nn.Module):
    def __init__(self, d_column, d_ff, out_row, dropout=None):
        super(Row_wise_nn, self).__init__()
        self.w_1 = nn.Linear(d_column, d_ff)
        self.w_2 = nn.Linear(d_ff, d_ff)
        self.w_3 = nn.Linear(d_ff, out_row)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        d_k = x.size(-1)
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        output = self.w_3(self.dropout(F.relu(output))) / math.sqrt(d_k)
        output = F.softmax(output, dim=-1)
        if self.dropout is not None:
            output = self.dropout(output)

        return output

class ConvPoser(nn.Module):
    def __init__(self):
        super(ConvPoser, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv1(x)

class Interactor(nn.Module):
    def __init__(self, d_column, d_ff, out_row=30, dropout=0.1):
        '''
        :param d_row: dimension of output row number
        :param d_column: dimension of input column number
        :param d_ff: dimension of middle neural
        :param dropout: default 0.1
        '''
        super(Interactor, self).__init__()
        self.column_wise_nn = Column_wise_nn(out_row, d_ff, dropout)
        self.row_wise_nn = Row_wise_nn(d_column, d_ff, out_row, dropout)
        self.conv = ConvPoser()

    def forward(self, x):
        conv_input = x.unsqueeze(1)
        conv_output = self.conv(conv_input)
        print(conv_output.size)
        left_transposer = conv_output.squeeze(1)
        print(left_transposer.size)
        middle_term = torch.matmul(left_transposer.permute(0,2,1), x)
#        output = self.column_wise_nn(middle_term)
        output = self.column_wise_nn(middle_term)
        #output = torch.matmul(middle_term, right_transposer.permute(0,2,1))
        return output

