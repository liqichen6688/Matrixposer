import torch
from torch.autograd import Variable
import torch.nn.functional as F
import math
from torch import nn


class Column_wise_nn(nn.Module):
    def __init__(self, d_row, d_ff, dropout=None):
        '''
        initialize column-wise neural network
        :param d_row: input row number
        :param d_ff: middle size row number
        :param dropout: default None
        '''
        super(Column_wise_nn, self).__init__()
        self.w_1 = nn.Linear(d_row, d_ff)
        self.w_2 = nn.Linear(d_ff, d_row)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = torch.t(x)
        d_k = x.size(-1)
        output = self.w_2(self.dropout(F.relu(self.w_1(x)))) / math.sqrt(d_k)
        output = F.softmax(output, dim=-1)
        if self.dropout is not None:
            output = self.dropout(output)

        return torch.t(output)

class Row_wise_nn(nn.Module):
    def __init__(self, d_column, d_ff, out_row, dropout=None):
        super(Row_wise_nn, self).__init__()
        self.w_1 = nn.Linear(d_column, d_ff)
        self.w_2 = nn.Linear(d_ff, out_row)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        d_k = x.size(-1)
        output = self.w_2(self.dropout(F.relu(self.w_1(x)))) / math.sqrt(d_k)
        output = F.softmax(output, dim=-1)
        if self.dropout is not None:
            output = self.dropout(output)

        return output




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

    def forward(self, x):
        left_transposer = self.row_wise_nn(x)
        middle_term = torch.dot(torch.t(left_transposer), x)
        right_transposer = self.column_wise_nn(middle_term)
        output = torch.dot(middle_term, torch.t(right_transposer))
        return output
