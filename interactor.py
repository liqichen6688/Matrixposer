import torch
from torch.nn.parameter import Parameter
from sublayer import SublayerOutput
import torch.nn.functional as F
import math
from torch import nn
from torch.autograd import Variable
from train_utils import clones
import numpy as np

class Column_wise_nn(nn.Module):
    def __init__(self, d_row, d_ff, d_out,dropout=0.1):
        '''
        initialize column-wise neural network
        :param d_row: input row number
        :param d_ff: middle size row number
        :param dropout: default None
        '''
        super(Column_wise_nn, self).__init__()
        self.w_1 = nn.Linear(d_row, d_ff)
        self.w_2 = nn.Linear(d_ff, d_ff)
        self.w_3 = nn.Linear(d_ff, d_out)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = x.permute(0,2,1)
        d_k = x.size(-1)
        #output = self.w_2(self.dropout(F.relu(self.w_1(x)))) / math.sqrt(d_k)
        #output = F.softmax(output, dim=-1)
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        output = self.w_3(self.dropout(F.relu(output))) / math.sqrt(d_k)
        if self.dropout is not None:
            output = self.dropout(output)

        return output.permute(0,2,1)


class Row_wise_nn(nn.Module):
    def __init__(self, d_column, d_ff, out_row, dropout=None, softmax = False):
        super(Row_wise_nn, self).__init__()
        self.w_1 = nn.Linear(d_column, d_ff)
        self.w_2 = nn.Linear(d_ff, d_ff)
        self.w_3 = nn.Linear(d_ff, out_row)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.softmax = softmax

    def forward(self, x):
        d_k = x.size(-1)
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        output = self.w_3(self.dropout(F.relu(output))) / math.sqrt(d_k)
        if self.softmax:
            output = F.softmax(output, dim=-1)
        if self.dropout is not None:
            output = self.dropout(output)

        return output


class Mapper(nn.Module):
    def __init__(self, d_row, d_column, map_size=None,out_row=None):
        super(Mapper, self).__init__()
        if map_size == None:
            self.map_size = d_row
        else:
            self.map_size = map_size
        if out_row == None:
            self.out_row = d_column
        else:
            self.ou  = out_row
        self.d_all_input = d_all_input =d_row * d_column

        self.all_filters = nn.ParameterList()
        for _ in range(self.out_row):
            self.all_filters.append(Parameter(torch.rand(d_all_input)/2, requires_grad=True))

        self.indlist = []
        self.renew_mask()
        self.freeze = False

    def forward(self, x):
        x = x.view(-1, self.d_all_input)
        output = []
        for i in range(self.out_row):
            ind = self.indlist[i]
            if self.freeze == True:
                one_out = x[:,ind]
            else:
                one_out = x[:, ind] * self.all_filters[i][ind]
            one_out = one_out.unsqueeze(2)
            output.append(one_out)
        output = torch.cat(output, 2)
        return output


    def renew_mask(self):
        for i in range(self.out_row):
            filter = np.array(self.all_filters[i].tolist())
            ind = np.argpartition(filter, -self.map_size)[-self.map_size:]
            ind = ind[np.argsort(filter[ind])].tolist()
            self.indlist.append(ind)


    def freeze_parameter(self):
        for param in self.parameters():
            param.requires_grad = False

        self.freeze = True

    def unfreeze_parameter(self):
        for param in self.parameters():
            param.requires_grad = True

        self.freeze = False

class Interactor(nn.Module):
    def __init__(self, d_column, d_ff, out_row=30, dropout=0.1):
        '''
        :param d_row: dimension of output row number
        :param d_column: dimension of input column number
        :param d_ff: dimension of middle neural
        :param dropout: default 0.1
        '''
        super(Interactor, self).__init__()
        self.column_wise_nn1 = Column_wise_nn(2 * out_row, d_ff, 1, dropout)
        self.row_wise_nn1 = Row_wise_nn(d_column, d_ff, out_row, dropout)
        self.row_wise_nn2 = Row_wise_nn(d_column, d_ff, out_row, dropout)
        self.mapper = Mapper(out_row, d_column, map_size = 2 * out_row)

    def forward(self, x):
        left_transposer1 = self.row_wise_nn1(x)
        output1 = torch.matmul(left_transposer1.permute(0,2,1), x)
        left_transposer2 = self.row_wise_nn2(output1)
        output2 = torch.matmul(left_transposer2.permute(0, 2, 1), output1)
        output = self.mapper(output2)
        output = self.column_wise_nn1(output)
        #output = self.column_wise_nn(outp
        #ut)
        #output = torch.matmul(middle_term, right_transposer.permute(0,2,1))
        return output

