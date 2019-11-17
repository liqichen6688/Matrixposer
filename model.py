import torch
import random
from torch import nn
from copy import deepcopy
from train_utils import Embeddings, PositionalEncoding
from interactor import Interactor
from encoder import EncoderLayer, Encoder
from feed_forward import PositionwiseFeedForward
from utils import *
import numpy as np

class Matposer(nn.Module):
    def __init__(self, config, src_vocab, TEXT1, TEXT2, pretrain=False):
        super(Matposer, self).__init__()
        self.config = config
        self.src_vocab = src_vocab
        d_row, N, dropout = self.config.d_row, self.config.N, self.config.dropout
        d_ff =  self.config.d_ff
        d_model1 = 300
        d_model2 = 50

        #inter = Interactor(d_model, d_ff, out_row=d_row, dropout=dropout, pretrain=config.pretrain)
        ff = PositionwiseFeedForward(d_model1, d_ff, dropout)
        position1 = PositionalEncoding(d_model1, dropout)
        position2 = PositionalEncoding(d_model1, dropout)


        #self.encoder = Encoder(EncoderLayer(d_model, deepcopy(inter), deepcopy(ff), dropout), N)
        #self.mappers = nn.ModuleList()
        #for one_encoder in self.encoder.layers:
        #    self.mappers.append(one_encoder.interactor.mapper)
        self.src_embed1 = nn.Sequential(
            Embeddings(d_model1, src_vocab, TEXT1), deepcopy(position1)
        )

        self.src_embed2 = nn.Sequential(
            Embeddings(d_model1, src_vocab, TEXT1), deepcopy(position2)
        )

        a = nn.Linear(d_model1,d_model1)
        b = nn.ReLU()
        print(src_vocab)
        c = nn.Linear(d_model1, src_vocab)
        self.fc = nn.Sequential(a, b, c)

        self.class_fc = nn.Sequential(
            nn.Linear(d_model1,d_model1),nn.ReLU(), nn.Linear(d_model1,config.output_size)
        )


        self.softmax = nn.Softmax()

        self.pretrain = pretrain


    def forward(self, x1, x2):
        embedded_sents1 = self.src_embed1(x1) # shape = (batch_size, sen_len, d_model)
        embedded_sents2 = self.src_embed2(x2)
        #encoded_sents = self.encoder(embedded_sents)
        encoded_sents = torch.matmul(embedded_sents2.permute(0,2,1), embedded_sents1)
        final_feature_map = encoded_sents
        #final_out = self.fc(final_feature_map)
        #class_out = self.class_fc(final_feature_map[:,-1,:])
        class_out = self.class_fc(final_feature_map.diagonal())
        if self.pretrain:
            return final_feature_map
        else:
            return self.softmax(class_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def triangle_lr(self, total_iter, epoch, itr):
        cut = self.config.max_epochs * total_iter * self.config.cut_frac
        t = itr + epoch * total_iter
        if t < cut:
            p = t/cut
        else:
            p = 1 - (t - cut)/(cut * (1/self.config.cut_frac - 1))
        lr = self.config.max_lr * (1 + p * (self.config.ratio - 1))/(self.config.ratio)

        for g in self.optimizer.param_groups:
            g['lr'] = lr



    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []

        # Reduce learning rate as number of epochs increase
        if self.config.learning_method == 'reduce':
            if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
                self.reduce_lr()
        for i, batch in enumerate(train_iterator):
            if self.config.learning_method == 'trian':
                self.triangle_lr(len(train_iterator), epoch, i)
            self.optimizer.zero_grad()
            x1 = batch.text1.clone().permute(1, 0)
            x2 = batch.text2.clone().permute(1, 0)
            if self.pretrain:
                y = []
                delete_list = []
                for i in range(x.size()[0]):
                    delete_ind = np.random.randint(0, x.size()[1])
                    y.append(x[i, delete_ind])
                    delete_list.append(delete_ind)
                    if np.random.binomial(1, p=0.3) == 0:
                        x[i, delete_ind] = 0
                y = torch.LongTensor(y)
                if torch.cuda.is_available():
                    x1 = x1.type(torch.cuda.LongTensor)
                    x2 = x2.type(torch.cuda.LongTensor)
                    y = y.type(torch.cuda.LongTensor)
                else:
                    x1 = x1.type(torch.LongTensor)
                    x2 = x2.type(torch.LongTensor)
                    y = y.type(torch.LongTensor)
                y_pred = self.softmax(self.fc(self.__call__(x)[list(range(0, x.size()[0])), delete_list, :]))
                loss = self.loss_op(y_pred, y.cuda())
            else:
                if torch.cuda.is_available():
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    y = (batch.label - 1).type(torch.cuda.LongTensor)
                else:
                    y = (batch.label - 1).type(torch.LongTensor)

                y_pred = self.__call__(x1, x2)
                loss = self.loss_op(y_pred, y.cuda())
            #y = y.permute(1, 0)
            #y_onehot = torch.FloatTensor(y.size()[0], self.src_vocab)
            #y_onehot.zero_()
            #y_onehot.scatter_(1, y, 1)
            try:
                loss.backward()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

            if i % 100 == 0:
                print("Iter: {}".format(i + 1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []

                # Evalute Accuracy on validation set
                if not self.pretrain:
                    val_accuracy = evaluate_model(self, val_iterator)
                    print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                    val_accuracies.append(val_accuracy)
                self.train()

        return train_losses