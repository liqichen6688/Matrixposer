import torch
import random
from torch import nn
from copy import deepcopy
from train_utils import Embeddings, PositionalEncoding, Matrix_Embedding
from interactor import Interactor
from encoder import EncoderLayer, Encoder, Decoder
from feed_forward import PositionwiseFeedForward
import torch.nn.functional as F
from utils import *
import numpy as np

class Matposer(nn.Module):
    def __init__(self, config, src_vocab, TEXT1, TEXT2, pretrain=False, dst_vocab = None):
        super(Matposer, self).__init__()
        self.step = 0
        self.config = config
        self.src_vocab = src_vocab
        d_row, N, dropout = self.config.d_row, self.config.N, self.config.dropout
        d_ff =  self.config.d_ff
        d_model1 = 300
        d_model2 = 50

        #inter = Interactor(d_model, d_ff, out_row=d_row, dropout=dropout, pretrain=config.pretrain)
        ff = PositionwiseFeedForward(d_model1, d_ff, dropout)
        position1 = PositionalEncoding(d_model1, dropout)
        position2 = PositionalEncoding(d_model2, dropout)


        #self.encoder = Encoder(EncoderLayer(d_model, deepcopy(inter), deepcopy(ff), dropout), N)
        #self.mappers = nn.ModuleList()
        #for one_encoder in self.encoder.layers:
        #    self.mappers.append(one_encoder.interactor.mapper)
        self.src_embed1 = nn.Sequential(
            Embeddings(d_model1, src_vocab, TEXT1), deepcopy(position1)
        )

        self.src_embed2 = nn.Sequential(
            Embeddings(d_model2, src_vocab, TEXT2), deepcopy(position2)
        )

        self.dst_embed = nn.Sequential(
            Embeddings(d_model2, dst_vocab), deepcopy(position2)
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
        self.decoder = Decoder(dst_vocab, d_model1)
        #self.matrix_embedding = Matrix_Embedding(d_model2,dst_vocab)
        self.ma_weight = nn.Parameter(torch.empty((d_model2, d_model2)).normal_(mean=0, std=0.0001))
        self.ma_bias = nn.Parameter(torch.empty((d_model2, d_model1)).normal_(mean=0, std=0.0001))


    def forward(self, x1, x2):
        embedded_sents1 = self.src_embed1(x1) # shape = (batch_size, sen_len, d_model)
        embedded_sents2 = self.src_embed2(x2)
        #encoded_sents = self.encoder(embedded_sents)
        encoded_sents = torch.matmul(embedded_sents2.permute(0,2,1), embedded_sents1)
        final_feature_map = encoded_sents
        #final_out = self.fc(final_feature_map)
        #class_out = self.class_fc(final_feature_map[:,-1,:])
        class_out = self.class_fc(final_feature_map.mean(dim=-2))
        if self.pretrain or self.config.translate:
            return final_feature_map / x1.shape[1]
        else:
            return self.softmax(class_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def loss_with_smoothing(self, pred, gold):
        gold = gold.contiguous().view(-1)
        #eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        #one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        #log_prb = F.log_softmax(pred, dim=1)
        #print(log_prb)

        non_pad_mask = gold.ne(2)
        #loss = self.loss_op(pred.masked_select(non_pad_mask), gold.masked_select(non_pad_mask))
        loss = -(one_hot * pred).sum(dim=1)
        #print(loss)
        #print(loss.masked_select(non_pad_mask))
        loss = loss.masked_select(non_pad_mask).sum()  # average later


        return loss

    def reduce_lr(self):
        for g in self.optimizer.param_groups:
            g['lr'] = 50 ** -0.5 * min(self.step ** -0.5, self.step * 2000 ** -1.5)

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
        #if self.config.learning_method == 'reduce':
        #    if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
        #        self.reduce_lr()
        for i, batch in enumerate(train_iterator):
            #self.step += 1
            #self.reduce_lr()
            if self.config.learning_method == 'trian':
                self.triangle_lr(len(train_iterator), epoch, i)
            self.optimizer.zero_grad()
            x1 = batch.text1.clone().permute(1, 0)
            x2 = batch.text2.clone().permute(1, 0)
            if not self.config.translate:
                if torch.cuda.is_available():
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    y = (batch.label - 1).type(torch.cuda.LongTensor)
                else:
                    y = (batch.label - 1).type(torch.LongTensor)

                y_pred = self.__call__(x1, x2)
                loss = self.loss_op(y_pred, y.cuda())
            else:
                x3 = batch.text3.clone().permute(1, 0)

                if torch.cuda.is_available():
                    x1 = x1.type(torch.cuda.LongTensor)
                    x2 = x2.type(torch.cuda.LongTensor)
                    x3 = x3.type(torch.cuda.LongTensor)
                loss = 0
                embed_matrix = self.__call__(x1, x2)
                embed_matrix += F.tanh(embed_matrix)

                x3_sent = self.dst_embed(x3)
                for j in range(1, x3.shape[1]):
                    #filter = self.matrix_embedding(x3[:, j - 1])
                    info_matrix = F.tanh(torch.matmul(self.ma_weight, embed_matrix) + self.ma_bias)
                    output = self.decoder(x3_sent[:, j-1:j].float(), info_matrix.float()).squeeze(1)
                    #print(output)
                    loss += self.loss_with_smoothing(output, x3[:, j].type(torch.cuda.LongTensor))
                    #loss += self.loss_op(output, x3[:,j].type(torch.cuda.LongTensor))
                    #print(self.loss_op(output, x3[:,j].type(torch.cuda.LongTensor)) / x3.shape[0])
                    #right, left = self.matrix_embedding(x3[:, i - 1])
                    #embed_matrix = torch.matmul(left.cuda(), (torch.matmul(embed_matrix, right.cuda())))
            #if self.pretrain:
            #    y = []
            #    delete_list = []
            #    for i in range(x.size()[0]):
            #        delete_ind = np.random.randint(0, x.size()[1])
            #        y.append(x[i, delete_ind])
            #        delete_list.append(delete_ind)
            #        if np.random.binomial(1, p=0.3) == 0:
            #            x[i, delete_ind] = 0
            #    y = torch.LongTensor(y)
            #    if torch.cuda.is_available():
            #        x1 = x1.type(torch.cuda.LongTensor)
            #        x2 = x2.type(torch.cuda.LongTensor)
            #        y = y.type(torch.cuda.LongTensor)
            #    else:
            #        x1 = x1.type(torch.LongTensor)
            #        x2 = x2.type(torch.LongTensor)
            #        y = y.type(torch.LongTensor)
            #    y_pred = self.softmax(self.fc(self.__call__(x)[list(range(0, x.size()[0])), delete_list, :]))
            #    loss = self.loss_op(y_pred, y.cuda())
            #else:
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
            losses.append(loss.data.cpu().numpy()/x3.shape[1]/x3.shape[0])
            self.optimizer.step()

            if i % 100 == 0:
                print("Iter: {}".format(i + 1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []

                # Evalute Accuracy on validation set
                #if not self.pretrain:
                #    val_accuracy = evaluate_model(self, val_iterator, self.config.translate)
                #    print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                #    val_accuracies.append(val_accuracy)
                self.train()
        return train_losses