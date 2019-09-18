import torch
from torch import nn
from copy import deepcopy
from train_utils import Embeddings, PositionalEncoding
from interactor import Interactor
from encoder import EncoderLayer, Encoder
from feed_forward import PositionwiseFeedForward
from utils import *

class Matposer(nn.Module):
    def __init__(self, config, src_vocab, TEXT):
        super(Matposer, self).__init__()
        self.config = config

        d_row, N, dropout = self.config.d_row, self.config.N, self.config.dropout
        d_model, d_ff = self.config.d_model, self.config.d_ff

        inter = Interactor(d_model, d_ff, out_row=d_row, dropout=dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        self.encoder = Encoder(EncoderLayer(d_model, deepcopy(inter), deepcopy(ff), dropout), N)
        self.mappers = nn.ModuleList()
        for one_encoder in self.encoder.layers:
            self.mappers.append(one_encoder.interactor.mapper)
        self.src_embed = nn.Sequential(
            Embeddings(d_model, src_vocab, TEXT), deepcopy(position)
        )

        self.fc = nn.Linear(
            d_model,
            self.config.output_size
        )

        self.softmax = nn.Softmax()

    def forward(self, x):
        embedded_sents = self.src_embed(x.permute(1, 0)) # shape = (batch_size, sen_len, d_model)
        encoded_sents = self.encoder(embedded_sents)
        final_feature_map = encoded_sents[:,-1,:]
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)

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
        for j in range(2):
            if j == 0:
                print('training on weights')
                for param in self.parameters():
                    param.requires_grad = True
                for mapper in self.mappers:
                    mapper.freeze_parameter()
            else:
                print('training on mappers')
                for param in self.parameters():
                    param.requires_grad = False
                for mapper in self.mappers:
                    mapper.unfreeze_parameter()
                    mapper.renew_mask()
            for i, batch in enumerate(train_iterator):

                if self.config.learning_method == 'trian':
                    self.triangle_lr(len(train_iterator), epoch, i)
                self.optimizer.zero_grad()
                if torch.cuda.is_available():
                    x = batch.text.cuda()
                    y = (batch.label - 1).type(torch.cuda.LongTensor)
                else:
                    x = batch.text
                    y = (batch.label - 1).type(torch.LongTensor)
                y_pred = self.__call__(x)
                loss = self.loss_op(y_pred, y)
                loss.backward()
                losses.append(loss.data.cpu().numpy())
                self.optimizer.step()

                if i % 100 == 0:
                    print("Iter: {}".format(i + 1))
                    avg_train_loss = np.mean(losses)
                    train_losses.append(avg_train_loss)
                    print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                    losses = []

                    # Evalute Accuracy on validation set
                    val_accuracy = evaluate_model(self, val_iterator)
                    print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                    val_accuracies.append(val_accuracy)
                    self.train()

        return train_losses, np.mean(val_accuracies)