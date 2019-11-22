import torch
from torchtext import data
from torchtext.vocab import Vectors
import pandas as pd
import numpy as np
import spacy
from torchtext.vocab import Vectors, GloVe
from sklearn.metrics import accuracy_score
import dill
import time

def get_embedding_matrix(vocab_chars):
    # return one hot emdding
    vocabulary_size = len(vocab_chars)
    onehot_matrix = np.eye(vocabulary_size, vocabulary_size)
    return onehot_matrix


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}

    def parse_label(self, label):
        '''
        Get the actual labels from label string
        Input:
            label (string) : labels of the form '__label__2'
        Returns:
            label (int) : integer value corresponding to label string
        '''
        if not isinstance(label, str):
            raise Exception(
                'type of label should be str. The type of label was {}'.format(
                    type(label)))
        begin = label.rfind('_') + 1

        return int(label.strip()[begin:])

    def get_pandas_df(self, filename, filename2 = None):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''
        if self.config.pretrain:
            with open(filename, 'r') as datafile:
                data = [line.strip() for line in datafile]
            full_df = pd.DataFrame({"text": data})
        elif self.config.translate:
            with open(filename, 'r') as datafile, open(filename2, 'r') as datafile2:
                ori_data = [line.strip() for line in datafile]
                data_text1 = list(map(lambda x: x, ori_data))
                data_text2 = list(map(lambda x: x, ori_data))
                dst_data = [line.strip() for line in datafile2]
                data_text3 = list(map(lambda x: x, dst_data))
                full_df = pd.DataFrame({"text1": data_text1, "text2": data_text2, "text3": data_text3})
        else:
            with open(filename, 'r') as datafile:
                data = [line.strip().split(',', maxsplit=1) for line in datafile]
                data_text1 = list(map(lambda x: x[1], data))
                data_text2 = list(map(lambda x: x[1], data))
                data_label = list(map(lambda x: self.parse_label(x[0]), data))

            full_df = pd.DataFrame({"text1": data_text1, "text2":data_text2,"label": data_label})

        return full_df


    def load_data(self, train_file, test_file, config,val_file=None, dst_file = None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data

        Inputs:
            train_file (String): absolute path to training file
            test_file (String): absolute path to test file
            val_file (String): absolute path to validation file
        '''
        # Loading Tokenizer
        NLP = spacy.load('en')


        def tokenizer(sent): return list(
            x.text for x in NLP.tokenizer(sent) if x.text != " ")

        # Creating Filed for data
        TEXT1 = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=None)
        TEXT2 = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=None)

        datafields = [("text1", TEXT1), ("text2", TEXT2)]

        if self.config.translate:
            NLP2 = spacy.load('de')

            def tokenizer2(sent): return list(
                x.text for x in NLP2.tokenizer(sent) if x.text != " ")

            TEXT3 = data.Field(sequential=True, tokenize=tokenizer2, lower=True, fix_length=None, init_token='<init>')
            datafields.append(("text3", TEXT3))
        if config.classification:
            LABEL = data.Field(sequential=False, use_vocab=False)
            datafields.append(("label",LABEL))

        # Load data from pd.DataFrame into torchtext.data.Dataset
        if not config.pretrain:
            train_df = self.get_pandas_df(train_file, filename2=dst_file)
            train_examples = [
                data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
            train_data = data.Dataset(train_examples, datafields)
        else:
            #train_df = pd.read_csv("../data/wiki/data/ruwiki_2018_09_25.csv")['text']
            #print("storing_training_csv")
            #train_df.to_csv("../data/wiki/data/train.csv", index=False)
            print("loading_training_data")
            train_data = data.TabularDataset(path='../data/wiki/data/train_filt.csv', format='CSV', fields=datafields)
            print("done...")





        if not config.pretrain and not config.translate:
            test_df = self.get_pandas_df(test_file)
            test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
            test_data = data.Dataset(test_examples, datafields)



        # If validation file exists, load it. Otherwise get validation data
        # from training data
        if val_file != None:
            val_df = self.get_pandas_df(val_file)
            val_examples = [
                data.Example.fromlist(
                    i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            print('right!')
            train_data, val_data = train_data.split(split_ratio=0.5)


        TEXT1.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
        TEXT2.build_vocab(train_data, vectors=GloVe(name='6B', dim=50))
        if self.config.translate:
            TEXT3.build_vocab(train_data)
            #with open("pretrain_model/build_vocab", "wb") as dill_file:
            #    dill.dump(TEXT, dill_file)
            #    print("vocab saved")


        self.vocab1 = TEXT1.vocab
        self.vocab2 = TEXT2.vocab
        self.vocab3 = TEXT3.vocab

        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True
        )

        if not config.pretrain and not config.translate:
            self.val_iterator, self.test_iterator = data.BucketIterator(
                (val_data, test_data),
                batch_size=self.config.batch_size,
                sort_key=lambda x: len(x.text),
                repeat=False,
                shuffle=False)
        else:
            self.val_iterator = data.BucketIterator(
                (val_data),
                batch_size=self.config.batch_size,
                sort_key=lambda x: len(x.text),
                repeat=False,
                shuffle=False)


        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} validation examples".format(len(val_data)))


        return TEXT1, TEXT2


def evaluate_model(model, iterator, is_translate):
    all_preds = []
    all_y = []
    for idx, batch in enumerate(iterator):
        if torch.cuda.is_available():
            x1 = batch.text1.permute(1, 0).cuda()
            x2 = batch.text2.permute(1, 0).cuda()
        else:
            x1 = batch.text1.permute(1, 0)
            x2 = batch.text2.permute(1, 0)
        y_pred = model(x1, x2)
        if not is_translate:
            predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
            all_preds.extend(predicted.numpy())
            all_y.extend(batch.label.numpy())
        else:
            x3 = batch.text3.permute(1, 0).cuda()
            embed_matrix = y_pred
            x3_sent = model.dst_embed(x3)
            for i in range(1, x3.shape[1]):
                output = model.decoder(x3_sent[:, i-1], embed_matrix)
                print(output.cpu().max(1))
                all_preds.extend(output.cpu().max(1).numpy())
                all_y.extend(x3[:, i-1].cpu().numpy())
                right, left = model.matrix_embedding(x3[:, i - 1])
                embed_matrix = torch.matmul(left, (torch.matmul(embed_matrix, right)))
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score
