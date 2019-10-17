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

    def get_pandas_df(self, filename):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''
        if self.config.pretrain:
            with open(filename, 'r') as datafile:
                data = [line.strip() for line in datafile]
            full_df = pd.DataFrame({"text": data})
        else:
            with open(filename, 'r') as datafile:
                data = [line.strip().split(',', maxsplit=1) for line in datafile]
                data_text = list(map(lambda x: x[1], data))
                data_label = list(map(lambda x: self.parse_label(x[0]), data))

            full_df = pd.DataFrame({"text": data_text, "label": data_label})

        return full_df


    def load_data(self, train_file, test_file, config,val_file=None):
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
        #if config.pretrain:
        #    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        #else:
        with open("pretrain_model/build_vocab", "rb") as dill_file:
            TEXT = dill.load(dill_file)
            print("vocab loaded")
        datafields = [("text", TEXT)]
        if not config.pretrain:
            LABEL = data.Field(sequential=False, use_vocab=False)
            datafields.append(("label",LABEL))

        # Load data from pd.DataFrame into torchtext.data.Dataset
            train_df = self.get_pandas_df(train_file)
            train_examples = [
                data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
            train_data = data.Dataset(train_examples, datafields)
        else:
            #train_df = pd.read_csv("../data/wiki/data/ruwiki_2018_09_25.csv")['text']
            #print("storing_training_csv")
            #train_df.to_csv("../data/wiki/data/train.csv", index=False)
            print("loading_training_data")
            train_data = data.TabularDataset(train='../data/wiki/data/train.csv',format='csv', fields=datafields)
            print("done...")





        if not config.pretrain:
            test_df = self.get_pandas_df(test_file)
            test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
            test_data = data.Dataset(test_examples, datafields)



        # If validation file exists, load it. Otherwise get validation data
        # from training data
        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_examples = [
                data.Example.fromlist(
                    i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)


        if config.pretrain:
            TEXT.build_vocab(train_data, vectors=GloVe(name='840B', dim=config.d_model), max_size = 45000)
            with open("pretrain_model/build_vocab", "wb") as dill_file:
                dill.dump(TEXT, dill_file)
                print("vocab saved")


        self.vocab = TEXT.vocab
        print(self.vocab.itos[:3])

        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True
        )

        if not config.pretrain:
            self.val_iterator, self.test_iterator = data.BucketIterator.splits(
                (val_data, test_data),
                batch_size=self.config.batch_size,
                sort_key=lambda x: len(x.text),
                repeat=False,
                shuffle=False)
        else:
            self.val_iterator = data.BucketIterator.splits(
                (val_data),
                batch_size=self.config.batch_size,
                sort_key=lambda x: len(x.text),
                repeat=False,
                shuffle=False)


        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} validation examples".format(len(val_data)))

        return TEXT


def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx, batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
        else:
            x = batch.text
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score
