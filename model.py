import torch
from torch import nn
import numpy as np
from utils import *

class Matposer(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings):
        super(Matposer, self).__init__()
        self.config = config