class Config(object):
    N = 1
    d_model = 768
    d_ff = 1024
    d_row = 60
    dropout = 0.1
    output_size = 4
    lr = 0.005
    max_epochs = 50
    batch_size = 32
    max_sen_len = 60
    # determine lr
    learning_method = 'trian'
    # triangle learning
    cut_frac = 0.1
    ratio = 32
    max_lr = 0.01

'''
Iter: 2901
        Average training loss: -0.92257
        Val Accuracy: 0.9021
'''