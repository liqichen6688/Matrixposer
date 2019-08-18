class Config(object):
    N = 1
    d_model = 512
    d_ff = 1024
    d_row = 60
    dropout = 0.1
    output_size = 4
    lr = 0.005
    max_epochs = 100
    batch_size = 64
    max_sen_len = 60
    # determine lr
    learning_method = 'reduce'
    # triangle learning
    cut_frac = 0.1
    ratio = 32
    max_lr = 0.01