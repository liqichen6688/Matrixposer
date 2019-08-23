class Config(object):
    N = 1
    d_model = 512
    d_ff = 1024
    d_row = 60
    dropout = 0.1
    output_size = 4
    lr = 0.001
    max_epochs = 200
    batch_size = 64
    max_sen_len = 60
    # determine lr
    learning_method = 'reduce'
    # triangle learning
    cut_frac = 0.1
    ratio = 32
    max_lr = 0.01
    '''
Final Training Accuracy: 0.9738
Final Validation Accuracy: 0.9132
Final Test Accuracy: 0.9128
'''