from model import *
from config_log.config import Config
import sys
import torch.optim as optim
from torch import nn
import torch

if __name__=='__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    config = Config
    train_file = '../data/20ng.train'
    #train_file = '../data/wiki/wiki_sentences.txt'
    #train_file = '20ng_sentences'
    if len(sys.argv) > 1:
        config = getattr(__import__(sys.argv[1], fromlist=["Config"]), "Config")
        print(sys.argv[1])
    if len(sys.argv) > 2:
        train_file = sys.argv[2]
    test_file = '../data/20ng.test'
    if len(sys.argv) > 3:
        test_file = sys.argv[3]




    dataset = Dataset(config)
    TEXT = dataset.load_data(train_file, test_file, config)

    pretrained_dict = torch.load('pretrain_model/wiki0')

    model = Matposer(config, len(dataset.vocab), TEXT, pretrain=config.pretrain)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        model = model.module
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    Loss = nn.NLLLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(Loss)

    #if not config.pretrain:
    #    model_dict = model.state_dict()
    #    print(model_dict.keys())
    #    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #    del pretrained_dict['class_fc.weight']
    #    del pretrained_dict['class_fc.bias']
    #    model_dict.update(pretrained_dict)
    #    model.load_state_dict(model_dict)



    train_losses = []
    val_accuracies = []

    for i in range(config.max_epochs):
        print("Epoch: {}".format(i))
        train_loss = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        #if val_accuracy > 0.772:
          #  break
        if config.pretrain:
            torch.save(model.state_dict(), "pretrain_model/wiki"+str(i))



