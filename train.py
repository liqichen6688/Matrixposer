from model import *
from config_log.config import Config
import sys
import torch.optim as optim
from torch import nn
import torch

if __name__=='__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    config = Config
    #train_file = '../data/20ng.train'
    #train_file = '../data/wiki/wiki_sentences.txt'
    #train_file = '20ng_sentences'
    train_file = '../data/translate/English-German/en_simple'
    dst_file = '../data/translate/English-German/ger_simple'
    if len(sys.argv) > 1:
        config = getattr(__import__(sys.argv[1], fromlist=["Config"]), "Config")
        print(sys.argv[1])
    if len(sys.argv) > 2:
        train_file = sys.argv[2]
    test_file = '../data/ag_news.test'
    if len(sys.argv) > 3:
        test_file = sys.argv[3]




    dataset = Dataset(config)
    TEXT1, TEXT2 = dataset.load_data(train_file, test_file, config, dst_file=dst_file)

    if config.translate:
        model = Matposer(config, len(dataset.vocab1), TEXT1, TEXT2, pretrain=config.pretrain, dst_vocab=len(dataset.vocab3))
    else:
        model = Matposer(config, len(dataset.vocab1), TEXT1, TEXT2, pretrain=config.pretrain)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        model = model.module
    torch.cuda.empty_cache()
    model.to(device)
    #torch.cuda.empty_cache()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr,  betas=(0.9, 0.98), eps=1e-09)
    Loss = nn.NLLLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(Loss)


    #if not config.pretrain:
    #    model_dict = model.state_dict()
    #    pretrained_dict = torch.load('pretrain_model/wiki_false0')
    #    print(model_dict.keys())
    #    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #    del pretrained_dict['class_fc.0.weight']
    #    del pretrained_dict['class_fc.0.bias']
    #    del pretrained_dict['class_fc.1.weight']
    #    del pretrained_dict['class_fc.1.bias']
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
        #if config.pretrain:
        torch.save(model.state_dict(), "translate_par/translation"+str(i))



