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
    train_file = '../data/mr.train'
    if len(sys.argv) > 1:
        config = getattr(__import__(sys.argv[1], fromlist=["Config"]), "Config")
        print(sys.argv[1])
    if len(sys.argv) > 2:
        train_file = sys.argv[2]
    test_file = '../data/mr.test'
    if len(sys.argv) > 3:
        test_file = sys.argv[3]




    dataset = Dataset(config)
    TEXT = dataset.load_data(train_file, test_file, config)

    model = Matposer(config, len(dataset.vocab), TEXT)
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

    train_losses = []
    val_accuracies = []

    for i in range(config.max_epochs):
        print("Epoch: {}".format(i))
        train_loss, val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        if val_accuracy > 0.772:
            break

    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)
    test_acc = evaluate_model(model, dataset.test_iterator)

    print('Final Training Accuracy: {:.4f}'.format(train_acc))
    print('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print('Final Test Accuracy: {:.4f}'.format(test_acc))


