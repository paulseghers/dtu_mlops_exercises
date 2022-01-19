import os
import pytest

import torch

from data.load_mnist import mnist
from model import MyAwesomeModel, mnist_classifier
from checkpoint_mgmt import load_checkpoint, save_checkpoint

#model based on: https://nextjournal.com/gkoehler/pytorch-mnist

import hydra
from hydra.utils import get_original_cwd
import logging


log = logging.getLogger(__name__)

def train(C):
    """
    input: config dictionary
    output: void
    train loop; can accept any torch.model, provided the
    appropriate data and hyperparameters are given
    """
    cfg =  C.train
    print("hyperparams: ", cfg.hyperparameters)
    print("workdir in train: ", os.getcwd(), "\n")
    log.info("Training homemade CNN classifier...")

    model = mnist_classifier()
    train_set, _ = mnist(get_original_cwd()+'/data/raw', cfg.hyperparameters.batch_size) 
    
    try:
        criterion = getattr(torch.nn, cfg.criterion)()
    except AttributeError:
        criterion = torch.nn.NLLLoss()

    try:
        optimizer = getattr(torch.optim, cfg.optimizer)(
                model.parameters(), cfg.hyperparameters.lr
                )
    except AttributeError:
        optimizer = torch.optim.Adam(model.parameters(), cfg.hyperparameters.lr)

    train_losses = []

    #training loop:
    for epoch in range(cfg.hyperparameters.epochs):
        running_loss = 0
        for images, labels in train_set:
            optimizer.zero_grad()
            log_ps = model(images.float())
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_losses.append(loss.item())
        print("Epoch: {}, Loss: {}".format(epoch, loss))

    #checkpoint should be created here
    save_checkpoint(get_original_cwd()+cfg.save_path, model)
    log.info("sucesfully trained & saved model at {}".format( get_original_cwd()+cfg.save_path))

  
def evaluate(C):
    """
    input: config dictionary
    output: void
    evaluation loop. loads model from saved path and 
    evaluates performance on test set.
    """
    cfg = C.evaluate
    log.info("Evaluating until hitting the ceiling")
    model = load_checkpoint(get_original_cwd()+cfg.load_path)
    log.info("loaded model from: {} ...".format(get_original_cwd()+cfg.load_path))
    _, test_loader = mnist(get_original_cwd()+'/data/raw', C.train.hyperparameters.batch_size)
    with torch.no_grad():
        model.eval()
        correct_preds, n_samples = 0, 0
        for images, labels in test_loader:
            ps = torch.exp(model.forward(torch.transpose(images.float(), 1, 2)))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            correct_preds += torch.sum(equals).item()
            n_samples += images.shape[0]

        accuracy = correct_preds / n_samples

    print('Accuracy of classifier: {}%'.format(accuracy*100))
    log.info('Accuracy of classifier: {}%'.format(accuracy*100))

@hydra.main(config_path="config", config_name='config_CNN.yaml')
def main(cfg):
    """ Helper class that will to launch train and test functions
    expects there to be a "command" field in the config file
    """
    try: 
        globals()[cfg.command]
    except AttributeError:
        print('Unrecognized command \'{}\''.format(cfg.command))
        exit(1)
    globals()[cfg.command](cfg)
    #log = logging.getLogger(__name__)
    #print("log, config: ", self.log, " ", self.config)
    #only needed here, as we can't execute a train then a test in a single run

if __name__ == '__main__':
    main()
    