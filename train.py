from Dataset import Dataset
from model import LSTMModel
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim

class Config:
    def __init__(self, debug=False):
        self.batch_size = 32
        self.max_len = 7000
        self.epoch_num = 20
        self.logger_interval = 1000
        self.validation_interval = 5000
        self.hidden_dim = 200
        self.learning_rate = 0.01
        self.is_cuda_available = torch.cuda.is_available()
        if debug:
            self.logger_interval = 1
            self.validation_interval = 1

def train(config):
    dataset = Dataset('dev_data.json', config.batch_size, config.max_len, config.is_cuda_available)
    dev_dataset = Dataset('dev_data.json', config.batch_size, config.max_len, config.is_cuda_available)
    dataset.new_epoch()
    model = LSTMModel(6, config.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.is_cuda_available:
        model = model.cuda()
    loss_function = nn.MSELoss()
    epoch_num = config.epoch_num
    logger_interval = config.logger_interval
    validation_interval = config.validation_interval
    logger_count, validation_count = 0, 0
    for epoch in range(epoch_num):
        print('===================')
        print('Start Epoch {}'.format(epoch))
        loss_sum = 0
        batch_count = 0
        dataset.new_epoch()
        for d in dataset.generate_batches():
            train_seq = d[0].permute(1, 0, 2)
            label = d[1]
            optimizer.zero_grad()
            pred = model(train_seq)
            cur_batch_size = train_seq.shape[1]
            logger_count += cur_batch_size
            validation_count += cur_batch_size
            loss = loss_function(pred, label)
            loss.backward()
            optimizer.step()
            loss_sum += loss.data.cpu().numpy()
            batch_count += 1
            logger_count, loss_sum, batch_count = __log_information(logger_count, logger_interval, loss_sum, batch_count)

        print('Current loss is {:.3f}'.format(loss_sum / batch_count))
        print('End Epoch {}'.format(epoch))
        print('=====================')
        print('\n\n')
        torch.save(model.state_dict(), 'model/lstm/epoch' + str(epoch))

def __log_information(logger_count, logger_interval, loss_sum, batch_count):
    if logger_count >= logger_interval:
        print('Current loss is {:.3f}'.format(loss_sum / batch_count))
        logger_count, batch_count, loss_sum = 0, 0, 0
    return logger_count, loss_sum, batch_count

def __validation(validation_count, validation_interval, dev_dataset, model):
    if validation_count  >= validation_interval:
        print('----------------------')
        print('Start Validation Stage')
        dev_dataset.new_epoch()
        for d in tqdm(dev_dataset.generate_batches()):
            training_seq, label = d[0].permute(1, 0, 2)
            pred = model()
    return validation_count

if __name__ == '__main__':
    config = Config(debug=True)
    train(config)