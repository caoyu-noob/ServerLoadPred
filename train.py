from Dataset import Dataset
from model import LSTMModel
from tqdm import tqdm
from loss import MaskedMSELoss

import os
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
        self.dropout_rate = 0.2
        self.is_cuda_available = torch.cuda.is_available()
        self.model_save_path = 'model/lstm_hidden_'  + str(self.hidden_dim) + '/'
        self.train_data_path = 'dev_data.json'
        self.dev_data_path = 'dev_data.json'
        if debug:
            self.logger_interval = 1
            self.validation_interval = 1

def train(config):
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    dataset = Dataset(config.train_data_path, config.batch_size, config.max_len, config.is_cuda_available)
    dev_dataset = Dataset(config.dev_data_path, config.batch_size, config.max_len, config.is_cuda_available)
    dataset.new_epoch()
    model = LSTMModel(6, config.hidden_dim, 0.2)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.is_cuda_available:
        model = model.cuda()
    loss_function = MaskedMSELoss(config.is_cuda_available)
    epoch_num = config.epoch_num
    logger_interval = config.logger_interval
    validation_interval = config.validation_interval
    prev_best_loss = 100
    logger_count, validation_count = 0, 0
    for epoch in range(epoch_num):
        print('===================')
        print('Start Epoch {}'.format(epoch))
        loss_sum = 0
        batch_count = 0
        dataset.new_epoch()
        for d in dataset.generate_batches():
            model.train()
            train_seq = d[0]
            label = d[1]
            mask = d[2]
            optimizer.zero_grad()
            pred = model(train_seq)
            cur_batch_size = train_seq.batch_sizes[0].item()
            logger_count += cur_batch_size
            validation_count += cur_batch_size
            loss = loss_function(pred, label, mask)
            loss.backward()
            optimizer.step()
            loss_sum += loss.data.cpu().numpy()
            batch_count += 1
            logger_count, loss_sum, batch_count = _log_information(logger_count, logger_interval, loss_sum, batch_count)
            prev_best_loss, validation_count = _validation(validation_count, validation_interval,
                    config.model_save_path, dev_dataset, model, loss_function, prev_best_loss)
        print('Current loss is {:.5f}'.format(loss_sum / batch_count))
        print('End Epoch {}'.format(epoch))
        print('=====================')
        print('\n\n')
        torch.save(model.state_dict(), config.model_save_path + 'epoch' + str(epoch))

def _log_information(logger_count, logger_interval, loss_sum, batch_count):
    if logger_count >= logger_interval:
        print('Current loss is {:.5f}'.format(loss_sum / batch_count))
        logger_count, batch_count, loss_sum = 0, 0, 0
    return logger_count, loss_sum, batch_count

def _validation(validation_count, validation_interval, save_path, dev_dataset, model, loss_function, prev_best_loss):
    if validation_count  >= validation_interval:
        print('----------------------')
        print('Start Validation Stage')
        model.eval()
        dev_dataset.new_epoch()
        calculate_count = 0
        current_loss = 0
        for d in tqdm(dev_dataset.generate_batches()):
            dev_seq, label, mask = d[0], d[1], d[2]
            pred = model(dev_seq)
            loss = loss_function(pred, label, mask, is_sum=True)
            calculate_count += sum(mask).item()
            current_loss += loss.item()
        current_loss = current_loss / calculate_count * 100
        print('Current MSE in dev set is {:.5f}'.format(current_loss))
        if current_loss < prev_best_loss:
            print('Current model BEATS the previous best model, previous is {:.5f}'.format(prev_best_loss))
            prev_best_loss = current_loss
            torch.save(model.state_dict(), save_path + 'dev_best')
        else:
            print('Current model CANNOT beats the previous best model, previous is {:.5f}'.format(prev_best_loss))
        print('End Validation Stage')
        print('---------------------')
        validation_count = 0
    return prev_best_loss, validation_count

if __name__ == '__main__':
    config = Config()
    train(config)