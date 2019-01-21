from Dataset import Dataset
from model import LSTMModel
from tqdm import tqdm
from loss import MaskedMSELoss, MaskedRMSELoss, MaskedMAEAndMAPELoss

import os
import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim
from ConfigLogger import config_logger

class Config:
    def __init__(self, debug=False):
        self.batch_size = 32
        self.max_len = 7000
        self.epoch_num = 30
        self.logger_interval = 1000
        self.validation_interval = 5000
        self.hidden_dim = 32
        self.learning_rate = 0.002
        self.dropout_rate = 0
        self.is_cuda_available = torch.cuda.is_available()
        self.model_save_path = 'model/lstm_hidden_'  + str(self.hidden_dim) + '/'
        self.logger = config_logger(self.model_save_path)
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
    logger = config.logger
    model = LSTMModel(6, config.hidden_dim, config.dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.is_cuda_available:
        model = model.cuda()
    loss_function = MaskedMSELoss(config.is_cuda_available)
    eval_rmse_loss_function = MaskedRMSELoss(config.is_cuda_available)
    eval_mae_loss_function = MaskedMAEAndMAPELoss(config.is_cuda_available)
    epoch_num = config.epoch_num
    logger_interval = config.logger_interval
    validation_interval = config.validation_interval
    prev_best_loss = 10000
    logger_count, validation_count = 0, 0
    for epoch in range(epoch_num):
        logger.info('===================')
        logger.info('Start Epoch %d', epoch)
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
            logger_count, loss_sum, batch_count = _log_information(logger, logger_count, logger_interval, loss_sum,
                    batch_count)
            prev_best_loss, validation_count = _validation(logger, validation_count, validation_interval,
                    config.model_save_path, dev_dataset, model, loss_function, eval_rmse_loss_function,
                    eval_mae_loss_function, prev_best_loss)
        logger.info('Current loss is %.5f', loss_sum / batch_count)
        logger.info('End Epoch %d', epoch)
        logger.info('=====================')
        logger.info('\n\n')
        torch.save(model.state_dict(), config.model_save_path + 'epoch' + str(epoch))

def _log_information(logger, logger_count, logger_interval, loss_sum, batch_count):
    if logger_count >= logger_interval:
        logger.info('Current loss is %.5f', loss_sum / batch_count)
        logger_count, batch_count, loss_sum = 0, 0, 0
    return logger_count, loss_sum, batch_count

def _validation(logger, validation_count, validation_interval, save_path, dev_dataset, model, loss_function,
        rmse_loss_function, mae_loss_function, prev_best_loss):
    if validation_count  >= validation_interval:
        logger.info('----------------------')
        logger.info('Start Validation Stage')
        model.eval()
        dev_dataset.new_epoch()
        calculate_count = 0
        current_loss = 0
        current_rmse_loss = 0
        current_mae_loss = 0
        current_mape_loss =0
        sample_count = 0
        sample_mape_count = 0
        for d in tqdm(dev_dataset.generate_batches()):
            dev_seq, label, mask = d[0], d[1], d[2]
            pred = model(dev_seq)
            loss = loss_function(pred, label, mask)
            rmse_loss = rmse_loss_function(pred, label, mask)
            mae_loss, mape_loss, mape_count = mae_loss_function(pred, label, mask)
            current_loss += loss.item() * mask.shape[0]
            sample_count += mask.shape[0]
            sample_mape_count += mape_count
            current_rmse_loss += rmse_loss.item() * mask.shape[0]
            current_mae_loss += mae_loss.item()
            current_mape_loss += mape_loss.item()
        current_loss = current_loss / sample_count
        logger.info('Current RMSE in dev set is %.5f', current_rmse_loss / sample_count)
        logger.info('Current MAE in dev set is %.5f', current_mae_loss / sample_count)
        logger.info('Current MAPE in dev set is %.5f', current_mape_loss / sample_count)
        if current_loss < prev_best_loss:
            logger.info('Current model BEATS the previous best model, previous best loss is %.5f, current loss is %.5f',
                prev_best_loss, current_loss)
            prev_best_loss = current_loss
            torch.save(model.state_dict(), save_path + 'dev_best')
        else:
            logger.info('Current model CANNOT beats the previous best model, previous is %.5f, current loss is %.5f',
                prev_best_loss, current_loss)
        logger.info('End Validation Stage')
        logger.info('---------------------')
        validation_count = 0
    return prev_best_loss, validation_count

if __name__ == '__main__':
    config = Config()
    train(config)