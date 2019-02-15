from Dataset import Dataset, DatasetWindow
from model import LSTMModel, GRUModel, EncoderDecoderModel, RNNDAModel, RNNDAModel_Att1, RNNDAModel_Att2, RNNDAModel_Att
from tqdm import tqdm
from loss import MaskedMSELoss, MaskedRMSELoss, MaskedMAEAndMAPELoss

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from ConfigLogger import config_logger

model_dict = {'gru' : GRUModel, 'lstm' : LSTMModel, 'encoder' : EncoderDecoderModel, 'rnnda' : RNNDAModel,
              'rnnda-att1':RNNDAModel_Att1, 'rnnda-att2':RNNDAModel_Att2, 'rnnda-att':RNNDAModel_Att}

class Config:
    def __init__(self, debug=False):
        self.batch_size = 256
        self.max_len = 7000
        self.epoch_num = 10
        self.logger_interval = 1000
        self.validation_interval = 5000
        self.hidden_dim = 256
        self.learning_rate = 0.001
        self.dropout_rate = 0
        self.is_cuda_available = torch.cuda.is_available()
        self.train_data_path = '../train_data'
        self.dev_data_path = 'dev_data.json'
        self.test_data_path = 'test_data.json'
        self.test_batch_size = 512
        self.dynamic_lr = True
        self.lr_change_batch_interval = 100000
        self.optimizer_type = 'adam'
        self.use_net_accumulation = False
        self.use_window = True
        self.window_size = 8
        self.model_type = 'rnnda-att'
        self.model_save_path = 'model/' + self.model_type + '_hidden_' + str(self.hidden_dim) + '/'
        if self.use_window:
            self.logger_interval = self.logger_interval * 6000
            self.validation_interval = self.validation_interval * 6000
            self.model_save_path = 'model/window' + str(self.window_size) + '_' + self.model_type + '_hidden_' + \
                                   str(self.hidden_dim) + '/'
        self.continue_train =False
        if self.continue_train:
            self.prev_model_name = 'epoch0'
        if debug:
            self.logger_interval = 1
            self.validation_interval = 1
            self.epoch_num = 1
        self.logger = config_logger(self.model_save_path)

def adjust_learning_rate(config, optimizer, logger):
    config.learning_rate = config.learning_rate * 0.8
    for param_group in optimizer.param_groups:
        param_group['lr'] = config.learning_rate
    logger.info('Change learning rate as %.5f', config.learning_rate)

def test(config):
    test_data = Dataset(config.test_data_path, 1, config.max_len, config.is_cuda_available, use_splitted_files=False)
    logger = config.logger
    if config.model_type == 'rnnda':
        model = model_dict[config.model_type](6, config.hidden_dim, config.window_size)
    else:
        model = model_dict[config.model_type](6, config.hidden_dim, config.dropout_rate, config.use_window)
    if config.is_cuda_available:
        model = model.cuda()
    model.load_state_dict(torch.load(config.model_save_path + 'dev_best'))
    loss_function = MaskedMSELoss(config.is_cuda_available, config.use_window)
    eval_rmse_loss_function = MaskedRMSELoss(config.is_cuda_available)
    eval_mae_loss_function = MaskedMAEAndMAPELoss(config.is_cuda_available)
    prev_best_loss, validation_count = _validation(config, logger, 0, test_data, model, loss_function,
            eval_rmse_loss_function, eval_mae_loss_function, 0, is_test=True)

def train(config):
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if config.use_window:
        dataset = DatasetWindow(config.train_data_path, config.batch_size, config.max_len, config.is_cuda_available,
            config.window_size, True, use_net_accumulation=config.use_net_accumulation)
        dev_dataset = Dataset(config.dev_data_path, 1, config.max_len, config.is_cuda_available, False,
            use_net_accumulation=config.use_net_accumulation)
        # dataset = DatasetWindow(config.test_data_path, config.batch_size, config.max_len, config.is_cuda_available,
        #     config.window_size, False, use_net_accumulation=config.use_net_accumulation)
    else:
        dataset = Dataset(config.train_data_path, config.batch_size, config.max_len, config.is_cuda_available,
                True, use_net_accumulation=config.use_net_accumulation)
        dev_dataset = Dataset(config.dev_data_path, config.batch_size, config.max_len, config.is_cuda_available,
                False, use_net_accumulation=config.use_net_accumulation)
    loss_function = MaskedMSELoss(config.is_cuda_available, config.use_window)
    logger = config.logger
    if config.model_type != 'rnnda' and config.model_type != 'rnnda-att1' and config.model_type != 'rnnda-att2' \
            and config.model_type != 'rnnda-att':
        model = model_dict[config.model_type](6, config.hidden_dim, config.dropout_rate, config.use_window)
    else:
        model = model_dict[config.model_type](6, config.hidden_dim, config.window_size)
    if config.continue_train:
        model.load_state_dict(torch.load(config.model_save_path + '/' + config.prev_model_name))
    if config.is_cuda_available:
        model = model.cuda()
    optimizer_dict = {'sgd': optim.SGD, 'adam': optim.Adam}
    optim_func = optimizer_dict[config.optimizer_type]
    optimizer = optim_func(model.parameters(), lr=config.learning_rate)
    eval_rmse_loss_function = MaskedRMSELoss(config.is_cuda_available, use_window=True)
    eval_mae_loss_function = MaskedMAEAndMAPELoss(config.is_cuda_available, use_window=True)
    epoch_num = config.epoch_num
    logger_interval = config.logger_interval
    prev_best_loss = 10000
    logger_count, validation_count = 0, 0
    batch_count = 1
    for epoch in range(epoch_num):
        logger.info('===================')
        logger.info('Start Epoch %d', epoch)
        loss_sum = 0
        dataset.new_epoch()

        # objgraph.show_growth()
        for d in tqdm(dataset.generate_batches()):
            # objgraph.show_growth()
            if config.dynamic_lr and batch_count % config.lr_change_batch_interval == 0:
                adjust_learning_rate(config, optimizer, logger)
            model.train()
            train_seq = d[0]
            label = d[1]
            mask = d[2]
            if train_seq is None:
                continue
            optimizer.zero_grad()
            pred = model(train_seq)
            cur_batch_size = train_seq.batch_sizes[0].item()
            logger_count += cur_batch_size
            validation_count += cur_batch_size
            # loss = nn.functional.mse_loss(pred, label)
            loss = loss_function(pred, label, mask, use_window=config.use_window)
            loss.backward()
            optimizer.step()
            loss_sum += loss.data.cpu().numpy()
            batch_count += 1
            logger_count, loss_sum, batch_count = _log_information(logger, logger_count, logger_interval, loss_sum,
                    batch_count)
            prev_best_loss, validation_count = _validation(config, logger, validation_count,
                    dev_dataset, model, loss_function, eval_rmse_loss_function, eval_mae_loss_function, prev_best_loss)
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

def _validation(config, logger, validation_count, dev_dataset, model, loss_function, rmse_loss_function,
            mae_loss_function, prev_best_loss, is_test=False):
    if validation_count >= config.validation_interval or is_test:
        logger.info('----------------------')
        logger.info('Start Validation Stage')
        model.eval()
        dev_dataset.new_epoch()
        current_loss = 0
        current_rmse_loss = 0
        current_mae_loss = 0
        current_mape_loss =0
        sample_count = 0
        use_pack = not config.use_window
        for d in tqdm(dev_dataset.generate_batches(use_pack=use_pack)):
            dev_seq, label, mask = d[0], d[1], d[2]
            batch_size = dev_seq.shape[0]
            if config.use_window:
                dev_seq = dev_seq[0]
                label = label[0][config.window_size - 1:]
                pred = None
                for index in range(config.window_size, dev_seq.shape[0] + 1, config.test_batch_size):
                    cur_batch_seq_data = dev_seq[index - config.window_size:index + config.test_batch_size - 1]
                    cur_batch_size = cur_batch_seq_data.shape[0] - config.window_size + 1
                    seq_batch = torch.zeros(cur_batch_size, config.window_size, dev_seq.shape[-1])
                    if config.is_cuda_available:
                        seq_batch = seq_batch.cuda()
                    for i in range(cur_batch_size):
                        seq_batch[i] = cur_batch_seq_data[i:i + config.window_size]
                    seq_lens = [config.window_size] * cur_batch_size
                    seq_batch = nn.utils.rnn.pack_padded_sequence(seq_batch, seq_lens, batch_first=True)
                    cur_pred = model(seq_batch)
                    if pred is None:
                        pred = cur_pred
                    else:
                        pred = torch.cat((pred, cur_pred), 0)
                label = label.unsqueeze(1)
                mask = [1] * (mask - config.window_size + 1).item()
                batch_size = 1
            else:
                pred = model(dev_seq)
            if pred.shape[0] != label.shape[0]:
                print('aaa')
            loss = loss_function(pred, label, mask, use_window=True)
            rmse_loss = rmse_loss_function(pred, label, mask)
            mae_loss, mape_loss = mae_loss_function(pred, label, mask)
            current_loss += loss.item() * batch_size
            sample_count += batch_size
            current_rmse_loss += rmse_loss.item() * batch_size
            current_mae_loss += mae_loss
            if mape_loss != 0:
                current_mape_loss += mape_loss
        current_loss = current_loss / sample_count
        logger.info('Current RMSE in dev set is %.5f', current_rmse_loss / sample_count)
        logger.info('Current MAE in dev set is %.5f', current_mae_loss / sample_count)
        logger.info('Current MAPE in dev set is %.5f', current_mape_loss / sample_count)
        if not is_test:
            if current_loss < prev_best_loss:
                logger.info('Current model BEATS the previous best model, previous best loss is %.5f, current loss is %.5f',
                    prev_best_loss, current_loss)
                prev_best_loss = current_loss
                torch.save(model.state_dict(), config.model_save_path + 'dev_best')
            else:
                logger.info('Current model CANNOT beats the previous best model, previous is %.5f, current loss is %.5f',
                    prev_best_loss, current_loss)
        logger.info('End Validation Stage')
        logger.info('---------------------')
        validation_count = 0
    return prev_best_loss, validation_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_test", action='store_true', help="Whether to run test or just do train.")
    args = parser.parse_args()
    config = Config(debug=False)
    config.logger.info(config)
    for key, value in vars(config).items():
        config.logger.info(str(key) + ' : ' + str(value))
    if args.do_test:
        test(config)
    else:
        train(config)