import json
import random
import torch
import torch.nn as nn

class Dataset():
    def __init__(self, json_name, batch_size, max_len, is_cuda_available):
        with open(json_name, 'r') as f:
            self.data = json.load(f)
        new_data = []
        for idx, d in enumerate(self.data):
            self.data[idx] = d['data']
        self.batch_size = batch_size
        self.data_size = len(self.data)
        self.is_new_epoch = True
        self.batch_idxs = None
        self.max_len = max_len
        self.is_cuda_available = is_cuda_available

    def new_epoch(self):
        self.is_new_epoch = True
        idxs = list(range(self.data_size))
        random.shuffle(idxs)
        self.batch_idxs = [idxs[i:i+self.batch_size] if i + self.batch_size <= self.data_size else idxs[i:]
                      for i in range(0, self.data_size, self.batch_size)]

    def generate_batches(self):
        return (self.prepro_data_for_batch([self.data[idx] for idx in batch_idx]) for batch_idx in self.batch_idxs)

    def prepro_data_for_batch(self, data):
        cur_max_len = max([len(d) for d in data])
        if cur_max_len - 1  > self.max_len:
            cur_max_len = self.max_len
        time_interval = []
        training_data, output_label = [], []
        # training_data = torch.zeros((self.batch_size, cur_max_len, 6))
        # output_label = torch.zeros((self.batch_size, cur_max_len))
        for index, d in enumerate(data):
            if len(d) > cur_max_len + 1:
                d = d[:cur_max_len + 1]
            current_output_label = [single_data[1] for single_data in d][1:]
            for index in range(len(d) - 1, 0, -1):
                time_interval.append(d[index][0] - d[index - 1][0])
            time_interval.reverse()
            d = d[:-1]
            current_training_data = [single_data[1:] + [time_interval[index]] for index, single_data in enumerate(d)]
            # training_data[index] = torch.Tensor(current_training_data)
            # output_label[index] = torch.Tensor(current_output_label)
            if len(current_training_data) < cur_max_len:
                current_training_data.extend([[0, 0, 0, 0, 0, 0]] * (cur_max_len - len(current_training_data)))
                current_output_label.extend([0] * (cur_max_len - len(current_output_label)))
            training_data.append(current_training_data)
            output_label.append(current_output_label)

        training_tensor, label_tensor = torch.FloatTensor(training_data), torch.FloatTensor(output_label)
        if self.is_cuda_available:
            training_tensor, label_tensor = training_tensor.cuda(), label_tensor.cuda()
        return (training_tensor, label_tensor)
