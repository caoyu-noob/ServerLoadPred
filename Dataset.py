import json
import random
import torch
import torch.nn as nn

class Dataset():
    def __init__(self, json_name, batch_size, max_len, is_cuda_available, use_splitted_files, use_net_accumulation=True):
        self.data = None
        if not use_splitted_files:
            with open(json_name, 'r') as f:
                self.data = json.load(f)
            for idx, d in enumerate(self.data):
                self.data[idx] = d['data']
            self.data_size = len(self.data)
        else:
            with open(json_name + '/summary.json', 'r') as f:
                self.data_size = json.load(f)[0]
        self.batch_size = batch_size
        self.is_new_epoch = True
        self.batch_idxs = None
        self.max_len = max_len
        self.is_cuda_available = is_cuda_available
        self.use_net_accumulation = use_net_accumulation

    def new_epoch(self):
        self.is_new_epoch = True
        idxs = list(range(self.data_size))
        random.shuffle(idxs)
        self.batch_idxs = [idxs[i:i+self.batch_size] if i + self.batch_size <= self.data_size else idxs[i:]
                      for i in range(0, self.data_size, self.batch_size)]

    def generate_batches(self, use_pack=True):
        return (self.prepro_data_for_batch([self.data[idx] for idx in batch_idx], use_pack)
                for batch_idx in self.batch_idxs)

    def prepro_data_for_batch(self, data, use_pack):
        seq_lens = [self.max_len if len(d) - 1 > self.max_len else len(d) - 1 for d in data]
        cur_max_len = max(seq_lens)
        if cur_max_len > self.max_len:
            cur_max_len = self.max_len
        time_interval, net_in, net_out = [], [], []
        training_data, output_label = [], []
        # training_data = torch.zeros((self.batch_size, cur_max_len, 6))
        # output_label = torch.zeros((self.batch_size, cur_max_len))
        for index, d in enumerate(data):
            if len(d) > cur_max_len + 1:
                d = d[:cur_max_len + 1]
            current_output_label = [single_data[1] for single_data in d][1:]
            for index in range(1, len(d)):
                time_interval.append(d[index][0] - d[index - 1][0])
                net_in.append((d[index][3] - d[index - 1][3]) * 100)
                net_out.append((d[index][4] - d[index - 1][4]) * 100)
            d = d[:-1]
            if not self.use_net_accumulation:
                current_training_data = [[single_data[1], single_data[2], net_in[index], net_out[index], single_data[5],
                        time_interval[index]] for index, single_data in enumerate(d)]
            else:
                current_training_data = [single_data[1:] + [time_interval[index]] for index, single_data in
                                         enumerate(d)]
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
        seq_lens = torch.IntTensor(seq_lens)
        sorted_seq_lens, sorted_index = torch.sort(seq_lens, descending=True)
        if use_pack:
            training_tensor = nn.utils.rnn.pack_padded_sequence(training_tensor[sorted_index], sorted_seq_lens,
                    batch_first=True)
        # label_pack = nn.utils.rnn.pack_padded_sequence(label_tensor[sorted_index], sorted_seq_lens, batch_first=True)
        return (training_tensor, label_tensor, seq_lens)

class DatasetWindow(Dataset):
    def __init__(self, json_name, batch_size, max_len, is_cuda_available, window_size, use_splitted_files,
            use_net_accumulation=True):
        super(DatasetWindow, self).__init__(json_name, batch_size, max_len, is_cuda_available, use_splitted_files,
                use_net_accumulation=use_net_accumulation)
        self.window_size = window_size
        self.epoch_finish = False
        self.use_splitted_files = use_splitted_files
        self.json_name = json_name

    def load_splitted_sample(self, sample_idx):
        with open(self.json_name + '/train_data_' + str(sample_idx) + '.json') as f:
            tmp_data = json.load(f)
        return tmp_data['data']

    def new_epoch(self):
        idxs = list(range(self.data_size))
        random.shuffle(idxs)
        idxs_sample_idx = []
        if not self.use_splitted_files:
            for idx in idxs:
                idxs_sample_idx.extend([(idx, idx_in_sample) for idx_in_sample in
                        range(self.window_size, len(self.data[idx]))])
            self.batch_idxs_sample_idxs = [
                idxs_sample_idx[i:i + self.batch_size] if i + self.batch_size <= len(idxs_sample_idx)
                else idxs_sample_idx[i:] for i in range(0, len(idxs_sample_idx), self.batch_size)]
        else:
            with open(self.json_name + '/summary.json', 'r') as f:
                summary = json.load(f)
                self.data_lens = summary[1]
        #     for idx in idxs:
        #         idxs_sample_idx.extend([(idx, idx_in_sample) for idx_in_sample in
        #                 range(self.window_size, data_lens[idx])])
            sample_index, index_in_cur_sample = 0, self.window_size
            self.batch_idxs_sample_idxs = []
            while sample_index < self.data_size:
                remain_batch_size = self.batch_size
                sample_idx_in_cur_batch = []
                while sample_index < self.data_size and \
                        self.data_lens[idxs[sample_index]] - index_in_cur_sample <= remain_batch_size:
                    sample_idx_in_cur_batch.append(idxs[sample_index])
                    remain_batch_size -= (self.data_lens[idxs[sample_index]] - index_in_cur_sample)
                    index_in_cur_sample = self.window_size
                    sample_index += 1
                if sample_index < self.data_size and remain_batch_size != 0:
                    sample_idx_in_cur_batch.append(idxs[sample_index])
                    index_in_cur_sample += remain_batch_size
                self.batch_idxs_sample_idxs.append(sample_idx_in_cur_batch)
        idxs_sample_idx = []
        if self.use_splitted_files:
            self.current_sample_idx = idxs[0]
            self.current_sample_data = self.load_splitted_sample(idxs[0])
        self.current_idx_in_a_sample = self.window_size
        self.epoch_finish = False

    def generate_batches(self):
        if self.use_splitted_files:
            return (self.prepro_data_for_batch_on_sample_idx(batch_idxs_sample_idx) for batch_idxs_sample_idx
                    in self.batch_idxs_sample_idxs)
        else:
            return (self.prepro_data_for_batch(batch_idxs_sample_idx) for batch_idxs_sample_idx
                    in self.batch_idxs_sample_idxs)

    def prepro_data_for_batch(self, batch_idxs_sample_idx):
        training_data = []
        label = []
        for idxs in batch_idxs_sample_idx:
            sample_idx = idxs[0]
            data_location = idxs[1]
            if not self.use_splitted_files:
                d = self.data[sample_idx][data_location - self.window_size:data_location + 1]
            else:
                if sample_idx != self.current_sample_idx:
                    self.current_sample_data = self.load_splitted_sample(sample_idx)
                    self.current_sample_idx = sample_idx
                d = self.current_sample_data[data_location - self.window_size:data_location + 1]
            label.append(d[-1][1])
            time_interval, net_in, net_out = [], [0], [0]
            for index in range(0, len(d) - 1):
                time_interval.append(d[index + 1][0] - d[index][0])
                net_in.append((d[index + 1][3] - d[index][3]) * 100)
                net_out.append((d[index + 1][4] - d[index][4]) * 100)
            net_in = net_in[:-1]
            net_out = net_out[:-1]
            d = d[:-1]
            if not self.use_net_accumulation:
                current_training_data = [[single_data[1], single_data[2], net_in[index], net_out[index], single_data[5],
                        time_interval[index]] for index, single_data in enumerate(d)]
            else:
                current_training_data = [single_data[1:] + [time_interval[index]] for index, single_data in
                                         enumerate(d)]
            training_data.append(current_training_data)
            self.current_idx_in_a_sample += 1
        training_tensor, label_tensor = torch.FloatTensor(training_data), torch.FloatTensor(label)
        if self.is_cuda_available:
            training_tensor, label_tensor = training_tensor.cuda(), label_tensor.cuda()
        seq_lens = torch.IntTensor([self.window_size] * training_tensor.shape[0])
        training_pack = nn.utils.rnn.pack_padded_sequence(training_tensor, seq_lens, batch_first=True)
        return (training_pack, label_tensor.unsqueeze(1), seq_lens)

    def prepro_data_for_batch_on_sample_idx(self, batch_sample_idx):
        training_data = []
        label = []
        remain_batch_size = self.batch_size
        for sample_idx in batch_sample_idx:
            if sample_idx != self.current_sample_idx:
                self.current_sample_data = self.load_splitted_sample(sample_idx)
                self.current_sample_idx = sample_idx
            if self.current_idx_in_a_sample + remain_batch_size < self.data_lens[sample_idx]:
                d = self.current_sample_data[self.current_idx_in_a_sample - self.window_size:
                                             self.current_idx_in_a_sample + remain_batch_size]
                self.current_idx_in_a_sample += remain_batch_size
            else:
                d = self.current_sample_data[self.current_idx_in_a_sample - self.window_size:]
                self.current_idx_in_a_sample = self.window_size
                remain_batch_size -= (len(d) - self.window_size)
            time_interval, net_in, net_out = [], [0], [0]
            for index in range(0, len(d) - 1):
                time_interval.append(d[index + 1][0] - d[index][0])
                net_in.append((d[index + 1][3] - d[index][3]) * 100)
                net_out.append((d[index + 1][4] - d[index][4]) * 100)
            net_in = net_in[:-1]
            net_out = net_out[:-1]
            label.extend([data[1] for data in d[8:]])
            d = d[:-1]
            for i in range(self.window_size, len(d) + 1):
                cur_d = d[i - self.window_size : i]
                if self.use_net_accumulation:
                    training_data.append([[single_data[1], single_data[2], net_in[index], net_out[index], single_data[5],
                            time_interval[index]] for index, single_data in enumerate(cur_d)])
                else:
                    training_data.append([single_data[1:] + [time_interval[index]] for index, single_data in
                             enumerate(cur_d)])
        training_tensor, label_tensor = torch.FloatTensor(training_data), torch.FloatTensor(label)
        if self.is_cuda_available:
            training_tensor, label_tensor = training_tensor.cuda(), label_tensor.cuda()
        seq_lens = torch.IntTensor([self.window_size] * training_tensor.shape[0])
        try:
            training_pack = nn.utils.rnn.pack_padded_sequence(training_tensor, seq_lens, batch_first=True)
        except:
            print(training_tensor.shape[0])
            return (None, None, None)
        return (training_pack, label_tensor.unsqueeze(1), seq_lens)

