import json
from model import LSTMModel, GRUModel, EncoderDecoderModel, RNNDAModel
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

model_dict = {'gru' : GRUModel, 'lstm' : LSTMModel, 'encoder' : EncoderDecoderModel, 'rnnda' : RNNDAModel}

def plot_func(xs, ys, preds):
    plt.figure()
    plt.title('RNNDA(128)预测值与真实值对比曲线', fontproperties='SimHei')
    plt.xlim([0, len(xs)])
    for i in range(len(xs)):
        plt.subplot(3, 1, i + 1)
        plt.plot(list(range(len(ys[i]))), ys[i], 'r-')
        plt.plot(list(range(len(ys[i]))), preds[i], '-')
        plt.ylabel('CPU负载/%', fontproperties='SimHei')
    plt.show()

if __name__ == '__main__':
    model_type = 'rnnda'
    hidden_dim = 256
    window_size = 8
    model_name = 'model'
    if model_type == 'rnnda':
        model = model_dict[model_type](6, hidden_dim, window_size)
    else:
        model = model_dict[model_type](6, hidden_dim, 0, True)
    model_save_path = 'model/window' + str(window_size) + '_' + model_type + '_hidden_' + str(hidden_dim) + '/'
    model.load_state_dict(torch.load(model_save_path + 'dev_best'))
    data_nums = [100, 200, 300]
    plot_length = 160
    start_position = 100
    with open('test_data.json', 'r') as f:
        test_data = json.load(f)
    preds, ys, xs = [], [], []
    for data_num in data_nums:
        data = test_data[data_num]['data']
        origin_data = data[start_position: window_size + plot_length + start_position]
        x = [d[0] for d in origin_data[window_size:]]
        y = [d[1] for d in origin_data[window_size:]]
        xs.append(x)
        ys.append(y)
        time_interval = []
        for index in range(0, len(origin_data) - 1):
            time_interval.append(origin_data[index + 1][0] - origin_data[index][0])
        feature = []
        for i in range(plot_length):
            cur_feature = []
            for j in range(window_size):
                cur_feature.append(origin_data[i + j][1:] + [time_interval[i + j]])
            feature.append(cur_feature)
        feature = torch.FloatTensor(feature)
        if torch.cuda.is_available():
            feature = feature.cuda()
            model = model.cuda()
        seq_lens = torch.IntTensor([window_size] * feature.shape[0])
        feature = nn.utils.rnn.pack_padded_sequence(feature, seq_lens, batch_first=True)
        cur_pred = model(feature)
        pred = cur_pred.data.cpu().numpy()[:, 0].tolist()
        preds.append(pred)
    d = {'xs':xs, 'ys':ys, 'preds': preds}
    with open('RNNDA_256_FIG.json', 'w') as f:
        json.dump(d, f)
    plot_func(xs, ys, preds)