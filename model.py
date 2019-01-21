import torch
import torch.nn as nn
from torch.autograd import *


class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cuda_available = torch.cuda.is_available()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1, bias=True)
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(p=dropout_rate)

    def init_hidden(self):
        # if not self.cuda_available:
        return (Variable(torch.zeros(1, 1, self.input_dim)), Variable(torch.zeros(1, 1, self.input_dim)))
        # else:
        #     return (Variable(torch.zeros(1, 1, self.input_dim)).cuda(), Variable(torch.zeros(1, 1, self.input_dim)).cuda())

    def forward(self, seq):
        seq_length = seq.batch_sizes.shape[0]
        batch_size = seq.batch_sizes[0]
        lstm_out, lstm_hidden = self.lstm(seq)
        lstm_out = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)[0]
        pred = self.output_layer(lstm_out)
        # pred = self.dropout(pred)
        pred = pred.squeeze()
        return pred
