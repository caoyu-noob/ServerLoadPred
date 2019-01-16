import torch
import torch.nn as nn
from torch.autograd import *


class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cuda_available = torch.cuda.is_available()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1, bias=True)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # if not self.cuda_available:
        return (Variable(torch.zeros(1, 1, self.input_dim)), Variable(torch.zeros(1, 1, self.input_dim)))
        # else:
        #     return (Variable(torch.zeros(1, 1, self.input_dim)).cuda(), Variable(torch.zeros(1, 1, self.input_dim)).cuda())

    def forward(self, seq):
        seq_length = seq.shape[0]
        batch_size = seq.shape[1]
        lstm_out, lstm_hidden = self.lstm(seq)
        pred = self.output_layer(lstm_out.view(seq_length, batch_size, -1))
        pred = pred.squeeze().permute(1, 0)
        return pred
