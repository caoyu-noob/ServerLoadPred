import torch
import torch.nn as nn
from torch.autograd import *


class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_rate, use_window):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cuda_available = torch.cuda.is_available()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1, bias=True)
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.use_window = use_window

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
        if self.use_window:
            pred = self.output_layer(lstm_out[:, -1, :])
        else:
            pred = self.output_layer(lstm_out)
            # pred = self.dropout(pred)
            pred = pred.squeeze()
        return pred

class GRUModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_rate, use_window):
        super(GRUModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cuda_available = torch.cuda.is_available()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1, bias=True)
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.use_window = use_window

    def init_hidden(self):
        # if not self.cuda_available:
        return (Variable(torch.zeros(1, 1, self.input_dim)), Variable(torch.zeros(1, 1, self.input_dim)))
        # else:
        #     return (Variable(torch.zeros(1, 1, self.input_dim)).cuda(), Variable(torch.zeros(1, 1, self.input_dim)).cuda())

    def forward(self, seq):
        seq_length = seq.batch_sizes.shape[0]
        batch_size = seq.batch_sizes[0]
        gru_out, gru_hidden = self.gru(seq)
        gru_out = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)[0]
        if self.use_window:
            pred = self.output_layer(gru_out[:, -1, :])
        else:
            pred = self.output_layer(gru_out)
            # pred = self.dropout(pred)
            pred = pred.squeeze()
        return pred

class EncoderDecoderModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_rate, use_window):
        super(EncoderDecoderModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cuda_available = torch.cuda.is_available()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.decoder = nn.GRUCell(1, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1, bias=True)
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.use_window = use_window

    def init_hidden(self):
        # if not self.cuda_available:
        return (Variable(torch.zeros(1, 1, self.input_dim)), Variable(torch.zeros(1, 1, self.input_dim)))

    def forward(self, seq):
        seq_length = seq.batch_sizes.shape[0]
        batch_size = seq.batch_sizes[0]
        encoder_out, encoder_hidden = self.encoder(seq)
        encoder_out = nn.utils.rnn.pad_packed_sequence(encoder_out, batch_first=True)[0]
        decoder_hidden_state = encoder_out[:, -1, :]
        decoder_pred = torch.zeros((batch_size, 1))
        if self.cuda_available:
            decoder_pred = decoder_pred.cuda()
        for i in range(seq_length + 1):
            decoder_hidden_state = self.decoder(decoder_pred, decoder_hidden_state)
            decoder_pred = self.output_layer(decoder_hidden_state)
        pred = decoder_pred
        return pred
