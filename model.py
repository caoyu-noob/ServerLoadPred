import torch
import torch.nn as nn
import torch.nn.functional as F
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

class RNNDAModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, window_size):
        super(RNNDAModel, self).__init__()
        self.encoder = Encoder(window_size, input_dim, hidden_dim)
        self.decoder = Decoder(window_size, input_dim, hidden_dim)

    def forward(self, seq):
        seq = nn.utils.rnn.pad_packed_sequence(seq, batch_first=True)[0]
        encoder_output = self.encoder(seq)
        prev_load = seq[:, :, 1]
        pred = self.decoder(encoder_output, prev_load)
        return pred

class RNNDAModel_Att1(nn.Module):

    def __init__(self, input_dim, hidden_dim, window_size):
        super(RNNDAModel_Att1, self).__init__()
        self.encoder = Encoder(window_size, input_dim, hidden_dim)
        self.decoder = Decoder(window_size, input_dim, hidden_dim)

    def forward(self, seq):
        seq = nn.utils.rnn.pad_packed_sequence(seq, batch_first=True)[0]
        encoder_output = self.encoder(seq, use_att=False)
        prev_load = seq[:, :, 1]
        pred = self.decoder(encoder_output, prev_load)
        return pred

class RNNDAModel_Att2(nn.Module):

    def __init__(self, input_dim, hidden_dim, window_size):
        super(RNNDAModel_Att2, self).__init__()
        self.encoder = Encoder(window_size, input_dim, hidden_dim)
        self.decoder = Decoder(window_size, input_dim, hidden_dim)

    def forward(self, seq):
        seq = nn.utils.rnn.pad_packed_sequence(seq, batch_first=True)[0]
        encoder_output = self.encoder(seq)
        prev_load = seq[:, :, 1]
        pred = self.decoder(encoder_output, prev_load, use_att=False)
        return pred

class RNNDAModel_Att(nn.Module):

    def __init__(self, input_dim, hidden_dim, window_size):
        super(RNNDAModel_Att, self).__init__()
        self.encoder = Encoder(window_size, input_dim, hidden_dim)
        self.decoder = Decoder(window_size, input_dim, hidden_dim)

    def forward(self, seq):
        seq = nn.utils.rnn.pad_packed_sequence(seq, batch_first=True)[0]
        encoder_output = self.encoder(seq, use_att=False)
        prev_load = seq[:, :, 1]
        pred = self.decoder(encoder_output, prev_load, use_att=False)
        return pred

class Encoder(nn.Module):
    """encoder in DARNN.
    All tensors are created by [Tensor].new_* , so new tensors
    are on same device as [Tensor]. No need for `device` to be
    passed
    """

    def __init__(self, window_size, feat_dim, hid_dim):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.feat_dim = feat_dim
        self.window_size = window_size

        # LSTM to encoder the input features
        self.lstm = nn.LSTM(
            input_size=self.feat_dim, hidden_size=self.hid_dim, batch_first=True)

        # Construct the input attention mechanism using a two-layer linear net
        self.attn = nn.Sequential(
            nn.Linear(2 * hid_dim + window_size, feat_dim), nn.Tanh(),
            nn.Linear(feat_dim, 1))

    def forward(self, X, use_att=True):
        encoder_out = X.new_zeros(X.shape[0], X.shape[1], self.hid_dim)

        # Eq. 8, parameters not in nn.Linear but to be learnt
        # v_e = torch.nn.Parameter(data=torch.empty(
        #     self.feat_dim, self.timesteps).uniform_(0, 1), requires_grad=True)
        # U_e = torch.nn.Parameter(data=torch.empty(
        #     self.timesteps, self.timesteps).uniform_(0, 1), requires_grad=True)

        # hidden, cell: initial states with dimention hidden_size
        h = self._init_state(X)
        s = self._init_state(X)

        for t in range(self.window_size):
            # (batch_size, feat_dim, (2*hidden_size + timesteps))
            # tensor.expand: do not copy data; -1 means no changes at that dim
            if use_att:
                x = torch.cat((h.expand(self.feat_dim, -1, -1).permute(1, 0, 2),
                               s.expand(self.feat_dim, -1, -1).permute(1, 0, 2),
                               X.permute(0, 2, 1)), dim=2)
                # (batch_size, feat_dim, 1)
                e = self.attn(x)

                # get weights by softmax
                # (batch_size, feat_dim)
                alpha = F.softmax(e.squeeze(dim=-1), dim=1)

                # get new input for LSTM
                x_tilde = torch.mul(alpha, X[:, t, :])
            else:
                x_tilde = X[:, t, :]

            # encoder LSTM
            self.lstm.flatten_parameters()
            # self.lstm has batch_first=True flag
            # x_tilde -> (batchsize, 1, feat_dim)
            _, final_state = self.lstm(x_tilde.unsqueeze(1), (h, s))
            h = final_state[0]  # (1, batchsize, hidden)
            s = final_state[1]  # (1, batchsize, hidden)
            encoder_out[:, t, :] = h

        return encoder_out

    def _init_state(self, X):
        batchsize = X.shape[0]
        # same dtype, device as X
        init_state = X.new_zeros([1, batchsize, self.hid_dim])
        return init_state

class Decoder(nn.Module):
    """decoder in DA_RNN."""

    def __init__(self, window_size, feat_dim, hid_dim):
        """Initialize a decoder in DA_RNN.
        feat_dim: encoder hidden state dim
        """
        super(Decoder, self).__init__()
        self.hid_dim = hid_dim
        self.window_size = window_size

        self.attn = nn.Sequential(
            nn.Linear(3 * hid_dim, feat_dim), nn.Tanh(),
            nn.Linear(feat_dim, 1))
        self.lstm = nn.LSTM(input_size=1, hidden_size=hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim + 1, 1)
        self.fc_final = nn.Linear(2 * hid_dim, 1)

    def forward(self, H, Y, use_att=True):
        """forward."""
        d_n = self._init_state(H)
        c_n = self._init_state(H)
        for t in range(self.window_size):

            if use_att:
                # (batch_size, window_size, 2*window_size + hidden_dim)
                x = torch.cat((d_n.expand(self.window_size, -1, -1).permute(1, 0, 2),
                               c_n.expand(self.window_size, -1, -1).permute(1, 0, 2), H), dim=2)

                # (batch_size, window_size)
                beta = F.softmax(self.attn(x).squeeze(dim=-1), dim=1)
                # Eqn. 14: compute context vector
                # (batch_size, hidden_dim)
                context = torch.bmm(beta.unsqueeze(1), H).squeeze(dim=1)
            else:
                weights = torch.ones(H.shape[0], self.window_size) / self.window_size
                if torch.cuda.is_available():
                    weights = weights.cuda()
                context = torch.bmm(weights.unsqueeze(1), H).squeeze(dim=1)
            # Eqn. 15
            # batch_size * 1
            y_tilde = self.fc(torch.cat((context, Y[:, t].unsqueeze(1)), dim=1))

            # Eqn. 16: LSTM
            self.lstm.flatten_parameters()
            _, final_states = self.lstm(y_tilde.unsqueeze(1), (d_n, c_n))
            # 1 * batch_size * hid_dim
            d_n = final_states[0]
            # 1 * batch_size * hid_dim
            c_n = final_states[1]
        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))

        return y_pred

    def _init_state(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

            Args:
                X
            Returns:
                initial_hidden_states

            """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        initial_state = X.new_zeros([1, X.shape[0], self.hid_dim])
        return initial_state
