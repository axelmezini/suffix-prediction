import torch.nn as nn
import torch.nn.functional as F
import torch

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class LSTM_model(nn.Module):
    def __init__(self, hidden_dim, vocab_size, tagset_size):
        super(LSTM_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 2
        self.input_size = vocab_size
        self.lstm = nn.LSTM(vocab_size, hidden_dim, self.num_layers, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        #batch_size = x.size()[0]
        #h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        #c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        #lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        lstm_out, (hn, cn) = self.lstm(x)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space, (hn, cn)

    def forward_from_state(self, x, state):
        lstm_out, (hn, cn) = self.lstm(x, state)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space, (hn, cn)

    def next_sym_prob(self, x, state):
        tag_space, state = self.forward_from_state(x, state)
        tag_space = F.softmax(tag_space, dim=-1)
        return tag_space, state

    def predict(self, sentence):
        tag_space = self.forward(sentence)
        out = F.softmax(tag_space, dim=1)[-1]
        return out
