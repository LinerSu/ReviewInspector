import torch
import torch.nn as nn
import torch.nn.init as init
import math


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.num_dir = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.linear = nn.Linear(hidden_dim*n_layers*self.num_dir, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_weights(hidden_dim)
    
    def init_weights(self, hidden_dim):
        """
        Ref:
            Linear: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L58
            RNN: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py#L120
            Embedding: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/sparse.py#L108
        """
        stdv = 1.0 / math.sqrt(hidden_dim)
        for weight in self.rnn.parameters():
            init.normal_(weight, 0, stdv)
        init.kaiming_normal_(self.linear.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.linear.weight)
        bound = 1 / math.sqrt(fan_in)
        init.normal_(self.linear.bias, -bound, bound)

    def forward(self, x):
        """
            x = [sent len, batch size]
        """
        #embedded = [sent len, batch size, emb dim]
        # embedded = self.dropout(self.embedding(x))
        batch_size = x.size(1)
        embedded = self.embedding(x)

        #output = [sent len, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        output, (hidden, cell) = self.rnn(embedded)

        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout

        hidden = self.dropout(hidden.transpose(0, 1).contiguous().view(batch_size, -1))

        #hidden = [batch size, hid dim * num directions]

        return self.linear(hidden)
