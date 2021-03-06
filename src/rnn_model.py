import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        """
            x = [sent len, batch size]
        """
        #embedded = [sent len, batch size, emb dim]
        embedded = self.embedding(x)

        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        output, hidden = self.rnn(embedded)

        assert torch.equal(output[-1,:,:], hidden.squeeze(0))

        return self.linear(hidden.squeeze(0))
