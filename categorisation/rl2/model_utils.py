import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
    
class MLP(torch.nn.Module):
    def __init__(self, num_inputs, num_layers=2, num_hidden=64, num_outputs=1, init_std=0.1, sparseness=0.2,
                 preactivation_noise_std=0.0, activation='tanh'):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(num_inputs, num_hidden)] + \
            [nn.Linear(num_hidden,num_hidden) for _ in range(num_layers-2)] + \
            [nn.Linear(num_hidden,num_outputs)]
        )

        self.init_std = init_std
        self.sparseness = sparseness
        self.reset_parameters()

        self.preactivation_noise_std = preactivation_noise_std
        self.activation = {
            'tanh': torch.nn.Tanh(),
            'relu': torch.nn.ReLU(),
            'elu': torch.nn.ELU(),
            'identity': torch.nn.Identity(),
        }[activation]

    def reset_parameters(self, init_std=None, sparseness=None):
        init_std = init_std if init_std is not None else self.init_std
        sparseness = sparseness if sparseness is not None else self.sparseness
        for linear in self.linears:
            linear.reset_parameters()

        with torch.no_grad():
            if init_std is not None:
                for linear in self.linears:
                    linear.weight.normal_(0, init_std)
                    linear.bias.normal_(0, init_std)

            if sparseness > 0.0:
                for linear in self.linears[1:-1]:
                    linear.weight /= (1. - sparseness) ** (1 / 2)
                    linear.weight *= torch.bernoulli(torch.ones_like(linear.weight) * (1. - sparseness))

    def forward(self, x):
        for linear in self.linears[:-1]:
            x = linear(x)
            x = x + torch.randn_like(x) * self.preactivation_noise_std
            x = torch.tanh(x)
        x = self.linears[-1](x)
        # pass x though sigmoid to get probabilities
        x = torch.sigmoid(x)
        return x