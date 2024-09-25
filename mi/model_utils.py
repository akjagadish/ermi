import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import re
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

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
            [nn.Linear(num_inputs, num_hidden)] +
            [nn.Linear(num_hidden, num_hidden) for _ in range(num_layers-2)] +
            [nn.Linear(num_hidden, num_outputs)]
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
                    linear.weight *= torch.bernoulli(
                        torch.ones_like(linear.weight) * (1. - sparseness))

    def forward(self, x):
        for linear in self.linears[:-1]:
            x = linear(x)
            x = x + torch.randn_like(x) * self.preactivation_noise_std
            x = torch.tanh(x)
        x = self.linears[-1](x)
        # pass x though sigmoid to get probabilities
        x = torch.sigmoid(x)
        return x

def parse_model_path(model_path, kwargs, return_data_info=False):
    # parse num_hidden, num_layers, d_model, num_head, paired, loss from model_path
    
    patterns = {
        "num_hidden": r"num_hidden=(\d+)",
        "num_layers": r"num_layers=(\d+)",
        "d_model": r"d_model=(\d+)",
        "num_head": r"num_head=(\d+)",
        "paired": r"paired=(True|False)",
        "loss": r"loss=([a-zA-Z0-9]+)",
        "std": r"std=([0-9.]+)",
        "ess": r"ess=([0-9.]+)",
    }

    # Initialize a dictionary to store the parsed parameters
    parameters = {}

    # Parse each parameter from the model_path string
    for param, pattern in patterns.items():
        match = re.search(pattern, model_path)
        if match:
            parameters[param] = match.group(1)

    num_hidden = int(parameters.get('num_hidden', 0))
    num_layers = int(parameters.get('num_layers', 0))
    d_model = int(parameters.get('d_model', 0))
    num_head = int(parameters.get('num_head', 0))
    loss_fn =  parameters.get('loss', 'nll')
    model_max_steps = kwargs.get('model_max_steps', 0)

    source = 'claude' if 'claude' in model_path else 'synthetic' if 'synthetic' in model_path else 'syntheticnonlinear' if 'syntheticnonlinear' in model_path else 'NA'
    condition = 'rank' if 'rank' in model_path else 'direction' if 'direction' in model_path else 'unknown'
    
    if return_data_info:
        return num_hidden, num_layers, d_model, num_head, loss_fn, model_max_steps, source, condition
    
    return num_hidden, num_layers, d_model, num_head, loss_fn, model_max_steps

def get_wd_from_std(std, ess):
    return 1 / ((std ** 2) * ess)

def compute_elbo(optimizer, model, std, packed_inputs, targets, sequence_lengths, eval_samples=100):
    # compute KLD
    p = Normal(torch.zeros([]), std)
    kld =  0
    for group in optimizer.param_groups:
        q_m = torch.cat([p.flatten() for p in group["params"] if p is not None], 0)
        q_s = 1 / torch.sqrt(group["ess"] * (group["hess"] + group["weight_decay"]))
        q = Normal(q_m, q_s)
        kld += kl_divergence(q, p).sum().item()

    # compute NLL
    nll = 0
    for _ in range(eval_samples):
        with optimizer.sampled_params():
            model.eval()
            with torch.no_grad():
                nll += (model.compute_loss(packed_inputs, targets, sequence_lengths)*targets.numel()).item()

    # compute ELBO
    return nll / eval_samples + kld

def compute_kld(optimizer, std):
    # compute KLD
    p = Normal(torch.zeros([]), std)
    kld =  0
    for group in optimizer.param_groups:
        q_m = torch.cat([p.flatten() for p in group["params"] if p is not None], 0)
        q_s = 1 / torch.sqrt(group["ess"] * (group["hess"] + group["weight_decay"]))
        q = Normal(q_m, q_s)
        kld += kl_divergence(q, p).sum().item()
    return kld

def annealed_ess(episode, num_episodes, ess_init, ess_final, annealing_fraction):
    if episode < (num_episodes * annealing_fraction):
        return (ess_init - ess_final) * (1 - episode / (num_episodes * annealing_fraction)) + ess_final
    return ess_final