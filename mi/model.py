import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import PositionalEncoding


class TransformerDecoderRegression(nn.Module):
    """ Meta-learned inference with vanilla transformer core"""

    def __init__(self, num_input, num_output, num_hidden, num_layers=1, d_model=256, num_head=1, dropout=0.1, beta=1, max_steps=20, loss='nll', device='cpu') -> None:
        super(TransformerDecoderRegression, self).__init__()

        self.device = torch.device(device)
        self.num_input = num_input + num_output
        self.num_output = num_output
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_head = num_head

        # check d_model is correct
        assert d_model % num_head == 0, "nheads must divide evenly into d_model"

        # embedding layer to move to d_model space
        self.embedding = nn.Linear(self.num_input, d_model)

        # position encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=max_steps,
        )

        # transformer encoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_head,
                                                   dim_feedforward=num_hidden,
                                                   dropout=dropout, batch_first=True)

        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # linear layer
        self.linear_mu = nn.Linear(d_model, num_output)
        self.linear_logscale = nn.Linear(d_model, num_output)

        # loss
        assert loss in ['mse', 'nll'], "loss must be either mse or nll"
        self.loss = loss

    def make_sequence_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_square_subsequent_mask(self, seq_len, batch_size):

        seq_len = torch.tensor(seq_len)
        # check if elements of the list seq_len are equal
        if (seq_len == seq_len[0]).all() and len(seq_len) > 1:
            mask = self.make_sequence_mask(seq_len[0])
        else:
            mask = self.make_sequence_mask(max(seq_len))
            # repeat the mask matrix self.num_head times
            mask = mask.unsqueeze(0).repeat(batch_size*self.num_head, 1, 1)
        return mask

    def forward(self, packed_inputs, sequence_lengths):

        inputs = self.embedding(packed_inputs.float())
        inputs_pos_encoded = self.pos_encoder(inputs)
        tgt_mask = self.generate_square_subsequent_mask(
            seq_len=sequence_lengths, batch_size=packed_inputs.shape[0])
        output = self.transformer(inputs_pos_encoded.float().to(self.device), inputs_pos_encoded.float(
        ).to(self.device), tgt_mask=tgt_mask.to(self.device), memory_mask=tgt_mask.to(self.device))

        mu = self.linear_mu(output.to(self.device))
        std = torch.exp(self.linear_logscale(output.to(self.device)))

        return torch.distributions.Normal(mu, std) if self.loss == 'nll' else mu

    def compute_loss(self, packed_inputs, targets, sequence_lengths=None):
        if self.loss == 'mse':
            criterion = nn.MSELoss()
            model_preds = self.forward(packed_inputs, sequence_lengths)
            model_preds = torch.concat([model_preds[i, :seq_len] for i, seq_len in enumerate(
                sequence_lengths)], axis=0).squeeze().float()
            true_targets = torch.concat(
                targets, axis=0).float().to(self.device)
            return criterion(model_preds, true_targets)
        elif self.loss == 'nll':
            predictive_posterior = self.forward(
                packed_inputs, sequence_lengths)
            return - \
                predictive_posterior.log_prob(
                    torch.stack(targets).unsqueeze(2).float().to(self.device)).mean()


class TransformerDecoderClassification(nn.Module):
    """ Meta-learning model with transformer core for the decisionmaking task """

    def __init__(self, num_input, num_output, num_hidden, num_layers=1, d_model=256, num_head=1, dropout=0.1, beta=1, max_steps=20, loss='bce', device='cpu') -> None:
        super(TransformerDecoderClassification, self).__init__()

        self.device = torch.device(device)
        self.num_input = num_input + num_output
        self.num_output = num_output
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_head = num_head

        # check d_model is correct
        assert d_model % num_head == 0, "nheads must divide evenly into d_model"

        # embedding layer to move to d_model space
        self.embedding = nn.Linear(self.num_input, d_model)

        # position encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=max_steps,
        )

        # transformer encoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_head,
                                                   dim_feedforward=num_hidden,
                                                   dropout=dropout, batch_first=True)

        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # linear layer
        self.linear = nn.Linear(d_model, num_output)
        self.sigmoid = nn.Sigmoid()
        self.beta = beta

        assert loss == 'bce', "loss must be binary cross entropy"
        self.loss = loss

    def make_sequence_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_square_subsequent_mask(self, seq_len, batch_size):

        seq_len = torch.tensor(seq_len)
        # check if elements of the list seq_len are equal
        if (seq_len == seq_len[0]).all() and len(seq_len) > 1:
            mask = self.make_sequence_mask(seq_len[0])
        else:
            mask = self.make_sequence_mask(max(seq_len))
            # repeat the mask matrix self.num_head times
            mask = mask.unsqueeze(0).repeat(batch_size*self.num_head, 1, 1)
        return mask

    def forward(self, packed_inputs, sequence_lengths):

        inputs = self.embedding(packed_inputs.float())
        inputs_pos_encoded = self.pos_encoder(inputs)
        tgt_mask = self.generate_square_subsequent_mask(
            seq_len=sequence_lengths, batch_size=packed_inputs.shape[0])
        output = self.transformer(inputs_pos_encoded.float().to(self.device), inputs_pos_encoded.float(
        ).to(self.device), tgt_mask=tgt_mask.to(self.device), memory_mask=tgt_mask.to(self.device))
        y = self.linear(output.to(self.device))
        y = self.sigmoid(self.beta*y.to(self.device))

        return y

    def compute_loss(self, packed_inputs, targets, sequence_lengths=None):

        criterion = nn.BCELoss() if self.loss == 'bce' else None
        model_choices = self.forward(packed_inputs, sequence_lengths)
        model_choices = torch.concat([model_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        true_choices = targets.reshape(-1, 1).float().to(self.device).squeeze()
        # torch.concat(targets, axis=0).float().to(self.device)
        return criterion(model_choices, true_choices)
