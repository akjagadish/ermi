import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .model_utils import PositionalEncoding


class TransformerDecoder(nn.Module):
    """ Meta-learning model with transformer core for the categorisation task """

    def __init__(self, num_input, num_output, num_hidden, num_layers=1, d_model=256, num_head=1, dropout=0.1, beta=1, max_steps=200, device='cpu') -> None:
        super(TransformerDecoder, self).__init__()

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
        self.criterion = nn.BCELoss()
        self.beta = beta

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

    def make_inputs(self, inputs, prev_choices):
        raise NotImplementedError
        # TODO: concatenate inputs and previous_choices of all trials until then

    def initial_states(self, batch_size):
        return None, None

    def compute_loss(self, model_choices, true_choices):
        return self.criterion(model_choices, true_choices)
