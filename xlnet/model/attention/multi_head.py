import torch
import torch.nn as nn

from .absolute import AbsoluteAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.model.dropout_prob)
        self.absolute_attn = AbsoluteAttention(config)

    def forward(self, *input):
        pass
