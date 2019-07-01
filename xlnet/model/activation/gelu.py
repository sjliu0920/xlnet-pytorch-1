import math

import torch
import torch.nn as nn


"""
GELU code from codertimo/BERT-Pytorch
https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/utils/gelu.py
"""


class GELU(nn.Module):
    """
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        eq = torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
        return 0.5 * x * (1 + eq)


def get_activation(config) -> nn.Module:
    """get activation module depends on config"""
    activation_config = config.model.activation_fn
    if activation_config == "relu":
        return nn.ReLU()
    elif activation_config == "gelu":
        return GELU()
    raise ValueError("'activation_fn' config should be 'relu' or 'gelu' ")
