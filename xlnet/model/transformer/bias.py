import torch.nn as nn
import torch


class TransformerXLBias(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.r_w_bias = self.get_bias_parameter(config)
        self.r_r_bias = self.get_bias_parameter(config)
        self.r_s_bias = self.get_bias_parameter(config)

    def forward(self, *input):
        raise NotImplementedError

    def get_bias_parameter(self, config):
        if config.model.untie_bias:
            bias_shape = [config.model.num_layers, config.model.head_num, config.model.head_dim]
        else:
            bias_shape = [config.model.head_num, config.model.head_dim]
        return torch.rand(bias_shape)
