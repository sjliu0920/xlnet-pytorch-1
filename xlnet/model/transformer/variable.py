import torch
import torch.nn as nn


class TransformerVariable(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.r_w_bias = self._get_bias_parameter(config)
        self.r_r_bias = self._get_bias_parameter(config)
        self.r_s_bias = self._get_bias_parameter(config)
        self.segment_embedding = self._get_segment_embed(config)

    def forward(self, *input):
        raise NotImplementedError

    def _get_segment_embed(self, config):
        embed_shape = [
            config.model.num_layers,
            2,
            config.model.head_num,
            config.model.head_dim,
        ]
        return nn.Parameter(torch.rand(embed_shape, dtype=torch.float))

    def _get_bias_parameter(self, config):
        if config.model.untie_bias:
            bias_shape = [
                config.model.num_layers,
                config.model.head_num,
                config.model.head_dim,
            ]
        else:
            bias_shape = [config.model.head_num, config.model.head_dim]
        return nn.Parameter(torch.rand(bias_shape, dtype=torch.float))


class TransformerLayerVariable(nn.Module):
    def __init__(self, config, variable: TransformerVariable, layer_num: int = None):
        super().__init__()
        if config.model.untie_r and layer_num is not None:
            self.r_w_bias = variable.r_w_bias[layer_num]
            self.r_r_bias = variable.r_r_bias[layer_num]
            self.r_s_bias = variable.r_s_bias[layer_num]
            self.seg_embed = variable.segment_embedding[layer_num]
        else:
            self.r_w_bias = variable.r_w_bias
            self.r_r_bias = variable.r_r_bias
            self.r_s_bias = variable.r_s_bias
            self.seg_embed = variable.segment_embedding

    def forward(self, *input):
        raise NotImplementedError
