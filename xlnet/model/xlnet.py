import torch.nn as nn

from .transformer import TransformerXL


class XLNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = TransformerXL(config)

    def forward(self, *input):
        pass
