import torch
import torch.nn as nn


class HeadProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_weight = torch.rand(
            [config.model.hidden_size, config.model.head_num, config.model.head_dim]
        )
        self.kernel = nn.Parameter(kernel_weight)

    def forward(self, head_input) -> torch.Tensor:
        return torch.einsum("ibh,hnd->ibnd", head_input, self.kernel)


class HeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q = HeadProjection(config)
        self.k = HeadProjection(config)
        self.v = HeadProjection(config)

    def forward(self, *inputs):
        q_head, k_head, v_head = (
            model.forward(source)
            for source, model in zip(inputs, [self.q, self.k, self.k])
        )
        output = HeadAttentionOutput(q_head, k_head, v_head)
        return output


class HeadAttentionOutput:
    def __init__(self, q_head, k_head, v_head):
        self.q_head = q_head
        self.k_head = k_head
        self.v_head = v_head
