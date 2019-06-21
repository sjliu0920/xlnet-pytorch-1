import torch

from xlnet.model.attention.core.head import HeadAttention, HeadProjection
from xlnet.model.attention.stream.relative import RelativeAttention
from xlnet.model.attention.core.post import PostAttention
from xlnet.model.transformer.variable import TransformerLayerVariable


class RelativeMultiHeadAttention(HeadAttention, RelativeAttention, PostAttention):
    def __init__(self, config, bias: TransformerLayerVariable):
        super().__init__(config, bias)
        self.config = config
        self.r = HeadProjection(config)

    def forward(self, h, r, seg_mat, attn_mask, mems):
        """Multi-head attention with relative positional encoding."""

        scale = 1 / (self.config.model.head_dim ** 0.5)
        cat = torch.cat([mems, h], 0) if mems is not None and len(mems) > 1 else h

        # content heads
        q_head_h, k_head_h, v_head_h = HeadAttention.forward(self, h, cat, cat)

        # positional heads
        k_head_r = self.r.forward(r)

        # core attention ops
        attn_vec = RelativeAttention.forward(self, q_head_h, k_head_h, v_head_h,
                                             k_head_r, seg_mat, attn_mask, scale=scale)

        output = PostAttention.forward(self, h, attn_vec)
        return output
