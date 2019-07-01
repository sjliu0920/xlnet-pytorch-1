import torch.nn as nn

from xlnet.model.attention.multi_head import RelativeMultiHeadAttention
from xlnet.model.attention.two_stream import TwoStreamRelativeAttention
from xlnet.model.dense.fnn import PositionWisedFNN
from xlnet.model.transformer.variable import TransformerLayerVariable


class TransformerLayer(nn.Module):
    def __init__(self, config, variable: TransformerLayerVariable):
        super().__init__()
        self.ff = PositionWisedFNN(config)
        self.with_input_query = config.data.with_intput_query

        if self.with_input_query:
            self.rel_attn = TwoStreamRelativeAttention(config, variable)
        else:
            self.rel_attn = RelativeMultiHeadAttention(config, variable)

    def forward_with_input_query(
        self, h, g, seg_mat, pos_embed, mem, target_mapping, non_tgt_mask, attn_mask
    ):
        self.rel_attn: TwoStreamRelativeAttention
        output_h, output_g = self.rel_attn.forward(
            h=h,
            g=g,
            r=pos_embed,
            mems=mem,
            seg_mat=seg_mat,
            attn_mask_h=non_tgt_mask,
            attn_mask_g=attn_mask,
            target_mapping=target_mapping,
        )

        output_g = self.ff.forward(output_g)
        output_h = self.ff.forward(output_h)
        return output_g, output_h

    def forward_without_input_query(self, h, seg_mat, pos_embed, mem, attn_mask):
        self.rel_attn: RelativeMultiHeadAttention
        output_h = self.rel_attn.forward(
            h=h, r=pos_embed, seg_mat=seg_mat, attn_mask=attn_mask, mems=mem
        )
        output_h = self.ff.forward(output_h)
        return output_h

    def forward(self, *input):
        raise NotImplementedError
