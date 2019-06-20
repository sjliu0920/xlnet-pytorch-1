import torch.nn as nn

from xlnet.model.attention.two_stream import TwoStreamRelativeAttention
from xlnet.model.attention.multi_head import RelativeMultiHeadAttention
from xlnet.model.dense.fnn import PositionWisedFNN
from xlnet.model.transformer.bias import TransformerXLBias


class TransformerXLLayer(nn.Module):
    def __init__(self, config, bias: TransformerXLBias, with_input_query: bool = True):
        super().__init__()
        if with_input_query:
            self.rel_attn = TwoStreamRelativeAttention(config, bias)
        else:
            self.rel_attn = RelativeMultiHeadAttention(config, bias)

        self.ff = PositionWisedFNN(config)
        self.with_input_query = with_input_query

    def forward(self, h, g, seg_mat, seg_embed, pos_embed, mem, attn_mask, target_mapping=None, non_tgt_mask=None):
        forward_input = (h, g, seg_mat, seg_embed, pos_embed, mem, target_mapping, non_tgt_mask, attn_mask)

        if self.with_input_query:
            return self.forward_with_input_query(*forward_input)
        else:
            return self.forward_without_input_query(*forward_input)

    def forward_with_input_query(self, h, g, seg_mat, seg_embed, pos_embed, mem,
                                 target_mapping, non_tgt_mask, attn_mask):
        self.rel_attn: TwoStreamRelativeAttention
        output_h, output_g = self.rel_attn.forward(
            h=h, g=g, r=pos_embed,
            mems=mem, seg_mat=seg_mat, seg_embed=seg_embed,
            attn_mask_h=non_tgt_mask, attn_mask_g=attn_mask,
            target_mapping=target_mapping
        )
        output_g = self.ff.forward(output_g)
        output_h = self.ff.forward(output_h)
        return output_g, output_h

    def forward_without_input_query(self, h, g, seg_mat, seg_embed, pos_embed, mem, attn_mask,
                                    target_mapping=None, non_tgt_mask=None):
        self.rel_attn: RelativeMultiHeadAttention
        output_h = self.rel_attn.forward(
            h=h, r=pos_embed, seg_mat=seg_mat, seg_embed=seg_embed, attn_mask=attn_mask, mems=mem
        )
        output_h = self.ff.forward(output_h)
        return output_h
