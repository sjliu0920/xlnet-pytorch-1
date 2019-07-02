import torch

from xlnet.model.attention.core.head import HeadAttentionOutput
from xlnet.model.attention.multi_head import RelativeMultiHeadAttention
from xlnet.model.transformer.variable import TransformerLayerVariable


class TwoStreamRelativeAttention(RelativeMultiHeadAttention):
    def __init__(self, config, variable: TransformerLayerVariable):
        super().__init__(config, variable)

    def forward(
        self,
        h,
        r,
        seg_mat,
        attn_mask,
        g=None,
        mems=None,
        attn_mask_g=None,
        target_mapping=None,
    ):
        scale = 1 / (self.config.model.head_dim ** 0.5)
        cat = torch.cat([mems, h], 0) if mems is not None and len(mems) > 1 else h

        # positional heads
        k_head_r = self.r.forward(r)

        # ------ h-stream -----
        output_h, head_output_h = self.get_h_output(
            h=h,
            k_head_r=k_head_r,
            cat=cat,
            scale=scale,
            seg_mat=seg_mat,
            attn_mask=attn_mask,
        )
        # ------g-stream -----
        output_g = self.get_g_output(
            g=g,
            head_output_h=head_output_h,
            k_head_r=k_head_r,
            target_mapping=target_mapping,
            seg_mat=seg_mat,
            attn_mask_g=attn_mask_g,
            scale=scale,
        )

        return output_h, output_g

    def get_h_output(self, h, k_head_r, cat, scale, seg_mat, attn_mask):
        # content heads
        head_output = super().HeadAttention.forward(self, h, cat, cat)

        # core attention ops
        attn_vec = super().RelativeAttention.forward(
            self, head_output, k_head_r, seg_mat, attn_mask, scale=scale
        )

        output = super().PostAttention.forward(self, h, attn_vec)
        return output, head_output

    def get_g_output(
        self,
        g,
        head_output_h: HeadAttentionOutput,
        k_head_r,
        target_mapping,
        seg_mat,
        attn_mask_g,
        scale,
    ):
        q_head_g = self.q.forward(g)

        # core attention ops
        if target_mapping is not None:
            q_head_g = torch.einsum("mbnd,mlb->lbnd", q_head_g, target_mapping)

        head_output_h.q_head = q_head_g
        attn_vec_g = super().RelativeAttention.forward(
            self,
            head_output=head_output_h,
            k_head_r=k_head_r,
            seg_mat=seg_mat,
            attn_mask=attn_mask_g,
            scale=scale,
        )

        if target_mapping is not None:
            attn_vec_g = torch.einsum("lbnd,mlb->mbnd", attn_vec_g, target_mapping)

        # post processing
        output_g = self.post_attn(g, attn_vec_g)
        return output_g
