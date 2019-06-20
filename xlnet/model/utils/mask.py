import torch


def create_mask(qlen, mlen, same_length=False):
    """create causal attention mask."""
    attn_mask = torch.ones([qlen, qlen], dtype=torch.float)
    mask_u = torch.triu(attn_mask, 0)
    mask_dia = torch.tril(attn_mask, 0) - torch.tril(attn_mask, -1)
    attn_mask_pad = torch.zeros([qlen, mlen], dtype=torch.float)

    ret = torch.cat([attn_mask_pad, mask_u - mask_dia], 1)

    if same_length:
        mask_l = torch.tril(attn_mask, -1)
        ret = torch.cat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)

    return ret
