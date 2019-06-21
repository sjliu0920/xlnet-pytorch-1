import torch


class MaskingUtil:
    def __init__(self, config):
        self.config = config

    def get_mask(self, batch_size: int, query_len: int, memory_len: int, input_mask, perm_mask):
        attn_mask = self.get_attn_mask(query_len, memory_len)
        data_mask = self.get_data_mask(input_mask, perm_mask)

        if data_mask is not None:
            mems_mask = torch.zeros([data_mask.size(0), memory_len, batch_size])
            data_mask = torch.cat([mems_mask, data_mask, 1])
            attn_mask = attn_mask + data_mask[:, :, :, None] if attn_mask else data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0).float()

        if attn_mask is not None:
            non_tgt_mask = -1 * torch.eye(query_len).float()
            non_tgt_mask = torch.cat([torch.zeros(query_len, memory_len, dtype=torch.float), non_tgt_mask], -1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).float()
        else:
            non_tgt_mask = None

        return attn_mask, data_mask, non_tgt_mask

    def get_attn_mask(self, qlen, mlen):
        attn_type = self.config.model.attn_type
        if attn_type == "uni":
            attn_mask = self.create_mask(qlen, mlen, self.config.model.same_attn_length)[:, :, None, None]
        elif attn_type == "bi":
            attn_mask = None
        else:
            raise ValueError(f'Unsupported attention type: {attn_type}')
        return attn_mask

    def get_data_mask(self, input_mask, perm_mask):
        if input_mask is not None:
            return input_mask[None] if perm_mask is None else input_mask[None] + perm_mask
        if perm_mask is not None:
            return perm_mask if input_mask is None else None

    def create_mask(self, qlen, mlen, same_length=False):
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
