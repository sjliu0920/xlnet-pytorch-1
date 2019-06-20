import torch
import torch.nn as nn

from .positional import PositionEmbedding


class RelativePositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.model.hidden_size
        self.positional_embed = PositionEmbedding()
        self.attn_type = config.model.attn_type
        self.is_bidirectional = config.model.is_bidirectional
        self.clamp_len = config.data.clamp_len

    def forward(self, qlen, klen, batch_size: int = None):
        frequency_seq = torch.range(0, self.hidden_size, 2.0)
        inverse_frequency = (1 / (10000 ** (frequency_seq / self.hidden_size))).float()

        if self.attn_type not in ['bi', 'uni']:
            raise ValueError(f'Unknown `attn_type` {self.attn_type}')

        begin, end = klen, -qlen if self.attn_type == 'bi' else -1
        pos_embed = self.get_positional_embed(self.is_bidirectional, inverse_frequency, begin, end,
                                              self.clamp_len, batch_size)
        return pos_embed

    def get_positional_embed(self, is_bidirectional, inv_freq, begin, end, clamp_len, batch_size):
        if is_bidirectional:
            return self.get_bi_directional_positional_embed(inv_freq, begin, end, clamp_len, batch_size)
        return self.get_default_positional_embed(inv_freq, begin, end, clamp_len, batch_size)

    def get_bi_directional_positional_embed(self, inv_freq, begin, end, clamp_len, batch_size):
        fwd_pos_seq = torch.range(begin, end, -1.0)
        bwd_pos_seq = torch.range(-begin, -end, 1.0)

        if clamp_len > 0:
            fwd_pos_seq = torch.clamp(fwd_pos_seq, -clamp_len, clamp_len)
            bwd_pos_seq = torch.clamp(bwd_pos_seq, -clamp_len, clamp_len)

        embed_batch_size = batch_size // 2 if batch_size else None
        fwd_pos_emb = self.positional_embed.forward(fwd_pos_seq, inv_freq, embed_batch_size)
        bwd_pos_emb = self.positional_embed.forward(bwd_pos_seq, inv_freq, embed_batch_size)

        pos_embed = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
        return pos_embed

    def get_default_positional_embed(self, inv_freq, begin, end, clamp_len, batch_size):
        fwd_pos_seq = torch.range(begin, end, -1.0)

        if clamp_len > 0:
            fwd_pos_seq = torch.clamp(fwd_pos_seq, -clamp_len, clamp_len)

        pos_emb = self.positional_embed.forward(fwd_pos_seq, inv_freq, batch_size)
        return pos_emb
