import torch
import torch.nn as nn

from xlnet.model.utils.mask import create_mask
from xlnet.model.embed.relative_positional import RelativePositionalEmbedding
from xlnet.model.transformer.bias import TransformerXLBias
from xlnet.model.transformer.layer import TransformerXLLayer


class TransformerXL(TransformerXLBias):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.word_embedding = nn.Embedding(config.model.vocab_size, config.model.hidden_size)
        self.segment_embedding = nn.Parameter(
            torch.rand([config.model.num_layers, 2, config.model.head_num, config.model.head_dim]))
        self.positional_embedding = RelativePositionalEmbedding(config)

        self.dropout = nn.Dropout(config.model.dropout_prob)
        self.mask_embed = nn.Parameter(torch.rand([1, 1, config.model.hidden_size], dtype=torch.float))
        self.layers = nn.ModuleList([TransformerXLLayer(config, self) for _ in range(self.config.model.num_layers)])

    def __getattr__(self, item):
        split_items = item.split("layer_")
        if len(split_items) == 2:
            return self.layers[int(split_items[1])]
        raise AttributeError(f"module {__name__} has no attribute {item}")

    def forward(self, inp_k, inp_q, segment_id, input_mask, perm_mask, target_mapping, mems=[]):
        batch_size, query_len = inp_k.size(1), inp_k.size(0)
        memory_len = mems[0].size(0) if mems else 0
        klen = memory_len + query_len

        attn_mask, data_mask, non_tgt_mask = self.get_mask(batch_size, query_len, memory_len, input_mask, perm_mask)
        word_embed_k = self.word_embedding.forward(inp_k)
        output_h, output_g = self.get_mask_embed(inp_q, word_embed_k, target_mapping, batch_size)

        seg_embed = self.get_segment_embed(segment_id, memory_len, batch_size)
        pos_embed = self.get_position_embed(query_len, klen)
        mems = [None] * self.config.model.num_layers if mems is None else mems

    def get_position_embed(self, query_len, key_len):
        pos_embed = self.positional_embedding.forward(query_len, key_len)
        pos_embed = self.dropout.forward(pos_embed)
        return pos_embed

    def get_segment_embed(self, seg_id, mlen, batch_size):
        mem_pad = torch.zeros(mlen, batch_size)
        cat_ids = torch.cat([mem_pad, seg_id], 0)
        seg_mat = ~(torch.eq(seg_id[None, :], (cat_ids[None, :]))).float()
        # todo: segment embedding calculation with one_hot
        return seg_mat

    def get_mask_embed(self, input_q, word_emb_k, target_mapping, batch_size):
        if target_mapping is not None:
            word_embed_q = input_q.repeat(target_mapping.size(0), batch_size, 1)
        else:
            input_q_ext = input_q[:, :, None]
            word_embed_q = input_q_ext * self.mask_embed + (1 - input_q_ext) * word_emb_k

        output_h = self.dropout(word_emb_k)
        output_g = self.dropout(word_embed_q) if input_q is not None else None
        return output_h, output_g

    def get_mask(self, batch_size, query_len, memory_len, input_mask, perm_mask):
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
            attn_mask = create_mask(qlen, mlen, self.config.model.same_attn_length)[:, :, None, None]
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
