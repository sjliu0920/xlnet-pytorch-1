import torch
import torch.nn as nn

from xlnet.model.embed.relative_positional import RelativePositionalEmbedding
from xlnet.model.transformer.layer import TransformerLayer
from xlnet.model.transformer.variable import (
    TransformerLayerVariable,
    TransformerVariable,
)
from xlnet.model.utils.mask import MaskingUtil


class TransformerXL(TransformerVariable):
    """TransformerXL Class"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.mask_util = MaskingUtil(config)

        self.word_embedding = nn.Embedding(
            config.model.vocab_size, config.model.hidden_size
        )
        self.positional_embedding = RelativePositionalEmbedding(config)

        self.dropout = nn.Dropout(config.model.dropout_prob)
        self.mask_embed = nn.Parameter(
            torch.rand([1, 1, config.model.hidden_size], dtype=torch.float)
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    config, variable=TransformerLayerVariable(config, self, layer_id)
                )
                for layer_id in range(self.config.model.num_layers)
            ]
        )

    def forward(
        self,
        inp_k,
        segment_id,
        input_mask,
        perm_mask,
        mems=None,
        inp_q=None,
        target_mapping=None,
        reuse_len: int = None,
    ):
        # get size of inputs and preparation for models
        batch_size, query_len, memory_len, klen, mems = self.forward_prefix(inp_k, mems)

        # make attention mask using input sizes
        attn_mask, data_mask, non_tgt_mask = self.mask_util.get_mask(
            batch_size, query_len, memory_len, input_mask, perm_mask
        )

        # get word embedding depends on the input
        if inp_q is not None:
            output_h, output_g = self.get_word_embed(
                inp_k, batch_size, target_mapping, inp_q
            )
        else:
            output_h, output_g = (
                self.get_word_embed(inp_k, batch_size, target_mapping),
                None,
            )

        # get segment matrix and position embedding
        seg_mat = self.get_segment_matrix(segment_id, memory_len, batch_size)
        pos_embed = self.get_position_embed(query_len, klen)

        # list of new memories
        new_mems = list()

        # run over the layers
        for i, layer in enumerate(self.layers):
            layer: TransformerLayer
            new_mems.append(self._cache_mem(output_h, mems[i], memory_len, reuse_len))

            if inp_q is not None:
                output_h, output_g = layer.forward_with_input_query(
                    h=output_h,
                    g=output_g,
                    seg_mat=seg_mat,
                    pos_embed=pos_embed,
                    mem=mems[i],
                    attn_mask=attn_mask,
                    target_mapping=target_mapping,
                    non_tgt_mask=non_tgt_mask,
                )
            else:
                output_h = layer.forward_without_input_query(
                    h=output_h,
                    seg_mat=seg_mat,
                    pos_embed=pos_embed,
                    mem=mems[i],
                    attn_mask=attn_mask,
                )

        output_target = output_g if inp_q is not None else output_h
        output = self.dropout(output_target)
        return output, new_mems

    def forward_prefix(self, inp_k, mems):
        """

        :param inp_k:
        :param mems:
        :param input_mask:
        :param perm_mask:
        :return:
        """
        batch_size, query_len = inp_k.size(1), inp_k.size(0)
        memory_len = mems[0].size(0) if mems else 0
        klen = memory_len + query_len
        mems = [None] * self.config.model.num_layers if mems is None else mems

        return batch_size, query_len, memory_len, klen, mems

    def get_position_embed(self, query_len, key_len):
        """

        :param query_len:
        :param key_len:
        :return:
        """
        pos_embed = self.positional_embedding.forward(query_len, key_len)
        pos_embed = self.dropout.forward(pos_embed)
        return pos_embed

    def get_segment_matrix(self, seg_id, mlen, batch_size):
        """

        :param seg_id:
        :param mlen:
        :param batch_size:
        :return:
        """
        mem_pad = torch.zeros(mlen, batch_size)
        cat_ids = torch.cat([mem_pad, seg_id], 0)
        seg_mat = ~(torch.eq(seg_id[None, :], (cat_ids[None, :]))).float()
        # todo: segment embedding calculation with one_hot
        return seg_mat

    def get_word_embed(self, input_k, batch_size, input_q=None, target_mapping=None):
        """

        :param input_k:
        :param batch_size:
        :param input_q:
        :param target_mapping:
        :return:
        """
        word_emb_k = self.word_embedding.forward(input_k)
        output_h = self.dropout(word_emb_k)

        if input_q is None:
            return output_h

        if target_mapping is not None:
            word_embed_q = self.mask_embed.repeat(target_mapping.size(0), batch_size, 1)
        else:
            input_q_ext = input_q[:, :, None]
            word_embed_q = (
                input_q_ext * self.mask_embed + (1 - input_q_ext) * word_emb_k
            )

        output_g = self.dropout(word_embed_q)
        return output_h, output_g

    def _cache_mem(self, curr_out, prev_mem, mem_len, reuse_len=None):
        """

        :param curr_out:
        :param prev_mem:
        :param mem_len:
        :param reuse_len:
        :return:
        """
        if mem_len is None or mem_len == 0:
            return None
        else:
            if reuse_len is not None and reuse_len > 0:
                curr_out = curr_out[:reuse_len]

            if prev_mem is None:
                new_mem = curr_out[-mem_len:]
            else:
                new_mem = torch.cat([prev_mem, curr_out], 0)[-mem_len:]

        new_mem.requires_grad = False
        return new_mem

    def __getattr__(self, item):
        split_items = item.split("layer_")
        if len(split_items) == 2:
            return self.layers[int(split_items[1])]
        raise AttributeError(f"module {__name__} has no attribute {item}")
