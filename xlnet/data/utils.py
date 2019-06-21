"""
Below codes are originally based on
https://github.com/zihangdai/xlnet/blob/master/data_utils.py, however, don't support python 2.
"""
import os
from typing import Dict

import numpy as np

from xlnet.utils import logger
from xlnet.data import preprocessing
from xlnet.data.indexer import TokenIndexer

special_symbols = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "<cls>": 3,
    "<sep>": 4,
    "<pad>": 5,
    "<mask>": 6,
    "<eod>": 7,
    "<eop>": 8,
}

VOCAB_SIZE = 32000
UNK_ID = special_symbols["<unk>"]
CLS_ID = special_symbols["<cls>"]
SEP_ID = special_symbols["<sep>"]
MASK_ID = special_symbols["<mask>"]
EOD_ID = special_symbols["<eod>"]


def format_filename(prefix,
                    batch_size,
                    seq_len,
                    bi_data,
                    suffix,
                    mask_alpha=5,
                    mask_beta=1,
                    reuse_len=None,
                    uncased=False,
                    fixed_num_predict=None):
    if reuse_len is None:
        reuse_len_str = ""
    else:
        reuse_len_str = "reuse-{}.".format(reuse_len)
    if not uncased:
        uncased_str = ""
    else:
        uncased_str = "uncased."
    if bi_data:
        bi_data_str = "bi"
    else:
        bi_data_str = "uni"
    if fixed_num_predict is not None:
        fnp_str = "fnp-{}.".format(fixed_num_predict)
    else:
        fnp_str = ""

    file_name = "{}.bsz-{}.seqlen-{}.{}{}{}.alpha-{}.beta-{}.{}{}".format(
        prefix, batch_size, seq_len, reuse_len_str, uncased_str, bi_data_str,
        mask_alpha, mask_beta, fnp_str, suffix)

    return file_name
