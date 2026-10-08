import numpy as np
import copy
import os
import mlx.core as mx
import mlx.nn as nn
from .config import *
from .mask import *
from .padding import *

class Batch:
    def __init__(self, src, tgt, pad=PAD):
        src = seq_padding(src)
        self.src = mx.array(src, dtype=mx.int32)
        self.src_mask = mlx_make_src_pad_mask(src)
        #
        bos_tgt = [[B_O_S] + seq for seq in tgt]
        self.tgt_in = mx.array(seq_padding(bos_tgt), dtype=mx.int32)
        self.tgt_mask = mlx_make_tgt_causal_pad_mask(self.tgt_in)
        #
        tgt_eos = [seq + [E_O_S] for seq in tgt]
        self.tgt_label = mx.array(seq_padding(tgt_eos), dtype=mx.int32)
        #
        tgt = seq_padding(tgt)
        self.tgt = mx.array(tgt, dtype=mx.int32)
        self.ntokens = (self.tgt_label != pad).sum()
        