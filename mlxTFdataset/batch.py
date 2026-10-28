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

class GptBatch:
    def __init__(self, src, pad=PAD):
        full_seq = seq_padding(src)
        # print("==src==", src.shape)
        x = full_seq[:,:-1] # 输入
        self.tokens = x
        # print("==x==", self.tokens.shape)
        self.labels = full_seq[:, 1:] #标签，错动一位
        B, T_x = self.tokens.shape
        # attn_mask 是给 input 用的（即x）
        self.attn_mask = mlx_make_causal_mask(T_x, self.tokens)
        # loss_mask 是给 label 用的
        self.loss_mask = mlx_make_loss_mask(self.labels, ID_ZH, PAD)
        # print("==attn_mask==", self.attn_mask.shape)
        # print("==loss_mask==", self.loss_mask.shape)
        






