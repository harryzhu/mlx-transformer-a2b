import mlx.core as mx
import mlx.nn as nn
from .config import *
from .padding import *
import os

def mlx_make_tgt_causal_pad_mask(tgt, pad=PAD):
    # tgt shape: [B, L]
    L = tgt.shape[-1]
    # 1. 因果mask [L, L]
    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
    # 2. padding mask：同时mask query行 + key列 [B, L, L]
    # print("causal_mask:",causal_mask, causal_mask.shape)
    is_pad = (tgt == pad)
    # print("is_pad:", is_pad.shape)
    query_pad_mask = mx.where(is_pad, mx.array(-float("inf"), dtype=mx.float32), mx.array(0.0, dtype=mx.float32))[:,:, None] # [B, L, 1]
    key_pad_mask = mx.where(is_pad, mx.array(-float("inf"), dtype=mx.float32), mx.array(0.0, dtype=mx.float32))[:, None, :] # [B, 1, L]
    pad_mask = mx.maximum(query_pad_mask, key_pad_mask)
    # 3. 合并因果mask和padding mask
    #combined_mask = mx.maximum(causal_mask, pad_mask)
    # 两个 mask 相加，任意一个是`-inf`，结果就是`-inf`，实现同时屏蔽未来 token + pad token。
    #`mx.maximum(0, -inf) = 0`，**PAD 的负无穷直接丢失！PAD 不会被屏蔽**
    combined_mask = causal_mask + pad_mask
    combined_mask = combined_mask[:,None,:]
    #print("mlx_make_tgt_causal_pad_mask: ", combined_mask.shape)
    #[B, 1, L, L]
    return combined_mask


def mlx_make_src_pad_mask(src, pad=PAD):
    is_pad = (src == pad)
    pad_mask = mx.where(is_pad, mx.array(-float("inf"), dtype=mx.float32), mx.array(0.0, dtype=mx.float32))
    pad_mask = pad_mask[:, None, None,:]
    #[B, 1, 1, L]
    return pad_mask

