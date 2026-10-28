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

def mlx_make_causal_mask(seq_len, src, pad=PAD):
    # 下三角因果mask
    causal = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
    causal_mask = causal[None, None, :, :] # [1,1,L,L]
    #print("----mask---", mask.shape)
    is_pad = (src == pad)
    # pad_mask: [B, L]，padding位置=-1e9
    pad_mask = mx.where(is_pad, mx.array(-float("inf"), dtype=mx.float32), mx.array(0.0, dtype=mx.float32))
    # [B, L]
    #print("----pad_mask---", pad_mask.shape) 
    pad_mask = pad_mask[:, None, None, :]
    combined_mask = causal_mask + pad_mask
    # [B, 1, L, L]
    #print("----combined_mask---", combined_mask.shape)
    return combined_mask

def mlx_make_loss_mask(tokens, zh_id, pad_id):
    """
    tokens: [B, L]
    zh_id: <|zh|> 的token id
    返回 loss_mask [B, L], float 0/1
    规则：<|zh|>以及它后面 → 1；<|zh|>前面 → 0；pad位置强制0
    """
    tokens = mx.array(tokens)
    B, L = tokens.shape
    # 找到每个样本里第一个 <|zh|> 的下标
    hit = (tokens == zh_id)
  
    idx = mx.where(hit, mx.arange(L), L)
    zh_pos = idx.min(-1)
    zh_pos = mx.expand_dims(zh_pos, -1)
    # 没有命中 <|zh|> 的样本，zh_pos 设成 L，全部mask=0，不计算loss

    # 位置 >= zh_pos → 1
    pos_arange = mx.arange(L)[None, :]
    mask = mx.where(pos_arange >= zh_pos, 1.0, 0.0)
    # pad位置强制置0
    pad_mask = (tokens != pad_id).astype(mx.float32)
    loss_mask = mask * pad_mask
    return loss_mask




