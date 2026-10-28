import math
import time
import os
import copy
from functools import partial
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from .config import *
import pickle
from mlxTFdataset.config import *
from mlxTFdataset.mask import *
   

def gpt_generate(model, src, use_greedy=False, max_gen_len=128):
    """
    prompt_tokens: list[list[int]] / np.ndarray, shape [B, L_prompt]
    无KV Cache版本，每次完整序列前向，简单好调试
    """
    model.eval()
    tokens = mx.array(src, dtype=mx.int32)
    #max_gen_len = tokens.shape[1] * 2

    B = tokens.shape[0]
    done = mx.zeros((B, 1), dtype=mx.bool_)

    penalty = "frequency"

    temperature = 0.9
    top_p = 0.9
    repetition_penalty = 1.2
    frequency_penalty = 0.5

    for _ in range(max_gen_len):
        L_cur = tokens.shape[1]
        # 构造当前完整序列的 causal attn_mask
        attn_mask = mlx_make_causal_mask(L_cur, tokens, pad=PAD)

        # 无kv_cache，不需要offset
        logits = model(tokens, attn_mask)
        #print("logits:", logits.shape)
        #取最后一个位置的 logits， 预测下一个token
        next_logit = logits[:,-1,:] # [B, vocab]
        

        # 重复惩罚
        if use_greedy:
            next_id = mx.argmax(next_logit, axis=-1, keepdims=True)
        else:
            prev_tokens = tokens
            if penalty  == "frequency":
                next_id = sample_next_token_frequency_penalty(
                    next_logit,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    prev_tokens=prev_tokens
                )
            else:
                next_id = sample_next_token_repetition_penalty(
                    next_logit,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    prev_tokens=prev_tokens
                )
        #print("next_id:",next_id)
        tokens = mx.concatenate([tokens, next_id], axis=-1)
        #print("tokens:",tokens)

        # 全部样本都输出EOT则提前停止
        new_end = (next_id == ID_END)
        done = mx.logical_or(done, new_end)
        if mx.all(done).item():
            break

    return tokens



def sample_next_token_repetition_penalty(logits: mx.array, temperature=1.0, top_p=0.9, repetition_penalty=1.2, prev_tokens=None):
    """
    logits: [B, vocab_size]
    temperature: 温度，0=等价贪心argmax
    top_p: nucleus采样阈值
    repetition_penalty: >1 惩罚已出现token，抑制重复
    prev_tokens: [B, L] 已经生成的token ids，用于重复惩罚
    return: next_token [B,1] int32
    """
    logits = mx.array(logits) # 防止原地修改原始logits
    B, V = logits.shape
    # 重复惩罚
    if repetition_penalty > 1.0 and prev_tokens is not None:
        for b in range(B):
            # 【修复】只取当前batch已生成token
            tokens_b = prev_tokens[b].reshape(-1)
            seen = set(tokens_b.tolist())
            for tok_id in seen:
                val = logits[b, tok_id]
                if val > 0:
                    logits[b, tok_id] /= repetition_penalty
                else:
                    logits[b, tok_id] *= repetition_penalty

    # 温度缩放 / 贪心分支
    if temperature > 0:
        logits = logits / temperature
    else:
        return mx.argmax(logits, axis=-1, keepdims=True)

    # Top-p Nucleus采样
    sorted_indices = mx.argsort(logits, axis=-1)[:,::-1]
    sorted_logits = logits[mx.arange(B)[:, None], sorted_indices]
    probs = mx.softmax(sorted_logits, axis=-1)
    cumulative_probs = mx.cumsum(probs, axis=-1)

    mask = cumulative_probs <= top_p
    # 保证至少保留1个token
    mask = mx.concatenate([mx.ones((B,1), dtype=mx.bool_), mask[:, :-1]], axis=-1)

    filtered_logits = mx.full_like(logits, -float("inf"))
    filtered_logits[mx.arange(B)[:, None], sorted_indices] = mx.where(mask, sorted_logits, -float("inf"))

    next_token = mx.random.categorical(filtered_logits, num_samples=1)
    return next_token


def sample_next_token_frequency_penalty(
    logits: mx.array,
    temperature=1.0,
    top_p=0.9,
    frequency_penalty=0.0,
    prev_tokens=None
):
    """
    logits: [B, vocab_size]
    temperature: 温度，0=贪心argmax
    top_p: nucleus采样阈值
    frequency_penalty: >0，对出现次数越多的token，扣减logit越多；推荐0.2~0.8
    prev_tokens: [B, L] int，已经生成的token ids
    return: next_token [B,1] int32
    """
    logits = mx.array(logits)
    B, V = logits.shape

    # ========== Frequency Penalty，用scatter_add实现bincount ==========
    if frequency_penalty > 0 and prev_tokens is not None:
        for b in range(B):
            tokens_b = prev_tokens[b] # shape [L]
            counts = mx.zeros(V, dtype=mx.float32)
            ones = mx.ones_like(tokens_b, dtype=mx.float32)
            counts = counts.at[tokens_b].add(ones)
            logits[b] = logits[b] - frequency_penalty * counts

    # Temperature
    if temperature > 0:
        logits = logits / temperature
    else:
        return mx.argmax(logits, axis=-1, keepdims=True)

    # Top-p Nucleus sampling
    sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]
    sorted_logits = logits[mx.arange(B)[:, None], sorted_indices]
    probs = mx.softmax(sorted_logits, axis=-1)
    cumulative_probs = mx.cumsum(probs, axis=-1)

    mask = cumulative_probs <= top_p
    mask = mx.concatenate([mx.ones((B,1), dtype=mx.bool_), mask[:, :-1]], axis=-1)
    filtered_logits = mx.full_like(logits, -float("inf"))
    filtered_logits[mx.arange(B)[:, None], sorted_indices] = mx.where(mask, sorted_logits, -float("inf"))

    next_token = mx.random.categorical(filtered_logits, num_samples=1)
    return next_token

