import math
import mlx.core as mx
import mlx.nn as nn
import copy
from .config import *


class GPTBlock(nn.Module):
    """
    Decoder Layer
    1. Decoder Self-Attention：使用RoPE
    2. Cross-Attention：不加RoPE
    """
    def __init__(self, norm_first: bool, rope):
        super().__init__()
        self.norm_first = norm_first
        self.rope = rope
        self.dims = DIM_MODEL
        self.num_heads = NUM_HEAD
        self.head_dim = DIM_MODEL // NUM_HEAD
        self.scale = math.sqrt(self.head_dim)
        # Self Attention
        self.w_qkv = nn.Linear(DIM_MODEL, DIM_MODEL * 3)
        # ffn
        self.ffn = nn.Sequential(
            nn.Linear(DIM_MODEL, DIM_MODEL * 4),
            nn.GELU(),
            nn.Linear(DIM_MODEL * 4, DIM_MODEL),
        )
        self.norm1 = nn.RMSNorm(DIM_MODEL)
        self.norm2 = nn.RMSNorm(DIM_MODEL)  
        self.dropout1 = nn.Dropout(DROPOUT)
        self.dropout2 = nn.Dropout(DROPOUT)

    
    def self_attn_rope(self, x, mask):
        B, L, D = x.shape
        qkv = self.w_qkv(x)
        # print("1.qkv:", qkv.shape) #[B, L, DIM_MODEL*3]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        # print("2.qkv.reshape:", qkv.shape) #[B, L, 3, H, head_dim]
        # 调换维度：B, 3, H, L, Dh
        qkv = qkv.transpose(0, 2, 3, 1, 4) 
        # print("3.qkv.transpose:", qkv.shape) #[B, 3, H, L, head_dim]
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2] # shape [B, H, L, Dh]
        
        
        # RoPE 只作用在 Q、K
        q = self.rope(q)
        k = self.rope(k)
            
        attn_scores = q @ k.transpose(0,1,3,2) / self.scale # [B, H, L, L]
        attn_scores = attn_scores + mask
        attn_weights = mx.softmax(attn_scores, axis=-1)
        attn_out = attn_weights @ v
         
        attn_out = attn_out.transpose(0,2,1,3).reshape(B, L, D)
        return attn_out

    def __call__(self, x, attn_mask):
        if self.norm_first:
            # Decoder Self-Attention with RoPE
            h = self.norm1(x)
            h = self.self_attn_rope(h, attn_mask)
            x = x + self.dropout1(h)

            # FFN
            h = self.norm2(x)
            h = self.ffn(h)
            x = x + self.dropout2(h)
        else:
            h = self.self_attn_rope(x, attn_mask)
            x = self.norm1(x + self.dropout1(h))
            #
            h = self.ffn(x)
            x = self.norm2(x + self.dropout2(h))
        return x

class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embedding = nn.Embedding(VOCAB_SIZE_GPT, DIM_MODEL)
        # rope 的 dims 必须为偶数（RoPE 需要两两一组做平面旋转）
        assert DIM_MODEL%NUM_HEAD == 0
        assert (DIM_MODEL//NUM_HEAD)%2 == 0
        self.rope = nn.RoPE(dims=DIM_MODEL // NUM_HEAD, base=10000.0)
        self.proj = nn.Linear(DIM_MODEL, VOCAB_SIZE_GPT, bias=False)
        self.scale = math.sqrt(DIM_MODEL)
        self.blocks = [
            GPTBlock(norm_first=True, rope=self.rope)
            for _ in range(NUM_LAYER)
            ]

    def __call__(self, tokens, attn_mask): 
        #B, L = tokens.shape
        x = self.tok_embedding(tokens) * self.scale
        for block in self.blocks:
            x = block(x, attn_mask)
        logits = self.proj(x)
        return logits

    def init_kv_cache(self, batch_size):
        cache = []
        for _ in range(NUM_LAYER):
            k_cache = mx.zeros((batch_size, NUM_HEAD, 0, DIM_MODEL//NUM_HEAD))
            v_cache = mx.zeros((batch_size, NUM_HEAD, 0, DIM_MODEL//NUM_HEAD))
            cache.append((k_cache,v_cache))
        return cache

