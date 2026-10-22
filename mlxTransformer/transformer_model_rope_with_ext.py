import math
import mlx.core as mx
import mlx.nn as nn
import time
import copy
from .config import *
from .alt_sdpa_wrapper import *
from .speedup_a2b_ext import *

def bhld_check(m, B, H, L, Dh):
    b,h,l,dh = m.shape
    assert b == B , "the 1st dim should be B"
    assert h == H , "the 2nd dim should be H"
    assert l == L , "the 3rd dim should be L"
    assert dh == Dh , "the 4th dim should be head_dim"
    return None

def bld_check(m, B, L, D):
    b,l,d = m.shape
    assert b == B , "the 1st dim should be B"
    assert l == L , "the 2nd dim should be L"
    assert d == D , "the 3rd dim should be DIM_MODEL"
    return None

class EncoderLayer(nn.Module):
    """Encoder Self-Attn：正弦PE，不用RoPE"""
    def __init__(self, norm_first: bool):
        super().__init__()
        self.norm_first = norm_first
        self.attn = nn.MultiHeadAttention(DIM_MODEL, NUM_HEAD)
        self.ffn = nn.Sequential(
            nn.Linear(DIM_MODEL, DIM_MODEL * 4),
            nn.GELU(),
            nn.Linear(DIM_MODEL * 4, DIM_MODEL),
        )
        self.norm1 = nn.LayerNorm(DIM_MODEL)
        self.norm2 = nn.LayerNorm(DIM_MODEL)
        self.dropout1 = nn.Dropout(DROPOUT)
        self.dropout2 = nn.Dropout(DROPOUT)

    def __call__(self, x, mask):
        # Pre-norm
        if self.norm_first:
            #h = self.norm1(x)
            h = tf_fn(x=x, fn=self.norm1)
            #h = self.attn(h, h, h, mask=mask)
            h = tf_encode_self_attention(x=h, mask=mask, fn=self.attn)
            #x = x + self.dropout1(h)
            x = tf_add(x, self.dropout1(h))
            #h = self.norm2(x)
            h = tf_fn(x=x, fn=self.norm2)
            #h = self.ffn(h)
            h = tf_fn(x=h, fn=self.ffn)
            #x = x + self.dropout2(h)
            x = tf_add(x, self.dropout2(h))
        else:
            #h = self.attn(x, x, x, mask=mask)
            h = tf_encode_self_attention(x=x, mask=mask, fn=self.attn)
            #x = self.norm1(x + self.dropout1(h))
            x = tf_fn(x=tf_add(x, self.dropout1(h)), fn=self.norm1)
            #h = self.ffn(x)
            h = tf_fn(x=x, fn=self.ffn)
            #x = self.norm2(x + self.dropout2(h))
            x = tf_fn(x = tf_add(x, self.dropout2(h)), fn=self.norm2)
        return x

class DecoderLayer(nn.Module):
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
        self.var_sqrt_head_dim = math.sqrt(self.head_dim)
        # Self Attention
        self.w_qkv = nn.Linear(DIM_MODEL, DIM_MODEL * 3)
        # Cross Attention
        self.cross_attn = nn.MultiHeadAttention(DIM_MODEL, NUM_HEAD)
        self.ffn = nn.Sequential(
            nn.Linear(DIM_MODEL, DIM_MODEL * 4),
            nn.GELU(),
            nn.Linear(DIM_MODEL * 4, DIM_MODEL),
        )
        self.norm1 = nn.RMSNorm(DIM_MODEL)
        self.norm2 = nn.RMSNorm(DIM_MODEL)
        self.norm3 = nn.RMSNorm(DIM_MODEL)     
        self.dropout1 = nn.Dropout(DROPOUT)
        self.dropout2 = nn.Dropout(DROPOUT)
        self.dropout3 = nn.Dropout(DROPOUT)

    
    def self_attn_rope(self, x, mask, kv_cache = None, offset = 0):
        B, L, D = x.shape
        #qkv = self.w_qkv(x)
        qkv = tf_fn(x=x, fn=self.w_qkv)
        # print("1.qkv:", qkv.shape) #[B, L, DIM_MODEL*3]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        # print("2.qkv.reshape:", qkv.shape) #[B, L, 3, H, head_dim]
        # 调换维度：B, 3, H, L, Dh
        qkv = qkv.transpose(0, 2, 3, 1, 4) 
        # print("3.qkv.transpose:", qkv.shape) #[B, 3, H, L, head_dim]
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2] # shape [B, H, L, Dh]
        # 所有 batch，取第二个维度的第 0 组（q）,第 1 组（k）,第 2 组（v）
        #
        # 或者用 mx.split, 但要用一下 squeeze，因为切开维度为：[B, 1, H, L, head_dim]
        # q, k, v = mx.split(qkv, indices_or_sections=3, axis=1)
        # q = mx.squeeze(q, axis=1)
        # k = mx.squeeze(k, axis=1)
        # v = mx.squeeze(v, axis=1)
        #
        # print("4.q:", q.shape) # shape [B, H, L, head_dim]
        # print("4.k:", k.shape)
        # print("4.v:", v.shape)
        if DEBUG:
            bhld_check(q, B, self.num_heads, L, self.head_dim)
            bhld_check(k, B, self.num_heads, L, self.head_dim)
            bhld_check(v, B, self.num_heads, L, self.head_dim)
        
        
        # RoPE 只作用在 Q、K
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        new_cache = None
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = mx.concatenate([prev_k, k], axis=2)
            v = mx.concatenate([prev_v, v], axis=2)
            new_cache = (k, v)
            
        # MHA score
        #attn_out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask)
        # scaled_dot_product_attention 只支持 additive_mask, 不支持 additive_mask + pad mask
        # 所以此处改写法, 速度比 mx.fast.scaled_dot_product_attention 慢一些
        attn_out = alt_sdpa_gpu(q,k,v,mask)
        # attn_out = alt_sdpa_mlx(q,k,v,mask)
        # attn_out = alt_sdpa_cpu(q,k,v,mask)

        if DEBUG:
            bld_check(attn_out, B, L, DIM_MODEL)
        return self.dropout1(attn_out), new_cache

    def __call__(self, x, memory, tgt_mask, memory_mask, kv_cache=None, offset=0):
        if self.norm_first:
            # Decoder Self-Attention with RoPE
            #h = self.norm1(x)
            h = tf_fn(x=x, fn=self.norm1)
            h, new_cache = self.self_attn_rope(h, tgt_mask, kv_cache=kv_cache, offset=offset)
            #x = x + h
            x = tf_add(x, h)

            # Cross Attention
            #h = self.norm2(x)
            h = tf_fn(x=x, fn=self.norm2)
            #h = self.cross_attn(h, memory, memory, memory_mask)
            h = tf_decode_cross_attention(x=h, memory=memory, memory_mask=memory_mask, fn=self.cross_attn)
            #x = x + self.dropout2(h)
            x = tf_add(x, self.dropout2(h))

            # FFN
            #h = self.norm3(x)
            h = tf_fn(x=x, fn=self.norm3)
            #h = self.ffn(h)
            h = tf_fn(x=h, fn=self.ffn)
            #x = x + self.dropout3(h) 
            x = tf_add(x, tf_fn(x=h, fn=self.dropout3))         
        else:
            h, new_cache = self.self_attn_rope(x, tgt_mask, kv_cache=kv_cache, offset=offset)
            #x = self.norm1(x + h)
            x = tf_fn(x = tf_add(x, h), fn=self.norm1)
            #h = self.cross_attn(x, memory, memory, mask=memory_mask)
            h = tf_decode_cross_attention(x=h, memory=memory, memory_mask=memory_mask, fn=self.cross_attn)
            #x = self.norm2(x + self.dropout2(h))
            x = tf_fn(x = tf_add(x, self.dropout2(h)), fn=self.norm2)
            #h = self.ffn(x)
            h = tf_fn(x = x, fn = self.ffn)
            #x = self.norm3(x + self.dropout3(h))
            x = tf_fn(x = tf_add(x, tf_fn(x=h, fn=self.dropout3)), fn=self.norm3)
        return x, new_cache

class TransformerModelRoPE(nn.Module):
    def __init__(self):
        super(TransformerModelRoPE, self).__init__()
        self.src_embedding = nn.Embedding(VOCAB_SIZE_SRC, DIM_MODEL)
        self.tgt_embedding = nn.Embedding(VOCAB_SIZE_TGT, DIM_MODEL)
        self.src_pos_enc = nn.SinusoidalPositionalEncoding(DIM_MODEL)
        # rope 的 dims 必须为偶数（RoPE 需要两两一组做平面旋转）
        assert DIM_MODEL%NUM_HEAD == 0
        assert (DIM_MODEL//NUM_HEAD)%2 == 0
        # 使用上面自定义rope（LLaMA3标准），约48微秒
        #self.rope = RoPE_LLaMA3()
        # 使用mlx内置rope，速度更快，约19微秒
        self.rope = nn.RoPE(dims=DIM_MODEL // NUM_HEAD, base=10000.0)

        self.proj = nn.Linear(DIM_MODEL, VOCAB_SIZE_TGT, bias=False)
        self.scale = math.sqrt(DIM_MODEL)
        self.encoder_layers = [
            EncoderLayer(norm_first=True)
            for _ in range(NUM_LAYER)
            ]
        self.decoder_layers = [
            DecoderLayer(norm_first=True, rope=self.rope)
            for _ in range(NUM_LAYER)
            ]

    def encode(self, src, src_mask):
        #不加这个self.scale缩放，embedding 幅值太大，位置编码被淹没，位置信息失效 → 模型分不清语序，翻译错乱
        #x = self.src_embedding(src) * self.scale
        x = tf_multiply(a=tf_fn(x=src, fn=self.src_embedding), b=self.scale)
        x = tf_add(x, tf_fn(x=mx.arange(src.shape[-1]), fn=self.src_pos_enc))
        x = tf_src_encode(x=x, src_mask = src_mask, src_encoder_layers=self.encoder_layers) 
        return x

    def decode_step(self, tgt, memory, tgt_mask, memory_mask, kv_cache=None): 
        #不加这个self.scale缩放，embedding 幅值太大，位置编码被淹没，位置信息失效 → 模型分不清语序，翻译错乱
        # decoder 里面 q、k 要使用 RoPE， 这里就不能再使用SinusoidalPositionalEncoding了
        #y = self.tgt_embedding(tgt) * self.scale 
        y = tf_multiply(a=tf_fn(x=tgt, fn=self.tgt_embedding), b=self.scale)
        new_cache = []
        for i, layer in enumerate(self.decoder_layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            
            if layer_cache is not None:
                offset = layer_cache[0].shape[2] 
            else:
                offset = 0             
            y, updated_layer_cache = layer(y, memory, tgt_mask, memory_mask, kv_cache=layer_cache, offset=offset)
            new_cache.append(updated_layer_cache)
        #logits = self.proj(y)
        logits = tf_fn(x=y, fn=self.proj)
        return logits, new_cache

    def __call__(self, src, tgt, src_mask, tgt_mask, memory_mask): 
        memory = self.encode(src, src_mask)
        logits, _ = self.decode_step(tgt, memory, tgt_mask, memory_mask)
        return logits

    def init_kv_cache(self, batch_size):
        cache = []
        for _ in range(NUM_LAYER):
            k_cache = mx.zeros((batch_size, NUM_HEAD, 0, DIM_MODEL//NUM_HEAD))
            v_cache = mx.zeros((batch_size, NUM_HEAD, 0, DIM_MODEL//NUM_HEAD))
            cache.append((k_cache,v_cache))
        return cache







