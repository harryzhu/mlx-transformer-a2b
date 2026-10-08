import math
import mlx.core as mx
import mlx.nn as nn
from .config import *

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

class RoPE_LLaMA3(nn.Module):
    """MLX RoPE 实现，适配 MHA 的 Q 向量"""
    def __init__(self, base: float = 10000.0):
        super().__init__()
        self.head_dim = DIM_MODEL // NUM_HEAD
        self.base = base
        # 标准 LLaMA：对 head_dim 全部维度做旋转，rope_dim = head_dim
        self.rope_dim = DIM_MODEL // NUM_HEAD
        # 只取偶数下标，步长2, 所以要除以2
        self.freq_dim = self.rope_dim // 2

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        # x: [B, H, L, D_head]
        B, H, L, D = x.shape
        assert D == self.head_dim, f"输入最后一维{D} != head_dim {self.head_dim}"
        positions = mx.arange(offset, offset + L, dtype=mx.float32)
        # 频率向量 [freq_dim]
        freqs = mx.exp(-mx.log(self.base) * mx.arange(0, self.freq_dim) / self.freq_dim)
        freqs = positions[:, None] * freqs[None, :] # [L, freq_dim]
    
        cos = mx.cos(freqs)  # [L, freq_dim]
        sin = mx.sin(freqs)  # [L, freq_dim]
        # 关键点：repeat 把 cos/sin 扩展到 head_dim，两两重复
        cos = mx.repeat(cos, repeats=2, axis=-1)  # [L, head_dim]
        sin = mx.repeat(sin, repeats=2, axis=-1)  # [L, head_dim]

        # 扩维到 [1,1,L,head_dim]，可以和 [B,H,L,head_dim]广播
        cos = mx.expand_dims(cos, (0, 1))
        sin = mx.expand_dims(sin, (0, 1))
        # 拆分奇偶
        x1 = x[..., ::2] # [B,H,L,freq_dim]
        x2 = x[..., 1::2] # [B,H,L,freq_dim]
        rx = mx.concatenate([-x2, x1], axis=-1) # [B,H,L, head_dim]
        return x * cos + rx * sin

class EncoderLayer(nn.Module):
    """Encoder Self-Attn：正弦PE，不用RoPE"""
    def __init__(self, norm_first: bool):
        super().__init__()
        self.norm_first = norm_first
        self.attn = nn.MultiHeadAttention(DIM_MODEL, NUM_HEAD)
        self.ffn = nn.Sequential(
            nn.Linear(DIM_MODEL, DIM_MODEL * 4),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(DIM_MODEL * 4, DIM_MODEL),
            nn.Dropout(DROPOUT),
        )
        self.norm1 = nn.LayerNorm(DIM_MODEL)
        self.norm2 = nn.LayerNorm(DIM_MODEL)
        self.dropout1 = nn.Dropout(DROPOUT)

    def __call__(self, x, mask):
        # Pre-norm
        if self.norm_first:
            h = self.norm1(x)
            h = self.attn(h, h, h, mask)
            x = x + self.dropout1(h)
            h = self.norm2(x)
            x = x + self.ffn(h)
        else:
            h = self.attn(x, x, x, mask)
            x = self.norm1(x + self.dropout1(h))
            x = self.norm2(x + self.ffn(x))
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
            nn.Dropout(DROPOUT),
            nn.Linear(DIM_MODEL * 4, DIM_MODEL),
            nn.Dropout(DROPOUT),
        )
        self.norm1 = nn.RMSNorm(DIM_MODEL)
        self.norm2 = nn.RMSNorm(DIM_MODEL)
        self.norm3 = nn.RMSNorm(DIM_MODEL)     
        self.dropout1 = nn.Dropout(DROPOUT)

    
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
        q = self.rope(q)
        k = self.rope(k)

        # MHA score
        #attn_out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask)
        # scaled_dot_product_attention 只支持 additive_mask, 不支持 additive_mask + pad mask
        # 所以此处改写法, 速度比 mx.fast.scaled_dot_product_attention 慢一些
        attn_scores = q @ k.transpose(0,1,3,2) / self.var_sqrt_head_dim # [B, H, L, L]
        if DEBUG:
            bhld_check(attn_scores, B, self.num_heads, L, L)
            bhld_check(mask, B, 1, L, L)
        attn_scores = attn_scores + mask
        attn_weights = mx.softmax(attn_scores, axis=-1)
        attn_out = attn_weights @ v
        if DEBUG:
            bhld_check(attn_out, B, self.num_heads, L, self.head_dim)
        
        attn_out = attn_out.transpose(0,2,1,3).reshape(B, L, D)
        if DEBUG:
            bld_check(attn_out, B, L, DIM_MODEL)
        return self.dropout1(attn_out)

    def __call__(self, x, memory, tgt_mask, memory_mask):
        if self.norm_first:
            # Decoder Self-Attention with RoPE
            h = self.norm1(x)
            h = self.self_attn_rope(h, tgt_mask)
            x = x + h

            # Cross Attention
            h = self.norm2(x)
            h = self.cross_attn(h, memory, memory, memory_mask)
            x = x + self.dropout1(h)

            # FFN
            h = self.norm3(x)
            x = x + self.ffn(h)
        else:
            h = self.self_attn_rope(x, tgt_mask)
            x = self.norm1(x + h)
            h = self.cross_attn(x, memory, memory, memory_mask)
            x = self.norm2(x + self.dropout1(h))
            x = self.norm3(x + self.ffn(x))
        return x

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
        x = self.src_embedding(src) * self.scale
        x = x + self.src_pos_enc(mx.arange(src.shape[-1]))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode_step(self, tgt, memory, tgt_mask, memory_mask): 
        #不加这个self.scale缩放，embedding 幅值太大，位置编码被淹没，位置信息失效 → 模型分不清语序，翻译错乱
        # decoder 里面 q、k 要使用 RoPE， 这里就不能再使用SinusoidalPositionalEncoding了
        y = self.tgt_embedding(tgt) * self.scale 
        for layer in self.decoder_layers:
            y = layer(y, memory, tgt_mask, memory_mask)
        logits = self.proj(y)
        return logits

    def __call__(self, src, tgt, src_mask, tgt_mask, memory_mask): 
        memory = self.encode(src, src_mask)
        logits = self.decode_step(tgt, memory, tgt_mask, memory_mask)
        return logits

