import math
import mlx.core as mx
import mlx.nn as nn
import copy
from config import *

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
