import math
import mlx.core as mx
import mlx.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, q, kv, mask = None):
        B, Lq, _ = q.shape
        B, Lk, _ = kv.shape

        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        # 分头
        q = q.reshape(B, Lq, self.num_heads, self.head_dim).transpose(0,2,1,3)
        k = k.reshape(B, Lk, self.num_heads, self.head_dim).transpose(0,2,1,3)
        v = v.reshape(B, Lk, self.num_heads, self.head_dim).transpose(0,2,1,3)

        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        attn = attn.transpose(0,2,1,3).reshape(B, Lq, self.dim)
        return self.out_proj(attn)
