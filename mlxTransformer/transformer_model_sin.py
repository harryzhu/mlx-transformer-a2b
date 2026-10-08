import math
import mlx.core as mx
import mlx.nn as nn
from .config import *
#from mlxTFdataset.mask import *


class TransformerModelSin(nn.Module):
    def __init__(self):
        super(TransformerModelSin, self).__init__()
        self.src_embedding = nn.Embedding(VOCAB_SIZE_SRC, DIM_MODEL)
        self.tgt_embedding = nn.Embedding(VOCAB_SIZE_TGT, DIM_MODEL)
        self.src_pos_enc = nn.SinusoidalPositionalEncoding(DIM_MODEL)
        self.tgt_pos_enc = nn.SinusoidalPositionalEncoding(DIM_MODEL)
        # rope 的 dims 必须为偶数（RoPE 需要两两一组做平面旋转）
        if DEBUG:
            assert DIM_MODEL%NUM_HEAD == 0
            assert (DIM_MODEL//NUM_HEAD)%2 == 0
        self.proj = nn.Linear(DIM_MODEL, VOCAB_SIZE_TGT, bias=False)
        self.var_scale = math.sqrt(DIM_MODEL)
        self.encoder_layers = [
            nn.TransformerEncoderLayer(dims=DIM_MODEL, num_heads=NUM_HEAD, dropout=DROPOUT, norm_first=True)
            for _ in range(NUM_LAYER)
            ]
        self.decoder_layers = [
            nn.TransformerDecoderLayer(dims=DIM_MODEL, num_heads=NUM_HEAD,dropout=DROPOUT, norm_first=True)
            for _ in range(NUM_LAYER)
            ]
    
    def encode(self, src, src_mask):
        #不加这个self.scale缩放，embedding 幅值太大，位置编码被淹没，位置信息失效 → 模型分不清语序，翻译错乱
        x = self.src_embedding(src) * self.var_scale
        x = x + self.src_pos_enc(mx.arange(src.shape[-1]))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode_step(self, tgt, memory, tgt_mask, memory_mask): 
        #不加这个self.scale缩放，embedding 幅值太大，位置编码被淹没，位置信息失效 → 模型分不清语序，翻译错乱
        y = self.tgt_embedding(tgt) * self.var_scale
        pe = self.tgt_pos_enc(mx.arange(tgt.shape[-1]))[None, :, :] # [1, S, D]
        y = y + pe  
        for layer in self.decoder_layers:
            y = layer(y, memory, tgt_mask, memory_mask)
        logits = self.proj(y)
        return logits

    def __call__(self, src, tgt, src_mask, tgt_mask, memory_mask): 
        memory = self.encode(src, src_mask)
        logits = self.decode_step(tgt, memory, tgt_mask, memory_mask)
        return logits

