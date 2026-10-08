import numpy as np
import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm
from mlxTFdataset import *
from mlxTransformer import *

sp_model_src = "cache/vocab_src.model"
sp_model_tgt = "cache/vocab_tgt.model"

spm_src = spm.SentencePieceProcessor(model_file=sp_model_src)
spm_tgt = spm.SentencePieceProcessor(model_file=sp_model_tgt)

#model_path = "test_sp_sinpe.safetensors"
model_path = "test_sp_rope.safetensors"

#model = TransformerModelSin()
model = TransformerModelRoPE()

