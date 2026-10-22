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

#model = TransformerModelSin()
model = TransformerModelRoPE()

#model_path = "test_sp_sinpe.safetensors"
model_path = "test_sp_rope.safetensors"

if os.path.exists(model_path):
    model.load_weights(model_path)

