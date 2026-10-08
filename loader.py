import numpy as np
import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm
from mlxTFdataset import *
from mlxTransformer import *

# 第 src 列为源文， 第 tgt 列为目标文
train_file = "data/data_train.csv"

sp_model_src = "cache/vocab_src.model"
sp_model_tgt = "cache/vocab_tgt.model"

data = SrcTgtData(fpath=train_file,
                  sp_model_src=sp_model_src, 
                  sp_model_tgt=sp_model_tgt, 
                  batch_size=BATCH_SIZE)

spm_src = spm.SentencePieceProcessor(model_file=sp_model_src)
spm_tgt = spm.SentencePieceProcessor(model_file=sp_model_tgt)

#model_path = "test_sp_sinpe.safetensors"
model_path = "test_sp_rope.safetensors"

#model = TransformerModelSin()
model = TransformerModelRoPE()

