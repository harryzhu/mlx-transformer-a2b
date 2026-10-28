import numpy as np
import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm
from mlxTFdataset import *
from mlxTransformer import *

sp_model_src = "cache/vocab_src.model"
sp_model_tgt = "cache/vocab_tgt.model"
sp_model_gpt = "cache/vocab_gpt.model"

spm_src = spm.SentencePieceProcessor(model_file=sp_model_src)
spm_tgt = spm.SentencePieceProcessor(model_file=sp_model_tgt)
#
spm_gpt = spm.SentencePieceProcessor(model_file=sp_model_gpt)


assert B_O_S == spm_src.bos_id()
assert E_O_S == spm_src.eos_id() 
assert UNK == spm_src.unk_id() 
assert PAD == spm_src.pad_id()

assert B_O_S == spm_tgt.bos_id()
assert E_O_S == spm_tgt.eos_id() 
assert UNK == spm_tgt.unk_id() 
assert PAD == spm_tgt.pad_id()

print(f'PAD: {PAD}')
print(f'UNK: {UNK}')
print(f'<|en|>: {spm_gpt.encode("<|en|>", return_type=int)}')
print(f'<|zh|>: {spm_gpt.encode("<|zh|>", return_type=int)}')
print(f'<|end|>: {spm_gpt.encode("<|end|>", return_type=int)}')

if ENABLE_GPT:
    print("Current: GPTModel")
    model = GPTModel()
    model_path = "gpt.safetensors"
else:
    # print("Current: TransformerModelSin")
    #model = TransformerModelSin()
    #model_path = "test_sp_sinpe.safetensors"
    # rope
    print("Current: TransformerModelRoPE")
    model = TransformerModelRoPE()
    model_path = "test_sp_rope.safetensors"

if os.path.exists(model_path):
    print(f"{model_path} is loading ...")
    model.load_weights(model_path)



