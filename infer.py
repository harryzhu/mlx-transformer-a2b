import os
import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm
from mlxTFdataset import *
from mlxTransformer import *
from loader import *



def infer(sentence):
    src_tokens = spm_src.encode(sentence, return_type=int)
    src_tokens = mx.array(src_tokens, dtype=mx.int32)
    src_tokens = src_tokens[None,:]
    print("src_tokens: ", src_tokens, src_tokens.shape)
    gen_ids = generate(model, src=src_tokens)
    gen_ids = gen_ids.tolist()
    
    #print("-----")
    for gid in gen_ids:
        try:
            print(sentence)
            trans_zh = spm_tgt.decode(gid)
            print(trans_zh)
        except Exception as err:
            print(err)
            print(", ERROR: ", gid, end="")
    print("\n")


if os.path.exists(model_path):
    print("model is loading ...")
    model.load_weights(model_path)



examples = []
examples.append("do not imagine that mathematics is hard and crabbed and repulsive to common sense , it is merely the etherealization of common sense .")

examples.append("未得君书")

for i in range(len(examples)):
    infer(examples[i])

