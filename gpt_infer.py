import os
import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm
from mlxTFdataset import *
from mlxTransformer import *
from loader import *

def gpt_infer(sentence):
    src_tokens = spm_gpt.encode(sentence, return_type=int)
    src_tokens.insert(0,ID_EN)
    src_tokens.append(ID_ZH)
    print(f'src_tokens: {src_tokens} , length: {len(src_tokens)}')
    src_tokens = mx.array(src_tokens, dtype=mx.int32)
    src_tokens = src_tokens[None,:]
    print("src_tokens: ", src_tokens, src_tokens.shape)
    #gen_ids = generate(model, src=src_tokens, use_greedy = False)
    gen_ids = gpt_generate(model, src=src_tokens, use_greedy = False)
    gen_ids = gen_ids.tolist()
    
    #print("-----")
    for gid in gen_ids:
        try:
            print(sentence)
            trans_zh = spm_gpt.decode(gid)
            print(trans_zh)
        except Exception as err:
            print(err)
            print(", ERROR: ", gid, end="")
    print("\n")


examples = []
#examples.append("do not be afraid to haggle on price - especially if you are not booking during the school holidays . ")

#examples.append("this is where the entertainer holds an envelope to his head, gives the answer, and then opens the envelope and reads the question.")
examples.append("Families of the Future") #in Euro area
examples.append("In fact, there are two lefts in the region")
#examples.append("The scientific activities of Swedish research groups")
#examples.append("细雨湿凄凄,")

for i in range(len(examples)):
    gpt_infer(examples[i])

