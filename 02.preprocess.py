import os
import sentencepiece as spm
from mlxTransformer.config import *
from collections import Counter

spm.SentencePieceTrainer.train(
    input='cache/all_src.txt', 
    model_prefix='cache/vocab_src', 
    vocab_size=VOCAB_SIZE_SRC
)

spm.SentencePieceTrainer.train(
    input='cache/all_tgt.txt', 
    model_prefix='cache/vocab_tgt', 
    vocab_size=VOCAB_SIZE_TGT
)


def count_freq(fpath,out_path):
    if os.path.exists(fpath):
        en_words = []
        with open(fpath,"r") as f:
            lines = f.read().split("\n")
            for line in lines:
                words = line.split(" ")
                en_words += words
            c_w = Counter(en_words)
            mc = c_w.most_common(VOCAB_SIZE_SRC)
            with open(out_path,"w") as f: 
                lines = [f'{wc[0]:20s}: {wc[1]}' for wc in mc]  
                f.write("\n".join(lines))

count_freq("cache/all_src.txt","cache/freq_src.txt")





