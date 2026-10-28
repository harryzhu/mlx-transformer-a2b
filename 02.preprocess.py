import os
import sentencepiece as spm
from mlxTransformer.config import *
from collections import Counter
from util import *

if os.path.exists('cache/all_src.txt'):
    spm.SentencePieceTrainer.train(
        input='cache/all_src.txt', 
        model_prefix='cache/vocab_src', 
        vocab_size=VOCAB_SIZE_SRC,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="PAD",
        unk_piece="UNK",
        bos_piece="B_O_S",
        eos_piece="E_O_S",
        normalization_rule_name="identity",
        add_dummy_prefix=False,
    )

if os.path.exists('cache/all_tgt.txt'):
    spm.SentencePieceTrainer.train(
        input='cache/all_tgt.txt', 
        model_prefix='cache/vocab_tgt', 
        vocab_size=VOCAB_SIZE_TGT,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="PAD",
        unk_piece="UNK",
        bos_piece="B_O_S",
        eos_piece="E_O_S",
        normalization_rule_name="identity",
        add_dummy_prefix=False,
    )

if os.path.exists('cache/all_gpt.txt'):
    spm.SentencePieceTrainer.train(
        input='cache/all_gpt.txt', 
        model_prefix='cache/vocab_gpt', 
        vocab_size=VOCAB_SIZE_GPT,
        user_defined_symbols=['<|en|>','<|zh|>','<|end|>'],
        pad_id=0,
        unk_id=1,
        pad_piece="PAD",
        unk_piece="UNK",
        bos_id=-1,
        eos_id=-1,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
    )

def count_freq(fpath,out_path):
    if os.path.exists(fpath):
        en_words = []
        with open(fpath,"r") as f:
            lines = f.read().split("\n")
            for line in lines:
                words = [char for char in line]
                en_words += words
                # if not has_chinese(line):
                #     words = [char for char in line.split(" ")]
                #     en_words += words
            uq_en_words = list(set(en_words))
            c_w = Counter(en_words)
            #mc = c_w.most_common(VOCAB_SIZE_SRC)
            mc = c_w.most_common(len(uq_en_words))
            with open(out_path,"w") as f: 
                lines = [f'{wc[0]:20s}: {wc[1]}' for wc in mc]  
                f.write("\n".join(lines))

count_freq("cache/all_src.txt","cache/freq_src.txt")
count_freq("cache/all_tgt.txt","cache/freq_tgt.txt")
count_freq("cache/all_gpt.txt","cache/freq_gpt.txt")






