import os
import csv
import json
import pickle
import numpy as np
import sentencepiece as spm
from .batch import *
from .config import *

class GptData():
    def __init__(self,fpath, sp_model_file, batch_size=3):
        self.spm = spm.SentencePieceProcessor(model_file=sp_model_file)
        self.src = self.load_file(fpath)
        self.batches = self.split_batch(self.src, batch_size=batch_size, shuffle=True)
        
    def load_file(self,fpath):
        fext = os.path.basename(fpath).split(".")[-1]
        if fext.lower() == "csv":
            return self.load_csv(fpath)
        if fext.lower() == "jsonl":
            return self.load_jsonl(fpath)

    def load_jsonl(self,fpath):
        pass


    def load_csv(self,fpath):
        src = []
        num_line = 0
        with open(fpath, mode="r", encoding="utf-8") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                sent_tgt = row['tgt'].strip()
                sent_src = row['src'].strip()
                if sent_tgt == "" or sent_src == "":
                    continue
            
                src_ids = self.spm.encode(sent_src, out_type=int)
                tgt_ids = self.spm.encode(sent_tgt, out_type=int)
                # print(src_ids)
                # print(tgt_ids)
                src_ids.insert(0, ID_EN)
                src_ids.append(ID_ZH)
                tgt_ids.append(ID_END)
                sent = src_ids + tgt_ids
                # print(sent)
                # print("---")
                src.append(sent)
                
                num_line += 1
        
        return src

    
    def split_batch(self,src,batch_size, shuffle=True):
        idx_list = np.arange(0, len(src), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)

        print("batches: ", len(idx_list))
        batch_idxes = []
        for idx in idx_list:
            batch_idxes.append(np.arange(idx, min(idx+batch_size, len(src))))
        
        batches = []
        for bidx in batch_idxes:
            batch_en = [src[i] for i in bidx]
            #
            batches.append(GptBatch(batch_en))
        return batches
