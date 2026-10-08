import os
import csv
import json
import pickle
import numpy as np
import sentencepiece as spm
#from .padding import *
from .batch import *

class SrcTgtData():
    def __init__(self,fpath, sp_model_src, sp_model_tgt, batch_size=3):
        self.sp_model_src = spm.SentencePieceProcessor(model_file=sp_model_src)
        self.sp_model_tgt = spm.SentencePieceProcessor(model_file=sp_model_tgt)
        self.src, self.tgt = self.load_file(fpath)
        self.batches = self.split_batch(self.src,self.tgt, batch_size=batch_size, shuffle=True)
        
    def dump_cache(self, fname, data):
        with open(fname,"wb")as fw:
            pickle.dump(data, fw)

    def load_cache(self, fname):
        with open(fname,"rb")as fr:
            return pickle.load(fr)

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
        tgt = []
        num_line = 0
        with open(fpath, mode="r", encoding="utf-8") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                sent_tgt = row['tgt'].strip()
                sent_src = row['src'].strip()
                if sent_tgt == "" or sent_src == "":
                    continue
            
                src_ids = self.sp_model_src.encode(sent_src, out_type=int)

                sent_tgt = sent_tgt.replace(" ", "")
                tgt_ids = self.sp_model_tgt.encode(sent_tgt, out_type=int)

                src.append(src_ids)
                tgt.append(tgt_ids)
             
                num_line += 1
        
        return src, tgt

    
    def split_batch(self,src,tgt,batch_size, shuffle=True):
        idx_list = np.arange(0, len(src), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)
            print("shuffle: batches: ", len(idx_list))
        
        batch_idxes = []
        for idx in idx_list:
            batch_idxes.append(np.arange(idx, min(idx+batch_size, len(src))))
        
        batches = []
        for bidx in batch_idxes:
            batch_en = [src[i] for i in bidx]
            batch_zh = [tgt[i] for i in bidx]
            #
            batches.append(Batch(batch_en,batch_zh))
        return batches
