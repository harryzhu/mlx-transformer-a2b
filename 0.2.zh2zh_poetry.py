import os
import re
import csv
from util import *

poetry_file = "/Users/harry/dev/ai/t04/data/poetry.txt"
                
def poetry_zh_zh_dataset(in_path=poetry_file, out_path=f'data/data_poetry.csv'):
    with open(out_path, "w")as fwcsv:
        csvWriter = csv.writer(fwcsv, quoting=csv.QUOTE_ALL)
        csvWriter.writerow(["src","tgt"])
        saverows = []
        with open(in_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                p_dot = line.find("。")
                sent_zh = line[p_dot+1:]
                sent_en = line[:p_dot+1]
                sent_zh = clean_cht_line(sent_zh)
                sent_en = clean_cht_line(sent_en)
                
                saverows.append([sent_en, sent_zh])
                if i % 10000 == 0:
                    csvWriter.writerows(saverows)
                    saverows = []
                if i >= 200000:
                    break
                i += 1
              
            csvWriter.writerows(saverows)
                


poetry_zh_zh_dataset()




