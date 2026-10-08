import os
import re
import csv
from util import *

poetry_file = "/Users/harry/dev/ai/t04/data/poetry.csv"
# max_lines 控制从全量数据集里面挑选出来的行数，依据设备的显存和期望的时间来确定
# 这些挑选出来的数据会写入 data/data_train.csv ，后续被 train 函数读取
max_lines = 8000

                
def poetry_zh_zh_dataset(in_path=poetry_file, out_path=f'data/data_poetry.csv'):
    with open(out_path, "w")as fwcsv:
        csvWriter = csv.writer(fwcsv)
        csvWriter.writerow(["src","tgt"])
        saverows = []
        with open(in_path, mode="r", encoding="utf-8") as f:
            csv_reader = csv.DictReader(f)
            i = 0
            for row in csv_reader:
                sent_zh = row['tgt']
                sent_en = row['src']
                if len(sent_en) < 2:
                    continue
                if len(sent_zh) < 2:
                    continue
                saverows.append([sent_en, sent_zh])
                if i % 10000 == 0:
                    csvWriter.writerows(saverows)
                    saverows = []
              
            csvWriter.writerows(saverows)
                


poetry_zh_zh_dataset()




