import os
import re
import sys
import csv

# max_lines 控制从全量数据集里面挑选出来的行数，依据设备的显存和期望的时间来确定
# 这些挑选出来的数据会写入 data/data_train.csv ，后续被 train 函数读取
max_lines = 10000

def make_data_train(in_path=None, out_path=f'data/data_train.csv'):
    uq_rows = {}
    with open(in_path, mode="r", encoding="utf-8") as f:
        zh = []
        en = []
        saverows = []
        i = 0
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            sent_zh = row['tgt'].strip()
            sent_en = row['src'].strip()
            if sent_en.count(" ") < 8:
                continue
            if len(sent_zh) < 8:
                continue
            if sent_en in uq_rows.keys():
                continue
            uq_rows[sent_en] = 1

            zh.append(sent_zh)
            en.append(sent_en)
            saverows.append((sent_en, sent_zh))
            if i >= max_lines-1:
                break
            i += 1

    with open("cache/all_tgt.txt", "w")as fwzh:
        fwzh.write("\n".join(zh))

    with open("cache/all_src.txt", "w")as fwen:
        fwen.write("\n".join(en))
    #按照英文句子长短排序，避免无效padding过多
    saverows.sort(key=lambda ele: len(ele[0]), reverse=False)
    with open(out_path, "w")as fwcsv:
        csvWriter = csv.writer(fwcsv, quoting=csv.QUOTE_ALL)
        csvWriter.writerow(["src","tgt"])
        csvWriter.writerows(saverows)
                          
                  
#
if __name__ == "__main__":
    data_full = None
    if len(sys.argv) == 1:
        data_full = "data/data_news.csv"
        print("pls add parameter: the path of dataset")

    if len(sys.argv) == 2:
        data_full = sys.argv[1]
        print(f"data source path: {data_full}")

    if os.path.exists(data_full):
        print(f"processing: {data_full}")
        make_data_train(in_path=data_full)
    else:
        print(f"file does not exist: {data_full}")

