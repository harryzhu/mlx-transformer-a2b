import os
import re
import sys
import csv

# max_lines 控制从全量数据集里面挑选出来的行数，依据设备的显存和期望的时间来确定
# 这些挑选出来的数据会写入 data/data_train.csv ，后续被 train 函数读取
max_lines = 8000

def make_data_train(in_path=None, out_path=f'data/data_train.csv'):
    with open(out_path, "w")as fwcsv:
        csvWriter = csv.writer(fwcsv)
        csvWriter.writerow(["src","tgt"])
        saverows = []
        with open("cache/all_src.txt", "w")as fwen:
            with open("cache/all_tgt.txt", "w")as fwzh:
                with open(in_path, mode="r", encoding="utf-8") as f:
                    zh = []
                    en = []
                    i = 0
                    csv_reader = csv.DictReader(f)
                    for row in csv_reader:
                        sent_zh = row['tgt']
                        sent_en = row['src']
                        if len(sent_en) < 2:
                            continue
                        if len(sent_zh) < 2:
                            continue
                        zh.append(sent_zh)
                        en.append(sent_en)
                        saverows.append([sent_en, sent_zh])
                        if i >= max_lines-1:
                            break
                        i += 1
                    
                    fwen.write("\n".join(en))
                    fwzh.write("\n".join(zh))
                    csvWriter.writerows(saverows)
                  

#
if __name__ == "__main__":
    data_full = None
    if len(sys.argv) == 1:
        data_full = "data/data_wmt.csv"
        print("pls add parameter: the path of dataset")

    if len(sys.argv) == 2:
        data_full = sys.argv[1]
        print(f"data source path: {data_full}")

    if os.path.exists(data_full):
        print(f"processing: {data_full}")
        make_data_train(in_path=data_full)
    else:
        print(f"file does not exist: {data_full}")

