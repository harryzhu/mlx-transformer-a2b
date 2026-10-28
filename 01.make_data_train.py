import os
import re
import sys
import csv
import util

# max_lines 控制从全量数据集里面挑选出来的行数，依据设备的显存和期望的时间来确定
# 这些挑选出来的数据会写入 data/data_train.csv ，后续被 train 函数读取
max_lines = 8000

# r'\u00a0',r'\u00ad',r'\u093E'
ban_words = [r'\u0153',r'\uFFFD',r'([\u00A0-\u00FF]|[\u0100-\u017F]|[\u0180-\u024F])']
def make_data_train(in_path=None, out_path=f'data/data_train.csv'):
    uq_rows = {}
    with open(in_path, mode="r", encoding="utf-8") as f:
        zh = []
        en = []
        gpt = []
        saverows = []
        i = 0
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            sent_zh = row['tgt'].strip()
            sent_en = row['src'].strip()
            if sent_en.count(" ") > 40:
                continue
            if len(sent_zh) > 64:
                continue
            if util.ratio_alpha(sent_en) < 0.3:
                continue
            is_ignore = False
            for bw in ban_words:
                bm_en = re.findall(bw, sent_en)
                if len(bm_en) != 0:
                    print(bm_en, ":::", sent_en)
                    is_ignore = True
                    break

                bm_zh = re.findall(bw, sent_zh)
                if len(bm_zh) != 0:
                    print(bm_zh, ":::", sent_zh)
                    is_ignore = True
                    break

            if is_ignore:
                continue
            try:
                if uq_rows[sent_en] == 1:
                    continue
            except:
                uq_rows[sent_en] = 1

            zh.append(sent_zh)
            en.append(sent_en)
            gpt.append(f'{sent_en} {sent_zh}')
            saverows.append((sent_en, sent_zh))
            if i >= max_lines-1:
                break
            i += 1

    with open("cache/all_tgt.txt", "w")as fwzh:
        fwzh.write("\n".join(zh))

    with open("cache/all_src.txt", "w")as fwen:
        fwen.write("\n".join(en))

    with open("cache/all_gpt.txt", "w")as fwgpt:
            fwgpt.write("\n".join(gpt))
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

