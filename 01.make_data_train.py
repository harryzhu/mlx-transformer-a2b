import os
import re
import csv

en_zh_file = "/Users/harry/dev/ai/t03/data/modelscope/iic/WMT-Chinese-to-English-Machine-Translation-Training-Corpus/wmt_zh_en_training_corpus.csv"
poetry_file = "/Users/harry/dev/ai/t04/data/poetry.csv"
# max_lines 控制从全量数据集里面挑选出来的行数，依据设备的显存和期望的时间来确定
# 这些挑选出来的数据会写入 data/data_train.csv ，后续被 train 函数读取
max_lines = 8000

pattern_chinese = re.compile(r'[\u4e00-\u9fff]')
def has_chinese(words):
    for w in words:
        if pattern_chinese.search(w) is not None:
            return True
    return False 

def has_word(sentence, letters=[]):
    for l in letters:
        if sentence.count(l) > 0:
            return True
    return False

def is_startswith(sentence, letters):
    for l in letters:
        if sentence.startswith(l):
            return True
    return False
                  

def wmt_en_zh_dataset(in_path=en_zh_file, out_path=f'data/data_train.csv'):         
    firstLetter = []
    with open(out_path, "w")as fwcsv:
        csvWriter = csv.writer(fwcsv)
        with open("cache/all_src.txt", "w")as fwen:
            with open("cache/all_tgt.txt", "w")as fwzh:
                with open(in_path, mode="r", encoding="utf-8") as f:
                            zh = []
                            en = []
                            duplicatelines = {}
                            saverows = []
                            saverows.append(('src', 'tgt'))
                            i = 0
                            csv_reader = csv.DictReader(f)
                            for row in csv_reader:
                                #sent_zh = row['0'].strip()
                                sent_zh = row['0'].strip()
                                sent_en = row['1'].strip()

                                sent_zh = sent_zh.replace(" ","")
                                if sent_zh == "" or sent_en == "":
                                    continue
                                # if sent_en.count(" ") < 40:
                                #     continue
                                if not sent_en[0].isalpha():
                                    continue
                                if has_word(sent_en, letters=["&","@","==","**"]):
                                    continue                          
                                if has_word(sent_zh, letters=["\ue009","\ue004","\ue5e5","\ue5e5","\ue009"]):
                                    continue

                                if is_startswith(sent_zh,["。","」","、","，","；","–","…","“”","–","——","＃","Idon","：","Youcan","QUOTE","VI","1","2","3","4","5","6","7","8","9","0","#","９","７","６","５","４","３","２","１","①","’’"]):
                                    continue
                                         
                                if has_chinese(sent_en):
                                    continue

                                dup_key = sent_en.replace(" ","")
                                try:
                                    if duplicatelines[dup_key] == 1:
                                        continue
                                except:
                                    pass
                                if not has_chinese(sent_zh[0]):
                                    firstLetter.append(sent_zh[0])
                                firstLetter = list(set(firstLetter))
                                for fl in firstLetter:
                                    if sent_zh.startswith(fl):
                                        continue
                                zh.append(sent_zh)
                                en.append(sent_en)

                                duplicatelines[dup_key] = 1
                                
                                saverows.append((sent_en, sent_zh))
                                if i % 500 == 0:
                                    fwen.write("\n".join(en))
                                    fwzh.write("\n".join(zh))
                                    csvWriter.writerows(saverows)
                                    en = []
                                    zh = []
                                    saverows = []
                                
                                if i > max_lines:
                                    break
                                i += 1
                fwen.write("\n".join(en))
                fwzh.write("\n".join(zh))
                csvWriter.writerows(saverows)

    print(list(set(firstLetter)))


def poetry_zh_zh_dataset(in_path=poetry_file, out_path=f'data/data_train.csv'):
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
                        if i >= max_lines:
                            break
                        i += 1
                    
                    fwen.write("\n".join(en))
                    fwzh.write("\n".join(zh))
                    csvWriter.writerows(saverows)
                        




#wmt_en_zh_dataset()
poetry_zh_zh_dataset()