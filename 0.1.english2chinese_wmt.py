import os
import re
import csv
from util import *

#en_zh_file = "/Volumes/HDD4/_models/modelscope/iic/WMT-Chinese-to-English-Machine-Translation-Training-Corpus/wmt_zh_en_training_corpus.csv"
en_zh_file = "data/data_news_commentary.csv"

ptn_eng = re.compile(r'[a-zA-Z0-9]')
                  

def wmt_en_zh_dataset(in_path=en_zh_file, out_path=f'data/data_news.csv'):         
    firstLetter = []
    with open(out_path, "w")as fwcsv:
        csvWriter = csv.writer(fwcsv, quoting=csv.QUOTE_ALL)
        with open(in_path, mode="r", encoding="utf-8") as f:
            duplicatelines = {}
            saverows = []
            saverows.append(('src', 'tgt'))
            i = 0
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                # sent_zh = row['0'].strip()
                # sent_en = row['1'].strip()
                sent_zh = row['tgt'].strip()
                sent_en = row['src'].strip()

                has_eng = re.findall(r'[a-zA-Z0-9]',sent_en)
                if len(has_eng) < 5:
                    print(sent_en)
                    continue

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
                            
                sent_en = clean_cht_line(sent_en)
                sent_zh = clean_cht_line(sent_zh)

                if has_chinese(sent_en):
                    continue
                if has_alpha(sent_zh):
                    continue
                if has_digit(sent_zh):
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
                
                duplicatelines[dup_key] = 1 
                saverows.append((sent_en, sent_zh))
                if i % 10000 == 0:
                    csvWriter.writerows(saverows)
                    saverows = []
                    print(i)
                # comment the following 2 lines as you need    
                if i > 200000:
                    break

                i += 1
                
        csvWriter.writerows(saverows)

    print(list(set(firstLetter)))
        

wmt_en_zh_dataset()



