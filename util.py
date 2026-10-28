import os
import csv
import re
import zhconv
import random
import numpy as np
import mlx.core as mx
import mlx.nn as nn

RE_CHINESE_CHAR = re.compile(r"[\u4e00-\u9fff]")
RE_ALPHA_CHAR = re.compile(r"[A-Za-z]")
RE_DIGIT_CHAR = re.compile(r"[0-9]")
RE_WHITESPACE = re.compile(r"\s+")

def cht_to_chs(traditional_text):
    simplified_text = zhconv.convert(traditional_text, 'zh-cn')
    return simplified_text

def full2half(s: str) -> str:
    out = []
    for ch in s:
        u = ord(ch)
        if u == 12288:          # 全角空格
            u = 32
        elif 65281 <= u <= 65374: # 全角符号范围
            u -= 65248
        out.append(chr(u))
    return "".join(out)

def clean_cht_line(text: str) -> str:
    if text is None or len(text.strip()) == 0:
        return text
    # 繁转简
    simp_text = cht_to_chs(text)
    # 全角转半角
    final_text = full2half(simp_text)
    return final_text


def is_all_chinese(words):
    for w in words:
        if RE_CHINESE_CHAR.search(w) is None:
            return False
    return True

def has_chinese(words):
    for w in words:
        if RE_CHINESE_CHAR.search(w) is not None:
            return True
    return False 

def has_alpha(words):
    for w in words:
        if RE_ALPHA_CHAR.search(w) is not None:
            return True
    return False

def ratio_alpha(words):
    nos = words.replace(" ","")
    len_nos = len(nos)
    a = RE_ALPHA_CHAR.findall(nos)
    r = float(len(a)) / float(len_nos)
    return r

def has_digit(words):
    for w in words:
        if RE_DIGIT_CHAR.search(w) is not None:
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
