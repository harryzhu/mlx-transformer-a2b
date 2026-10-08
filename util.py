import os
import csv
import re
import random
import numpy as np
import mlx.core as mx
import mlx.nn as nn

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
