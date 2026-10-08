import numpy as np
from .config import *

def seq_padding(X, padding=PAD):
    L = [len(x) for x in X]
    ML = max(L)
    # print("2.获取最长那句话的长度，比这句短的就要 PAD：", ML)
    padded = np.array([
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])
    
    return padded

