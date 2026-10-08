DEBUG = True

NUM_LAYER = 6                     # transformer中encoder、decoder层数
NUM_HEAD = 4                           # 多头注意力个数
DIM_MODEL = 128                       # 输入、输出词向量维数
DROPOUT = 0.1                       # dropout比例


PAD = 0                             # padding占位符的索引
UNK = 1                             # 未登录词标识符的索引

BATCH_SIZE = 20

VOCAB_SIZE_SRC = 10000
VOCAB_SIZE_TGT = 10000