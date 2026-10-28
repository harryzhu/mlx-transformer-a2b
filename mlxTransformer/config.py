DEBUG = True
ENABLE_EXT = False

NUM_LAYER = 4                         # transformer中encoder、decoder层数
NUM_HEAD = 4                          # 多头注意力个数
DIM_MODEL = 128                       # 输入、输出词向量维数
DROPOUT = 0.15                        # dropout比例

BATCH_SIZE = 20
LEARNING_RATE = 0.0005

#PAD = 0                             # padding占位符的索引
#UNK = 1                             # 未登录词标识符的索引

VOCAB_SIZE_SRC = 8000
VOCAB_SIZE_TGT = 10000

# gpt model only
ENABLE_GPT = False
VOCAB_SIZE_GPT = 30000
