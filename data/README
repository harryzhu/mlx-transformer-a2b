1) 训练数据 `data_train.csv` 放于 `data/` 文件夹。

格式必须是 `.csv` ， `第一行`必须是`表头`， 第一列为 `src`， 第二列为 `tgt`。

例如：

```csv
src,tgt
hello,你好
world,世界
```

2) 在上一级目录运行 `python 01.make_data_train.py` ,即会在 `cache` 文件夹下生成：

 - all_src.txt
 - all_tgt.txt


3) 修改 `mlxTransformer/config.py` 里面 `VOCAB_SIZE_SRC` 和 `VOCAB_SIZE_TGT` 的值，可以先修改为一个极大数，从`第 4）步`的 `报错信息` 中看到最准确的词数量，然后将 `VOCAB_SIZE_SRC` 和 `VOCAB_SIZE_TGT` 修改为`尽可能接近于该精确值`，越接近精确值，训练效果越好，但训练速度、显存占用也越受影响。根据自己的显存，搭配 `BATCH_SIZE` 来微调这两个值.


4) 在上一级目录运行 `python 02.preprocess.py` ,即会在 `cache` 文件夹下生成：

 - vocab_src.model
 - vocab_src.vocab
 - vocab_tgt.model
 - vocab_tgt.vocab

5）运行 `python train.py 200` 即可开始训练, 200 表示训练次数，如果次数小于 10，DEBUG 会自动设为 True。

6）按需修改 `infer.py` 里面的例句， 运行 `python infer.py` 即可开始生成。





