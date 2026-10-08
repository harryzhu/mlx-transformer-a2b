import sys
import os
import mlx.core as mx
import mlx.nn as nn
from mlxTFdataset import *
from mlxTransformer import *
from loader import *

train_epochs = 10

# 第 src 列为源文， 第 tgt 列为目标文
train_file = "data/data_train.csv"

data = SrcTgtData(fpath=train_file,
                  sp_model_src=sp_model_src, 
                  sp_model_tgt=sp_model_tgt, 
                  batch_size=BATCH_SIZE)


def loss_fn(model, src, tgt_in, tgt_label, src_mask, tgt_mask):
    logits = model(src, tgt_in, src_mask, tgt_mask, src_mask)
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    labels_flat = tgt_label.reshape(-1)
    ce = nn.losses.cross_entropy(logits_flat, labels_flat, label_smoothing=0.2, reduction="none")
    none_pad_mask = (labels_flat != PAD)
    ce = ce * none_pad_mask
    loss = mx.sum(ce) / mx.sum(none_pad_mask)
    return loss

loss_and_grad = nn.value_and_grad(model, loss_fn)       

def train():
    global DEBUG
    if train_epochs > 10:
        DEBUG = False
    print("In Debug Mode: ", DEBUG)
    train_data = data.batches
    # 原生 Adam 固定学习率很难训 Transformer。一般用 warmup + decay。
    lr_scheduler = optim.linear_schedule(init=0.0, end=1e-3, steps=100)
    #optimizer = optim.AdamW(learning_rate=lr_scheduler)
    optimizer = optim.Muon(learning_rate=lr_scheduler, weight_decay=1e-4)
    #optimizer = optim.Lion(learning_rate=lr_scheduler, weight_decay=1e-4)
    # train 模式
    model.train()
    print(f"epochs: {train_epochs}, train is starting ...")
    for epoch in range(train_epochs):
        t1 = time.perf_counter()
        for i,batch in enumerate(train_data):

            loss, grads = loss_and_grad(model, batch.src, batch.tgt_in, batch.tgt_label,batch.src_mask,batch.tgt_mask)
            # 梯度裁剪，防止爆炸
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            # 惰性求值，必须执行！
            mx.eval(loss, model.parameters(), optimizer.state)

            print(f"\riter {i+1}  / {epoch+1:3d} / {train_epochs} | LR: {optimizer.learning_rate:9f}", end="")
            #break

        t2 = time.perf_counter()
        print(f" | Epoch {epoch+1:3d} | Duration: {(t2 - t1):.2f} | Loss: {loss.item():.4f}")
        if epoch < 10 or (epoch+1) % 2 == 0:  
            model.save_weights(model_path)


#
if __name__ == "__main__":   
    if len(sys.argv) == 1:
        train_epochs = 1
    #
    if len(sys.argv) == 2:
        arg_1 = sys.argv[1]
        print(arg_1)
        if arg_1.isdigit():
            train_epochs = int(arg_1)
    #
    if os.path.exists(model_path):
        print("model is loading ...")
        model.load_weights(model_path)

    print(f"train_epochs: {train_epochs}")
    train()


