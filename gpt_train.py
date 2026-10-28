import sys
import os
import time
import mlx.core as mx
import mlx.nn as nn
from mlxTFdataset import *
from mlxTransformer import *
from loader import *

# defalut
train_epochs = 10

# 第 src 列为源文， 第 tgt 列为目标文
train_file = "data/data_train.csv"

data = GptData(fpath=train_file,
                  sp_model_file=sp_model_gpt, 
                  batch_size=BATCH_SIZE)


def loss_fn_gpt(model, tokens, labels, attn_mask, loss_mask): 
    x = mx.array(tokens, dtype=mx.int32)
    labels = mx.array(labels, dtype=mx.int32)
    
    logits = model(x, attn_mask)
    B, T, V = logits.shape
    # print(f"B: {B}, T: {T}, V: {V}")
    logits_flat = logits.reshape(-1, V)
    labels_flat = labels.reshape(-1)
    lm_mask_flat = loss_mask.reshape(-1)
    
    ce = nn.losses.cross_entropy(
        logits_flat,
        labels_flat,
        label_smoothing=0.2,
        reduction="none"
    )
    ce = ce * lm_mask_flat
    valid = mx.sum(lm_mask_flat)
    # 防止除 0
    # MLX推荐写法，不要if判断（if会破坏jit）
    loss = mx.sum(ce) / (valid + 1e-8)
    return loss

loss_and_grad = nn.value_and_grad(model, loss_fn_gpt)       

def gpt_train():
    global DEBUG
    if train_epochs > 10:
        DEBUG = False
    print("In Debug Mode: ", DEBUG)
    train_data = data.batches
    # 原生 Adam 固定学习率很难训 Transformer。一般用 warmup + decay。
    lr_scheduler = optim.linear_schedule(init=0.0, end=LEARNING_RATE, steps=100)
    #optimizer = optim.AdamW(learning_rate=lr_scheduler)
    optimizer = optim.Muon(learning_rate=lr_scheduler, weight_decay=1e-4)
    #optimizer = optim.Lion(learning_rate=lr_scheduler, weight_decay=1e-4)
    # train 模式
    model.train()
    print(f"epochs: {train_epochs}, train is starting ...")
    for epoch in range(train_epochs):
        t1 = time.perf_counter()
        for i,batch in enumerate(train_data):
            # print("batch.tokens: ", batch.tokens[0], batch.tokens.shape)
            # print("batch.labels: ", batch.labels[0], batch.labels.shape)
            # print("batch.attn_mask: ", batch.attn_mask[0][0][0], batch.attn_mask.shape)
            # print("batch.loss_mask: ", batch.loss_mask[0], batch.loss_mask.shape)
            #print("src decode:", spm_gpt.decode(batch.tokens[0]))
            #print("batch.attn_mask: ", batch.src_mask[0][0][0], batch.src_mask.shape)
            #print("batch.loss_mask: ", batch.loss_mask[0], batch.loss_mask.shape)
            #break
            loss, grads = loss_and_grad(model, batch.tokens, batch.labels, batch.attn_mask, batch.loss_mask)
            # 梯度裁剪，防止爆炸
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            # 惰性求值，必须执行！
            mx.eval(loss, model.parameters(), optimizer.state)

            print(f"\riter {i+1} / {epoch+1:3d} / {train_epochs} | Lr: {optimizer.learning_rate:9f}", end="")
            #break

        t2 = time.perf_counter()
        print(f" | {time.strftime('%H:%M')} | Epoch {epoch+1:3d} | Duration: {(t2 - t1):.2f} | Loss: {loss.item():.4f}")
        if epoch < 10 or (epoch+1) % 2 == 0:
            model.save_weights(model_path)


#
if __name__ == "__main__":  
    if len(sys.argv) == 1:
        train_epochs = train_epochs
    #
    if len(sys.argv) == 2:
        arg_1 = sys.argv[1]
        if arg_1.isdigit():
            train_epochs = int(arg_1)
    #
    print("config: ENABLE_GPT: ", ENABLE_GPT)
    assert ENABLE_GPT == True, "pls set ENABLE_GPT=True in config.py"
    print(f"train_epochs: {train_epochs}")
    t_start = time.perf_counter()
    gpt_train()
    t_stop = time.perf_counter()
    print(f"average time: {((t_stop - t_start)/float(train_epochs)):.2f}")


