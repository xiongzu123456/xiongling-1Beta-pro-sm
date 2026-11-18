#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xiongling-1Beta-pro-sm 训练脚本

两种后端：
  1) numpy  —— 免 GPU 极速冒烟后端（字符级二阶梯模型），几秒钟出模型文件，
               用于验证「训练 → 保存 → 加载 → 推理」整条链路是否通畅。
  2) torch  —— 正式的 LoRA 微调后端，需在免费 GPU 平台（Colab / Kaggle / HF）
               上运行，产物是 LoRA adapter。

用法：
  python3 train.py --backend numpy --data data/sample.txt --out model
  python3 train.py --backend torch --data data/corpus.txt \
      --base Qwen/Qwen3-0.6B --out model/adapter --epochs 3
"""
import argparse
import json
import math
import os
import time
from collections import defaultdict

MODEL_NAME = "xiongling-1Beta-pro-sm"
VERSION = "1.0.0-beta"


# ----------------------------------------------------------------------------
# 后端一：numpy 字符级二阶模型（冒烟测试用）
# ----------------------------------------------------------------------------
def train_numpy(data_path: str, out_dir: str, epochs: int = 1, smooth: float = 0.2):
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read().replace("\r\n", "\n")

    chars = sorted(set(text))
    idx = {c: i for i, c in enumerate(chars)}
    V = len(chars)

    # bigram[i][j] = count(char_i -> char_j)
    bigram = [[0.0] * V for _ in range(V)]
    unigram = [0.0] * V

    for epoch in range(max(1, epochs)):
        for a, b in zip(text[:-1], text[1:]):
            bigram[idx[a]][idx[b]] += 1.0
        for c in text:
            unigram[idx[c]] += 1.0
        print(f"[epoch {epoch + 1}/{epochs}] 已统计 {len(text)} 个字符")

    # 平滑 + 归一化为条件概率
    for i in range(V):
        row = bigram[i]
        total = sum(row) + smooth * V
        bigram[i] = [(row[j] + smooth) / total for j in range(V)]

    total_uni = sum(unigram)
    unigram = [u / total_uni for u in unigram]

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "model.json")
    payload = {
        "name": MODEL_NAME,
        "version": VERSION,
        "backend": "numpy-char-bigram",
        "vocab": chars,
        "unigram": unigram,
        "bigram": bigram,
        "meta": {
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "data_file": os.path.basename(data_path),
            "data_chars": len(text),
            "vocab_size": V,
            "epochs": epochs,
            "smooth": smooth,
            "params": V * V + V,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"[ok] 模型已保存 -> {path}")
    print(f"[ok] 词表大小 {V}，参数约 {V * V + V} 个")
    return path


# ----------------------------------------------------------------------------
# 后端二：torch + LoRA 正式微调（在免费 GPU 平台运行）
# ----------------------------------------------------------------------------
def train_torch(data_path: str, out_dir: str, base: str, epochs: int, lr: float):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto",
    )
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    raw = load_dataset("text", data_files=data_path, split="train")

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=1024)

    ds = raw.map(tok, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    Trainer(
        model=model, args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    ).train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "xiongling.json"), "w", encoding="utf-8") as f:
        json.dump({"name": MODEL_NAME, "version": VERSION,
                   "base_model": base, "backend": "torch-lora"}, f, ensure_ascii=False)
    print(f"[ok] LoRA adapter 已保存 -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    ap.add_argument("--data", default="data/sample.txt")
    ap.add_argument("--out", default="model")
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    print(f"=== 开始训练 {MODEL_NAME} (backend={args.backend}) ===")
    t0 = time.time()
    if args.backend == "numpy":
        train_numpy(args.data, args.out, args.epochs)
    else:
        train_torch(args.data, args.out, args.base, args.epochs, args.lr)
    print(f"=== 训练结束，耗时 {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
