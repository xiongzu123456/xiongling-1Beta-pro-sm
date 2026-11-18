#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xiongling-1Beta-pro-sm 推理引擎

根据 model/ 目录里实际存在的产物自动选择后端：
  - model/model.json        -> numpy 冒烟后端
  - model/adapter/          -> torch + LoRA 正式后端
"""
import json
import os
import random

MODEL_NAME = "xiongling-1Beta-pro-sm"


class Engine:
    def __init__(self, model_dir: str = "model"):
        self.model_dir = model_dir
        self.backend = None
        self.meta = {}
        self._load()

    # ---------------- 加载 ----------------
    def _load(self):
        json_path = os.path.join(self.model_dir, "model.json")
        adapter_dir = os.path.join(self.model_dir, "adapter")

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                self.card = json.load(f)
            self.vocab = self.card["vocab"]
            self.idx = {c: i for i, c in enumerate(self.vocab)}
            self.bigram = self.card["bigram"]
            self.unigram = self.card["unigram"]
            self.backend = "numpy-char-bigram"
            self.meta = self.card.get("meta", {})
            return

        if os.path.isdir(adapter_dir):
            try:
                import torch
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer

                card_path = os.path.join(adapter_dir, "xiongling.json")
                base = "Qwen/Qwen3-0.6B"
                if os.path.exists(card_path):
                    with open(card_path, "r", encoding="utf-8") as f:
                        base = json.load(f).get("base_model", base)

                self.tok = AutoTokenizer.from_pretrained(adapter_dir,
                                                         trust_remote_code=True)
                base_model = AutoModelForCausalLM.from_pretrained(
                    base, torch_dtype=torch.float32, trust_remote_code=True)
                self.model = PeftModel.from_pretrained(base_model, adapter_dir).eval()
                self.torch = torch
                self.backend = "torch-lora"
                self.meta = {"base_model": base, "adapter": adapter_dir}
                return
            except Exception as e:  # 依赖缺失时退化
                self.meta = {"load_error": str(e)}

        self.backend = "unavailable"
        self.meta = {"hint": "先跑 python3 train.py --backend numpy"}

    # ---------------- 推理 ----------------
    def generate(self, prompt: str, max_new_tokens: int = 128,
                 temperature: float = 0.8, top_k: int = 20) -> str:
        if self.backend == "numpy-char-bigram":
            return self._gen_numpy(prompt, max_new_tokens, temperature, top_k)
        if self.backend == "torch-lora":
            return self._gen_torch(prompt, max_new_tokens, temperature)
        raise RuntimeError("模型未加载：%s" % self.meta.get("hint", ""))

    def _gen_numpy(self, prompt, n, temperature, top_k):
        out = list(prompt)
        last = prompt[-1] if prompt else random.choice(self.vocab)
        cur = self.idx.get(last)
        if cur is None:
            cur = random.choices(range(len(self.vocab)), weights=self.unigram)[0]
            out.append(self.vocab[cur])

        for _ in range(int(n)):
            row = self.bigram[cur]
            if temperature <= 0.01:
                nxt = max(range(len(row)), key=lambda i: row[i])
            else:
                # 温度 + top-k
                cand = sorted(range(len(row)), key=lambda i: -row[i])[:max(1, top_k)]
                weights = [row[i] ** (1.0 / temperature) for i in cand]
                nxt = random.choices(cand, weights=weights)[0]
            out.append(self.vocab[nxt])
            cur = nxt
        return "".join(out)

    def _gen_torch(self, prompt, n, temperature):
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt")
        with self.torch.no_grad():
            gen = self.model.generate(**ids, max_new_tokens=int(n),
                                      do_sample=temperature > 0.01,
                                      temperature=max(temperature, 0.01),
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(gen[0][ids["input_ids"].shape[-1]:],
                               skip_special_tokens=True)

    # ---------------- 元信息 ----------------
    def info(self):
        return {
            "id": MODEL_NAME,
            "object": "model",
            "backend": self.backend,
            "version": getattr(self, "card", {}).get("version", "1.0.0-beta"),
            "meta": self.meta,
        }
