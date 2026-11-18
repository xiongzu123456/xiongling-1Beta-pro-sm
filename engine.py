#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xiongling-1Beta-pro-sm 推理引擎

两种后端，自动切换：
  upstream  —— 真模型：转发到 OpenAI 格式的推理服务（默认免费通道，配 HF_TOKEN 可走 Hugging Face）
  local     —— 本地小模型：model/model.json（numpy）或 model/adapter/（torch LoRA），作为兜底

环境变量：
  XL_BACKEND=auto|upstream|local   默认 auto（先用真模型，失败回落本地）
  XL_UPSTREAM_URL                 默认 https://text.pollinations.ai/openai
  XL_UPSTREAM_MODEL               默认 openai
  HF_TOKEN                        走 Hugging Face 时填 token
  XL_SYSTEM_PROMPT                给真模型的人设
"""
import json
import os
import random
import urllib.request

MODEL_NAME = "xiongling-1Beta-pro-sm"

UPSTREAM_URL = os.getenv("XL_UPSTREAM_URL", "https://text.pollinations.ai/openai")
UPSTREAM_MODEL = os.getenv("XL_UPSTREAM_MODEL", "openai")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
BACKEND_MODE = os.getenv("XL_BACKEND", "auto").strip().lower()
SYSTEM_PROMPT = os.getenv(
    "XL_SYSTEM_PROMPT",
    "你是雄灵一号（xiongling-1Beta-pro-sm），一个中文 AI 助手。回答直接、简洁，不要啰嗦，不要自称其他模型。",
)


class Engine:
    def __init__(self, model_dir: str = "model"):
        self.model_dir = model_dir
        self.backend = None
        self.meta = {}
        self.upstream_error = None
        self._load()

    # ---------------- 本地加载 ----------------
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
            except Exception as e:
                self.meta = {"load_error": str(e)}

        self.backend = "unavailable"
        self.meta = {"hint": "先跑 python3 train.py --backend numpy"}

    # ---------------- 是否启用真模型 ----------------
    @property
    def upstream_on(self):
        if BACKEND_MODE == "local":
            return False
        return bool(UPSTREAM_URL)

    def active_backend(self):
        return "upstream" if self.upstream_on else self.backend

    # ---------------- 真模型转发 ----------------
    def _call_upstream(self, messages, max_tokens, temperature):
        body = {
            "model": UPSTREAM_MODEL,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if HF_TOKEN:
            headers["Authorization"] = "Bearer " + HF_TOKEN
        req = urllib.request.Request(
            UPSTREAM_URL, data=json.dumps(body).encode("utf-8"),
            headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        choice = (data.get("choices") or [{}])[0]
        txt = ((choice.get("message") or {}).get("content") or choice.get("text") or "")
        return str(txt).strip()

    # ---------------- 对话（主入口） ----------------
    def chat(self, messages, max_tokens=128, temperature=0.8, top_k=20):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            msgs.append({"role": role if role in ("system", "user", "assistant") else "user",
                         "content": content})

        if self.upstream_on:
            try:
                out = self._call_upstream(msgs, max_tokens, temperature)
                if out:
                    self.upstream_error = None
                    return out
            except Exception as e:
                self.upstream_error = "%s: %s" % (type(e).__name__, e)

        prompt = "\n".join(m["content"] for m in msgs[1:]) or "你好"
        return self.generate_local(prompt, max_tokens, temperature, top_k)

    # 兼容旧调用名
    def generate(self, prompt, max_new_tokens=128, temperature=0.8, top_k=20):
        return self.chat([{"role": "user", "content": prompt}],
                         max_new_tokens, temperature, top_k)

    # ---------------- 本地推理 ----------------
    def generate_local(self, prompt, n=128, temperature=0.8, top_k=20):
        if self.backend == "numpy-char-bigram":
            return self._gen_numpy(prompt, n, temperature, top_k)
        if self.backend == "torch-lora":
            return self._gen_torch(prompt, n, temperature)
        raise RuntimeError("本地模型未加载：%s" % self.meta.get("hint", ""))

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
        desc = ("真模型转发 · %s" % UPSTREAM_MODEL) if self.upstream_on \
            else ("本地 %s" % self.backend)
        return {
            "id": MODEL_NAME,
            "object": "model",
            "backend": self.active_backend(),
            "local_backend": self.backend,
            "description": desc,
            "version": getattr(self, "card", {}).get("version", "1.0.0-beta"),
            "meta": self.meta,
            "upstream_error": self.upstream_error,
        }
