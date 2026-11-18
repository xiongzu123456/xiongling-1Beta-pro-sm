#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xiongling-1Beta-pro-sm API 服务

统一使用一把 API Key（只有一个）：
    默认 sk-xionglingautobaiduscshumi
    可用环境变量 XL_API_KEY 覆盖（部署到公开平台时建议换新 key）

启动：
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import Engine  # noqa: E402

MODEL_NAME = "xiongling-1Beta-pro-sm"
API_KEY = os.getenv("XL_API_KEY", "sk-xionglingautobaiduscshumi")
MODEL_DIR = os.path.join(ROOT, "model")
OUT_DIR = os.path.join(ROOT, "generated")
os.makedirs(OUT_DIR, exist_ok=True)

engine = Engine(MODEL_DIR)

app = FastAPI(title=f"{MODEL_NAME} API", version="1.0.0-beta")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ---------------- 鉴权：只认一把钥匙 ----------------
def require_key(x_api_key: str = Header(default=""),
                authorization: str = Header(default="")):
    token = x_api_key.strip()
    if not token and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail={
            "error": {"type": "missing_api_key",
                      "message": "缺少 API Key，请在 x-api-key 请求头中提供"}})
    if token != API_KEY:
        raise HTTPException(status_code=401, detail={
            "error": {"type": "invalid_api_key",
                      "message": "API Key 不正确，本服务只签发一把钥匙"}})
    return True


# ---------------- 请求模型 ----------------
class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 20


class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 20


class FileRequest(BaseModel):
    prompt: str
    filename: str = "output"
    ext: str = "md"          # md / txt / json / html / py
    max_tokens: int = 256
    temperature: float = 0.8


# ---------------- 路由 ----------------
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "backend": engine.backend,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


@app.get("/v1/models")
def models(_: bool = Depends(require_key)):
    return {"object": "list", "data": [engine.info()]}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest, _: bool = Depends(require_key)):
    prompt = "\n".join(m.content for m in req.messages) or "你好"
    try:
        text = engine.generate(prompt, req.max_tokens, req.temperature, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=503, detail={"error": str(e)})
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(text),
                  "total_tokens": len(prompt) + len(text)},
    }


@app.post("/v1/generate")
def generate(req: GenRequest, _: bool = Depends(require_key)):
    try:
        text = engine.generate(req.prompt, req.max_tokens,
                               req.temperature, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=503, detail={"error": str(e)})
    return {"model": MODEL_NAME, "prompt": req.prompt, "text": text}


@app.post("/v1/files")
def make_file(req: FileRequest, request: Request,
              _: bool = Depends(require_key)):
    """让模型直接产出一个文件，返回下载地址。"""
    try:
        body = engine.generate(req.prompt, req.max_tokens, req.temperature)
    except Exception as e:
        raise HTTPException(status_code=503, detail={"error": str(e)})

    safe_ext = req.ext if req.ext.isalnum() else "txt"
    name = "%s-%s.%s" % (req.filename.replace("/", "_").strip() or "output",
                         uuid.uuid4().hex[:6], safe_ext)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)

    base = str(request.base_url).rstrip("/")
    return {"file": name, "size": os.path.getsize(path),
            "download_url": f"{base}/files/{name}", "preview": body[:200]}


@app.get("/files/{name}")
def get_file(name: str):
    path = os.path.join(OUT_DIR, os.path.basename(name))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path, filename=os.path.basename(name))


@app.exception_handler(HTTPException)
def on_http_error(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content=exc.detail)
