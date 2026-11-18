#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xiongling-1Beta-pro-sm API 服务

一把钥匙：
    默认 sk-xionglingautobaiduscshumi（可用环境变量 XL_API_KEY 覆盖）

兼容多种客户端路径：
    OpenAI 风格    POST /v1/chat/completions  或  POST /chat/completions
    Anthropic 风格 POST /v1/messages         或  POST /messages
    根路径 GET /  给出可用的端点清单

启动：uvicorn api.server:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
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
                      "message": "缺少 API Key，请在 x-api-key 或 Authorization: Bearer 中提供"}})
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


class MsgItem(BaseModel):
    role: str = "user"
    content: object = ""


class MessagesRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[MsgItem] = Field(default_factory=list)
    max_tokens: int = 128
    temperature: float = 0.8


class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 20


class FileRequest(BaseModel):
    prompt: str
    filename: str = "output"
    ext: str = "md"
    max_tokens: int = 256
    temperature: float = 0.8


def _flat(messages) -> str:
    parts = []
    for m in messages:
        c = m.content
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(str(b.get("text", "")))
        else:
            parts.append(str(c))
    return "\n".join(parts) or "你好"


# ---------------- 根路径：端点清单 ----------------
@app.get("/", response_class=HTMLResponse)
def home():
    lines = "".join(f"<li><code>{m} {p}</code> — {d}</li>" for m, p, d in [
        ("GET", "/health", "健康检查（不需要 Key）"),
        ("GET", "/v1/models", "模型信息"),
        ("POST", "/v1/chat/completions", "OpenAI 风格聊天（也支持 /chat/completions）"),
        ("POST", "/v1/messages", "Anthropic 风格聊天（也支持 /messages）"),
        ("POST", "/v1/generate", "纯文本续写"),
        ("POST", "/v1/files", "让模型产出一个可下载的文件"),
        ("GET", "/files/{name}", "下载生成的文件"),
    ])
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>{MODEL_NAME} API</title>
<body style="font:15px/1.7 -apple-system,sans-serif;max-width:760px;margin:40px auto;padding:0 18px">
<h2>{MODEL_NAME} API 在线</h2>
<p>当前后端：<b>{engine.active_backend()}</b>（本地：{engine.backend}）　状态：<b>ok</b></p>
<p>所有 <code>/v1/*</code> 接口都需要 Key（请求头 <code>x-api-key</code> 或 <code>Authorization: Bearer</code>）。</p>
<ul>{lines}</ul>
<p>示例：<code>curl -X POST /v1/chat/completions -H "x-api-key: &lt;你的key&gt;" -H "Content-Type: application/json" -d '{{"messages":[{{"role":"user","content":"你好"}}]}}'</code></p>
</body></html>"""


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME,
            "backend": engine.active_backend(), "local": engine.backend,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


@app.get("/v1/models")
def models(_: bool = Depends(require_key)):
    return {"object": "list", "data": [engine.info()]}


# ---------------- OpenAI 风格 ----------------
def _do_chat(req: ChatRequest) -> dict:
    prompt = _flat(req.messages)
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        text = engine.chat(msgs, req.max_tokens, req.temperature, req.top_k)
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
        "provider": engine.active_backend(),
    }


@app.post("/v1/chat/completions")
def chat(req: ChatRequest, _: bool = Depends(require_key)):
    return _do_chat(req)


@app.post("/chat/completions")
def chat_alias(req: ChatRequest, _: bool = Depends(require_key)):
    """给 base_url 已经带 /v1 的客户端兜底。"""
    return _do_chat(req)


# ---------------- Anthropic 风格 ----------------
def _do_messages(req: MessagesRequest) -> dict:
    prompt = _flat(req.messages)
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    if not isinstance(msgs[0]["content"], str):
        msgs = [{"role": m.role, "content": _flat([m])} for m in req.messages]
    try:
        text = engine.chat(msgs, req.max_tokens, req.temperature)
    except Exception as e:
        raise HTTPException(status_code=503, detail={"error": str(e)})
    return {
        "id": "msg_" + uuid.uuid4().hex[:12],
        "type": "message",
        "role": "assistant",
        "model": MODEL_NAME,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": len(prompt), "output_tokens": len(text)},
    }


@app.post("/v1/messages")
def messages(req: MessagesRequest, _: bool = Depends(require_key)):
    return _do_messages(req)


@app.post("/messages")
def messages_alias(req: MessagesRequest, _: bool = Depends(require_key)):
    return _do_messages(req)


# ---------------- 文本与文件 ----------------
@app.post("/v1/generate")
def generate(req: GenRequest, _: bool = Depends(require_key)):
    try:
        text = engine.generate(req.prompt, req.max_tokens,
                               req.temperature, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=503, detail={"error": str(e)})
    return {"model": MODEL_NAME, "prompt": req.prompt, "text": text,
            "provider": engine.active_backend()}


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
