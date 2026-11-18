# xiongling-1Beta-pro-sm

一个可以「训练 → 测试 → 打 API → 上网页」的小模型工程，全部走免费资源。
模型名：**xiongling-1Beta-pro-sm**，API Key 只有一把：`sk-xionglingautobaiduscshumi`。

---

## 1. 目录结构

```
xiongling-1Beta-pro-sm/
├── configs/xiongling-1Beta-pro-sm.json   # 模型配置卡（名字/结构/LoRA 参数）
├── data/sample.txt                       # 训练语料（可以换成你自己的）
├── train.py                              # 训练脚本（numpy 冒烟 / torch LoRA 正式）
├── engine.py                             # 推理引擎（自动选择后端）
├── model/model.json                      # 训练产物（执行 make train 生成）
├── api/server.py                         # FastAPI 服务（带 API Key 鉴权）
├── api/Dockerfile                        # 容器镜像（HF Space / Render 部署用）
├── app/index.html                        # 前端 App（GitHub Pages 直接托管）
├── .github/workflows/pages.yml           # 推送即自动发布 Pages
├── scripts/test_api.sh                   # 接口测试（9 项）
├── tests/test_engine.py                  # 单元测试（3 项）
└── README.md
```

---

## 2. 30 秒本地跑通（四步）

```bash
pip install -r api/requirements.txt                                             # 1. 装依赖
python3 train.py --backend numpy --data data/sample.txt --out model --epochs 3  # 2. 训练
python3 -m uvicorn api.server:app --port 8000                                   # 3. 起服务
./scripts/test_api.sh                                                           # 4. 测试
```

然后浏览器打开 `app/index.html`，在「设置」里填 `http://127.0.0.1:8000` 和那把 Key 即可对话。

### 实测结果（开发机已验证）

| 检查项 | 结果 |
|---|---|
| 训练（numpy 后端，3 epoch） | 471 字符语料，词表 197，参数 39006，耗时 0.1s |
| 单元测试 `pytest tests/ -q` | 3 passed |
| 接口测试 `scripts/test_api.sh` | 9 passed / 0 failed |
| 无 Key 访问 `/v1/models` | 401 `missing_api_key` |
| 错误 Key 访问 | 401 `invalid_api_key` |
| 正确 Key 调用聊天 / 生成文件 / 下载文件 | 200 / 200 / 200 |

> `numpy` 后端是**字符级小模型**，作用是在没有任何显卡的环境里几十秒把「训练—保存—加载—推理—出文件」整条链跑一遍。它写得出中文，但语义能力很弱。要正经验果，用第 4 节的 LoRA 后端。

---

## 3. 免费训练平台怎么选（都能用 GitHub 一键登录）

| 平台 | 免费额度 | 登录方式 | 适合干什么 |
|---|---|---|---|
| **Hugging Face** | CPU 2 核 16G、免费公开仓库、免费 Space | 支持 GitHub OAuth 一键登录 | 建仓库、跑小模型、托管 API（**主推**） |
| **Google Colab** | 免费 T4 GPU（有时长限制） | Google 账号（无 GitHub 登录） | 真正跑 LoRA 微调 |
| **Kaggle** | 免费 GPU 每周约 30 小时 | 邮箱 / Google | 同样可跑 LoRA，量大 |

**结论**：要「免费 + GitHub 一键登录」→ 用 **Hugging Face**。

1. 打开 `huggingface.co` → `Sign up` → 点 **Continue with GitHub**；
2. 授权后即登录，无需邮箱验证（首次会要求同意条款）；
3. 新建 Model 仓库，名字就叫 `xiongling-1Beta-pro-sm`。

想拿到 GPU，再补一个 Google 账号开 Colab（免费 T4），或用 Kaggle。

---

## 4. 正式训练（LoRA，免费 GPU）

在 Colab / Kaggle 的 Notebook 里：

```bash
git clone https://github.com/xiongzu123456/xiongling-1Beta-pro-sm && cd xiongling-1Beta-pro-sm
pip install -r api/requirements.txt
pip install torch transformers peft datasets accelerate
```

准备语料：把 `data/corpus.txt` 换成你自己的文本（一行一段，或直接一整个 txt）。

```bash
python3 train.py --backend torch \
  --data data/corpus.txt \
  --base Qwen/Qwen3-0.6B \
  --out model/adapter \
  --epochs 3 --lr 2e-4
```

参数在 `configs/xiongling-1Beta-pro-sm.json`：LoRA r=16 / alpha=32 / dropout=0.05，目标模块 q/k/v/o_proj，max_seq_len=1024，batch=1 + 梯度累积 8（显存小也能跑）。

训练完把 `model/adapter/` 推到 Hugging Face：

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload 你的用户名/xiongling-1Beta-pro-sm model/adapter .
```

`engine.py` 会自动识别：有 `model/model.json` 用冒烟后端，有 `model/adapter/` 用 LoRA 后端，不用改代码。

---

## 5. 部署 API（免费）

### 方式 A：Hugging Face Space（推荐，免费 CPU）

1. 新建 Space → SDK 选 **Docker** → 名称 `xiongling-1Beta-pro-sm`；
2. 把本项目文件上传（`api/Dockerfile` 已写好，端口 7860）；
3. Settings → Variables and secrets 新增 Secret：`XL_API_KEY` = `sk-xionglingautobaiduscshumi`（勿建成公开 Variable）；
4. 构建完成后地址形如 `https://用户名-xiongling-1beta-pro-sm.hf.space`，这就是前端要填的 API 地址。

### 方式 B：Render / Railway 免费额度

New → Web Service → 连接 GitHub 仓库 → Build: `pip install -r api/requirements.txt` → Start: `uvicorn api.server:app --host 0.0.0.0 --port $PORT` → 环境变量加 `XL_API_KEY`。

---

## 6. 部署 App 站点到 GitHub Pages

本仓库已内置 `.github/workflows/pages.yml`：

1. 仓库 Settings → Pages → Source 选 **GitHub Actions**；
2. 推送代码后 Actions 自动把 `app/` 发布成站点；
3. 访问 `https://xiongzu123456.github.io/xiongling-1Beta-pro-sm/`。

打开页面 → 右上角「设置」→ 填 API 地址 + Key → 点「保存并测试连接」，看到「后端在线」就能用。
页面有两个模式：**聊天**（`/v1/chat/completions`）、**生成文件**（`/v1/files`，直接吐出一个可下载的 .md）。

---

## 7. 接口文档

所有 `/v1/*` 接口都必须带 Key，两种写法任选：

```
x-api-key: sk-xionglingautobaiduscshumi
Authorization: Bearer sk-xionglingautobaiduscshumi
```

| 方法 | 路径 | 鉴权 | 作用 |
|---|---|---|---|
| GET | `/health` | 否 | 健康检查 |
| GET | `/v1/models` | 是 | 模型信息 |
| POST | `/v1/chat/completions` | 是 | OpenAI 格式聊天补全 |
| POST | `/v1/generate` | 是 | 纯文本续写 |
| POST | `/v1/files` | 是 | 让模型直接产出一个文件 |
| GET | `/files/{name}` | 否 | 下载生成的文件 |

```bash
# 聊天
curl -X POST https://你的地址/v1/chat/completions \
  -H "x-api-key: sk-xionglingautobaiduscshumi" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"雄灵一号能做什么"}],"max_tokens":120}'

# 让它做出一个文件
curl -X POST https://你的地址/v1/files \
  -H "x-api-key: sk-xionglingautobaiduscshumi" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"写一份训练记录","filename":"xiongling-note","ext":"md","max_tokens":200}'
```

服务端只签发**一把**钥匙，其他 key 一律 401，逻辑在 `api/server.py`：`token != API_KEY → 401 invalid_api_key`。

---

## 8. 安全提醒

- Key 写在前端 HTML 里 = 谁打开网页都能看到。自己玩没问题；**要给别人用就去 HF Space 的 Secret 里换一把新 key**，前端同步改。
- `.env` 已在 `.gitignore` 里，别把真 key 提交到公开仓库。
- 免费 Space 长时间无人访问会休眠，第一次请求可能等 30 秒左右才唤醒。

---

## 9. 常见问题

| 现象 | 原因 / 解决 |
|---|---|
| 前端提示 `Failed to fetch` | API 地址错 / 后端没起 / 跨域。服务端已开 `allow_origins=["*"]`，多半是地址写错或 Space 休眠中 |
| `/health` 通但 `/v1/models` 401 | Key 没填或填错 |
| 启动报 `模型未加载` | 还没训练，先跑 `python3 train.py --backend numpy` |
| 换成 LoRA 后还是旧输出 | 删掉 `model/model.json`，`engine.py` 才会优先走 `model/adapter/` |
| `numpy` 输出不像人话 | 正常，字符级小模型只看相邻字概率，正式效果请用 LoRA |
