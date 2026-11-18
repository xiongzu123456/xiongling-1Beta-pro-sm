#!/usr/bin/env bash
# 一键本地跑通：训练 → 起服务 → 测试
set -e
cd "$(dirname "$0")/.."
echo "[1/3] 训练"
python3 train.py --backend numpy --data data/sample.txt --out model --epochs 3
echo "[2/3] 启动 API"
python3 -m uvicorn api.server:app --host 127.0.0.1 --port 8000 &
PID=$!
sleep 4
echo "[3/3] 接口测试"
./scripts/test_api.sh http://127.0.0.1:8000 || true
echo "服务已启动，PID=$PID，前端打开 app/index.html"
wait $PID
