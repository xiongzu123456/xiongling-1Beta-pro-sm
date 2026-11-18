#!/usr/bin/env bash
# 在自己电脑上开一个公网地址（临时，进程在就活着）
# 用法：./scripts/public_tunnel.sh
set -e
cd "$(dirname "$0")/.."
PORT="${PORT:-8000}"

# 1. 起服务
python3 -m uvicorn api.server:app --host 127.0.0.1 --port "$PORT" &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
sleep 3

# 2. 下载 cloudflared（已在 PATH 中则跳过）
BIN=""
if command -v cloudflared >/dev/null 2>&1; then
  BIN="cloudflared"
else
  OS=$(uname -s | tr 'A-Z' 'a-z'); ARCH=$(uname -m)
  case "$ARCH" in x86_64) A=amd64;; arm64|aarch64) A=arm64;; *) A=amd64;; esac
  mkdir -p ./bin
  curl -sL -o ./bin/cloudflared "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-${OS}-${A}"
  chmod +x ./bin/cloudflared
  BIN="./bin/cloudflared"
fi

echo "正在建立公网通道，等 10 秒左右…"
"$BIN" tunnel --url "http://127.0.0.1:${PORT}" --no-autoupdate 2>&1 | tee /tmp/xl_tunnel.out &
for i in $(seq 1 30); do
  URL=$(grep -o "https://[a-z0-9-]*\.trycloudflare\.com" /tmp/xl_tunnel.out 2>/dev/null | head -1)
  [ -n "$URL" ] && break
  sleep 1
done
echo
echo "====================================================="
echo " 公网 API 地址：${URL:-(未获取到，请看上面的日志)}"
echo " 健康检查：    ${URL}/health"
echo " API Key：     sk-xionglingautobaiduscshumi"
echo " 注意：关掉这个窗口，地址就失效"
echo "====================================================="
wait
