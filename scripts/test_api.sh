#!/usr/bin/env bash
# 接口测试：./scripts/test_api.sh [BASE_URL] [API_KEY]
set -u
BASE="${1:-http://127.0.0.1:8000}"
KEY="${2:-sk-xionglingautobaiduscshumi}"
pass=0; fail=0
chk(){ if [ "$2" = "$3" ]; then echo "  PASS  $1 (期望 $3, 实际 $2)"; pass=$((pass+1));
       else echo "  FAIL  $1 (期望 $3, 实际 $2)"; fail=$((fail+1)); fi; }

code(){ curl -s -o /tmp/xl_body -w "%{http_code}" "$@"; }

echo "== 1 健康检查"
chk "GET /health" "$(code "$BASE/health")" "200"

echo "== 2 鉴权：无钥匙应被拒"
chk "无 key 访问 /v1/models" "$(code "$BASE/v1/models")" "401"

echo "== 3 鉴权：错钥匙应被拒"
chk "错 key 访问 /v1/models" \
  "$(code -H "x-api-key: sk-not-the-one" "$BASE/v1/models")" "401"

echo "== 4 鉴权：正确钥匙通过"
chk "正确 key 访问 /v1/models" "$(code -H "x-api-key: $KEY" "$BASE/v1/models")" "200"

echo "== 5 聊天补全"
chk "POST /v1/chat/completions" \
  "$(code -X POST "$BASE/v1/chat/completions" -H "x-api-key: $KEY" \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"雄灵一号"}],"max_tokens":40}')" "200"
grep -q '"choices"' /tmp/xl_body && { echo "  PASS  返回体含 choices"; pass=$((pass+1)); } \
                                || { echo "  FAIL  返回体缺 choices"; fail=$((fail+1)); }

echo "== 6 生成文件"
chk "POST /v1/files" \
  "$(code -X POST "$BASE/v1/files" -H "x-api-key: $KEY" \
     -H "Content-Type: application/json" \
     -d '{"prompt":"训练","filename":"t","ext":"md","max_tokens":60}')" "200"
DL=$(python3 -c "import json;print(json.load(open('/tmp/xl_body'))['download_url'])" 2>/dev/null)
chk "下载生成的文件 $DL" "$(code "$DL")" "200"

echo "== 7 Bearer 形式也能用"
chk "Authorization: Bearer" \
  "$(code -H "Authorization: Bearer $KEY" "$BASE/v1/models")" "200"

echo
echo "结果：通过 $pass 项，失败 $fail 项"
[ "$fail" -eq 0 ]
