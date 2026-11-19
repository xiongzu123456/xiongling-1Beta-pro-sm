// xiongling-1Beta-pro-sm · Cloudflare Worker 版 API
// 用法：Cloudflare 控制台 → Workers & Pages → Create Worker → 粘贴本文件 → Deploy
// 环境变量（可选）：XL_API_KEY、XL_UPSTREAM_URL、XL_UPSTREAM_MODEL

const DEFAULT_KEY = 'sk-xionglingautobaiduscshumi';
const DEFAULT_UPSTREAM = 'https://text.pollinations.ai/openai';
const DEFAULT_MODEL = 'openai';
const MODEL_NAME = 'xiongling-1Beta-pro-sm';
const SYSTEM_PROMPT = '你是雄灵一号（xiongling-1Beta-pro-sm），一个中文 AI 助手。回答直接、简洁，不要啰嗦，不要自称其他模型。';

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type,x-api-key,Authorization',
  'Access-Control-Max-Age': '86400',
};

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8', ...CORS },
  });

const err = (type, message, status = 401) =>
  json({ error: { type, message } }, status);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, '') || '/';

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS });
    }

    if (path === '/health') {
      return json({ status: 'ok', model: MODEL_NAME, backend: 'cloudflare-worker',
                    upstream: env.XL_UPSTREAM_URL || DEFAULT_UPSTREAM,
                    time: new Date().toISOString() });
    }

    if (path === '/' && request.method === 'GET') {
      return new Response(
        `<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>${MODEL_NAME} API</title>
<body style="font:15px/1.7 -apple-system,sans-serif;max-width:720px;margin:40px auto;padding:0 18px">
<h2>${MODEL_NAME} API 在线</h2>
<p>所有接口都需要 Key：请求头 <code>x-api-key</code> 或 <code>Authorization: Bearer</code></p>
<ul>
<li><code>GET /health</code> 健康检查</li>
<li><code>POST /v1/chat/completions</code> OpenAI 风格聊天</li>
<li><code>POST /v1/messages</code> Anthropic 风格聊天</li>
</ul>
</body></html>`,
        { headers: { 'Content-Type': 'text/html; charset=utf-8', ...CORS } }
      );
    }

    const apiKey = env.XL_API_KEY || DEFAULT_KEY;
    const raw = request.headers.get('x-api-key') ||
                (request.headers.get('authorization') || '').replace(/^Bearer\s+/i, '').trim();
    if (!raw) return err('missing_api_key', '缺少 API Key，请在 x-api-key 请求头中提供');
    if (raw !== apiKey) return err('invalid_api_key', 'API Key 不正确，本服务只签发一把钥匙');

    if (path !== '/v1/chat/completions' && path !== '/chat/completions' &&
        path !== '/v1/messages' && path !== '/messages') {
      return json({ error: { type: 'not_found', message: '没有这个接口，可用：/v1/chat/completions、/v1/messages' } }, 404);
    }

    let body;
    try { body = await request.json(); } catch (e) { return err('bad_request', '请求体不是合法 JSON', 400); }

    const messages = [{ role: 'system', content: SYSTEM_PROMPT }];
    for (const m of (body.messages || [])) {
      let c = m.content;
      if (Array.isArray(c)) c = c.filter(b => b && b.type === 'text').map(b => b.text).join('\n');
      messages.push({ role: ['system','user','assistant'].includes(m.role) ? m.role : 'user', content: String(c ?? '') });
    }
    if (!body.messages || !body.messages.length) messages.push({ role: 'user', content: '你好' });

    const upstream = env.XL_UPSTREAM_URL || DEFAULT_UPSTREAM;
    let text = '';
    let lastErr = '';
    // 上游偶发空响应 / 限流，自动重试一次
    for (let attempt = 1; attempt <= 2; attempt++) {
      try {
        const up = await fetch(upstream, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...(env.HF_TOKEN ? { Authorization: 'Bearer ' + env.HF_TOKEN } : {}) },
          body: JSON.stringify({
            model: env.XL_UPSTREAM_MODEL || DEFAULT_MODEL,
            messages,
            max_tokens: body.max_tokens || 400,
            temperature: body.temperature ?? 0.8,
            stream: false,
          }),
        });
        const rawText = await up.text();
        let j = null;
        try { j = JSON.parse(rawText); } catch (e) {
          lastErr = '上游返回非 JSON（' + up.status + '）：' + rawText.slice(0, 100);
          continue;
        }
        text = (j?.choices?.[0]?.message?.content || '').trim();
        if (text) break;
        lastErr = j?.error?.message ? String(j.error.message) : ('上游 HTTP ' + up.status + ' 无内容');
      } catch (e) {
        lastErr = String(e.message || e);
      }
      if (attempt < 2) await new Promise(r => setTimeout(r, 1200));
    }
    if (!text) {
      return json({ error: { type: 'upstream_error', message: lastErr || '上游没有返回内容' } }, 502);
    }

    if (path === '/v1/messages' || path === '/messages') {
      return json({
        id: 'msg_' + crypto.randomUUID().replace(/-/g, '').slice(0, 20),
        type: 'message', role: 'assistant', model: MODEL_NAME,
        content: [{ type: 'text', text }],
        stop_reason: 'end_turn',
        usage: { input_tokens: messages.length, output_tokens: text.length },
      });
    }

    return json({
      id: 'chatcmpl-' + crypto.randomUUID().replace(/-/g, '').slice(0, 20),
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: MODEL_NAME,
      choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: text } }],
      usage: { prompt_tokens: messages.length, completion_tokens: text.length,
               total_tokens: messages.length + text.length },
      provider: 'cloudflare-worker',
    });
  },
};
