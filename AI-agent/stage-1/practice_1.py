# 需要：pip install openai      （用 OpenAI-compatible SDK 与 Ollama 通信）
# 运行前：ollama pull gemma4:e4b && ollama serve
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Windows 终端默认编码可能不支持中文输出，这段强制用 UTF-8，防止打印中文时乱码或报错。

from openai import OpenAI
# openai 这个库是"帮打电话的助手"——它把 HTTP 请求、JSON 解析这些脏活全封装了。
# 它最初是 OpenAI 官方的 SDK，但因为接口成了行业标准。
# 现在 Ollama、DeepSeek、Kimi 全都"假装自己是 OpenAI"，所以使用同一个库。

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama 不检查这个占位值
)
# 告诉 SDK"去哪个地址打电话"。base_url = 服务器的门牌号；api_key = 门禁卡（Ollama 不检查，随便填占位）。
# 为什么这行是宝藏：以后想换 DeepSeek/OpenAI，只改这一行。

r = client.chat.completions.create(
    model="gemma4:e4b",   # 已安装时也可换成 qwen2.5:3b / llama3.2:3b
    max_tokens=512,   # gemma4:e4b 会先"思考"再回答，预算必须覆盖思考+回答（100 会被思考耗尽导致空响应）
    messages=[{"role": "user", "content": "用一句话自我介绍。"}],
)
# r (返回对象)
# ├── choices[0]          ← 通常只有一个候选回答
# │   ├── message.content  ← 模型的回答文字
# │   └── finish_reason    ← 为什么结束："stop"=自然说完 / "length"=被截断
# └── usage               ← "账单"
#     ├── prompt_tokens      （输入花了多少 token）
#     ├── completion_tokens  （输出花了多少 token）
#     └── total_tokens

# === 自我验证 ===
text = r.choices[0].message.content
print("响应：", text)
print("usage:", r.usage)

assert r.choices[0].finish_reason in ("stop", "length"), f"非预期 finish_reason: {r.choices[0].finish_reason}"
# 结束原因合法
assert len(text) > 0, "响应不应为空"
# 内容非空
assert r.usage.completion_tokens > 0, "output token 应大于 0"
# 确实生成答案

# assert 条件,"报错提示字符串"
# 若条件成立毫无影响
# 若不成立直接AssertionError 再给出后面的报错提示

print("✅ 练习 1 通过 — Ollama gemma4:e4b 已能在本地响应，每次 $0")

# 响应： 我是一个由Google DeepMind开发的Gemma 4大型语言模型，随时准备用文字为您提供信息、解答疑问和创意助力。
# usage: CompletionUsage(completion_tokens=459, prompt_tokens=22, total_tokens=481, completion_tokens_details=None, prompt_tokens_details=None)
# ✅ 练习 1 通过 — Ollama gemma4:e4b 已能在本地响应，每次 $0  