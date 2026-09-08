# 需要：pip install openai     （用 OpenAI-compatible SDK 与 Ollama 通信）
# 运行前：ollama pull gemma4:e4b && ollama serve
import sys, statistics
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

PROMPTS = {
    "中文": "用一句话描述一只猫在做什么。",
    "English": "Describe in one sentence what a cat is doing.",
}

N = 10  # 本地运行较慢时先用小一点的 N，确认成功后再加大
for label, prompt in PROMPTS.items():
    output_tokens = []
    for _ in range(N):
        r = client.chat.completions.create(
            model="gemma4:e4b",
            max_tokens=800,
            temperature=1.0,  # 调高 temperature 以观察变化
            messages=[{"role": "user", "content": prompt}],
        )
        # print(f"{r.choices[0].message.content}")
        output_tokens.append(r.usage.completion_tokens)
    print(f"\n[{label}] prompt: {prompt}")
    print(f"  input tokens: {r.usage.prompt_tokens}")
    print(f"  output tokens — min={min(output_tokens)} max={max(output_tokens)} mean={statistics.mean(output_tokens):.1f} stdev={statistics.stdev(output_tokens):.1f}")

# === 自我验证 ===
assert len(output_tokens) == N and all(n > 0 for n in output_tokens), "每个 output token 数都应大于 0"
print("\n✅ 练习 2 通过 — 已观察到两种语言的 output token，本地运行 $0")
print("💡 token 数会受 tokenizer 和实际内容影响；不要只按字数推算，也不要预设某种语言一定较多。")

# [中文] prompt: 用一句话描述一只猫在做什么。
#   input tokens: 25
#   output tokens — min=15 max=764 mean=503.9 stdev=282.1

# [English] prompt: Describe in one sentence what a cat is doing.
#   input tokens: 26
#   output tokens — min=17 max=433 mean=234.3 stdev=159.4

# ✅ 练习 2 通过 — 已观察到两种语言的 output token，本地运行 $0
# 💡 token 数会受 tokenizer 和实际内容影响；不要只按字数推算，也不要预设某种语言一定较多。