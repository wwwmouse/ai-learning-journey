# 需要：pip install openai
# 运行前：ollama pull gemma4:e4b && ollama serve
import sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# 测量 5 次 latency 与 output token
latencies = []
output_tokens = []
for _ in range(5):
    t0 = time.time()
    r = client.chat.completions.create(
        model="gemma4:e4b",
        max_tokens=800,
        messages=[{"role": "user", "content": "你好！请自我介绍一下。"}],
    )
    latencies.append(time.time() - t0)
    output_tokens.append(r.usage.completion_tokens)

# 统计
avg_latency = sum(latencies) / len(latencies)
out_tok_avg = sum(output_tokens) / len(output_tokens)  # 五次平均值
tps = out_tok_avg / avg_latency if avg_latency > 0 else 0

print(f"model: gemma4:e4b（本地）")
print(f"5 次 latency（秒）: min={min(latencies):.2f} max={max(latencies):.2f} mean={avg_latency:.2f}")
print(f"avg output: {out_tok_avg} tokens，约 {tps:.1f} tokens/sec")
print(f"\n1000 次成本：$0（本地），预计时长：{avg_latency * 1000 / 60:.1f} 分钟")

# === 自我验证 ===
assert avg_latency > 0, "latency 应大于 0"
assert out_tok_avg > 0, "output token 应大于 0"
print(f"\n✅ 练习 3 通过 — 本地 model 每次 $0，但运行 1000 次约需 {avg_latency * 1000 / 60:.0f} 分钟")
print("💡 对照 Path B Anthropic：请按实际 input/output usage 与官方费率估算 1000 次成本，再与本地等待时间比较。")

# model: gemma4:e4b（本地）
# 5 次 latency（秒）: min=8.78 max=26.43 mean=13.52
# avg output: 660.2 tokens，约 48.8 tokens/sec

# 1000 次成本：$0（本地），预计时长：225.4 分钟

# ✅ 练习 3 通过 — 本地 model 每次 $0，但运行 1000 次约需 225 分钟
# 💡 对照 Path B Anthropic：请按实际 input/output usage 与官方费率估算 1000 次成本，再与本地等待时间比较。