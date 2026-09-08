# -*- coding: utf-8 -*-
"""
Stage 2 练习 2 最简版：固定六题，Zero-Shot 与 Few-Shot 各跑一轮，肉眼对比
"""
import sys
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# 固定六题（两轮都不换）
CASES = ["我被扣款两次", "发票上的金额不对", "按下登录后画面全白",
         "更新后一直闪退", "你们周末上班吗", "谢谢你帮我处理"]

# 三个范例（Few-Shot 轮才拼进 prompt）
EXAMPLES = """
范例：
输入：信用卡又扣了一次
输出：billing

输入：提交表单后没有反应
输出：bug

输入：可以更改联系邮箱吗
输出：other"""

for label, add_ex in [("Zero-Shot", ""), ("Few-Shot", EXAMPLES)]:
    print(f"===== {label} =====")
    for c in CASES:
        prompt = f"""目标：将客服留言分到 billing、bug 或 other。
资料：<input_data>{c}</input_data>
规则：只根据资料分类；不知道时选 other。
输出：只返回一个小写标签。{add_ex}"""
        r = client.chat.completions.create(
            model="gemma4:e4b",
            temperature=0,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        print(f"{c}  →  {r.choices[0].message.content}")
