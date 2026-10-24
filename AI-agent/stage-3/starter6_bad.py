"""練習 6：Schema 設計 — Bad schema（Path A、Ollama 默認）。

【这个文件在做什么】
    故意写一个"坏 schema"，与 starter6_good.py 组成**对照组**：
    同一道题（把 32°C 换成 °F）、同一套程序，**只改 TOOLS_SPEC**。
    本脚本只做一次询问，拿到模型的点单后直接退出分析点单工具，不存在agent_loop
    观察：schema 写得含糊时，模型会不会选错工具？

    坏在四件事（对应 README 的对照表）：
      · description 太短（"Process data." / "Convert a value."）→ 模型看不出何时该用
      · 参数全是 string（value: string）→ 类型信息丢失
      · 没有 required → 模型不知道哪些必须填
      · 没有 enum / additionalProperties → unit 可以是任何值、可以塞多余字段

【链路】__main__ → select_and_run(question)
                        ├─ create(tools=TOOLS_SPEC)   ← 唯一一次模型调用
                        └─ execute_tool → _validate_args → 工具实现
【状态】无（模块级只有 TOOLS_SPEC / TOOL_IMPL 两张常量表，没有跨调用数据）
【出口】3 个：没有 tool_calls / 多单拒绝 / 正常执行

【⚠️ 关键：schema 坏了，程序侧验证也不能省】
    schema 是"给模型看的说明书"，不是"程序的保证"。
    所以 execute_tool 仍然要查 allowlist、验字段与类型 —— 这一层与 schema 质量无关。

【跑法】（$0，走本机 Ollama）
    ollama serve
    python starter6_bad.py                    # 真机：看它可能选错
    python test6_schema_design.py             # 契约测试，不调模型
    对照：python starter6_good.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

MODEL = os.environ.get("MODEL", "qwen2.5:3b")

# 【阅读图例】⭐ 核心（本练习要研究的对象）｜🟢 必读（程序侧机制）｜🟡 看关键（只看外层）｜⚪ 可跳过（业务/胶水）


# 【⚪ 可跳过】工具实现：故意做成"什么都接受、返回一句废话"
# 这样焦点才在 schema 上 —— 工具本身越无意义，越能凸显"选错工具"的代价。
def process_data(data: str = "") -> str:
    return f"processed generic data: {data}"


# 【⚪ 可跳过】同上。注意参数类型写成 str —— 这是坏 schema 的一部分。
def convert_temperature(value: str = "") -> str:
    return f"converted something from: {value}"


# 【⭐ 核心 —— 本练习唯一要研究的东西】
# Anti-pattern：description 太短、参数都用 string、没有 required、没有 enum、
# 没有 additionalProperties: false。四条全踩。
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "process_data",
            "description": "处理数据",
            "parameters": {
                "type": "object",
                "properties": {"data": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_temperature",
            "description": "转换数值",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}, "unit": {"type": "string"}},
            },
        },
    },
]


# 【⚪ 可跳过】allowlist：名字 → 执行函数。派发只能查表，不能 globals()[name]()。
TOOL_IMPL = {
    "process_data": lambda i: process_data(i.get("data", "")),
    "convert_temperature": lambda i: convert_temperature(i.get("value", "")),
}


# 【🟢 必读】程序侧参数验证 —— 与 schema 质量无关，必须独立存在
# 逐项：字段集合必须正好匹配、每个值是非空字符串、unit 只能 celsius/fahrenheit。
def _validate_args(name: str, args: dict) -> str | None:
    expected = {
        "process_data": {"data"},
        "convert_temperature": {"value", "unit"},
    }[name]
    if set(args) != expected:
        return "arguments contain unexpected or missing fields"
    for field, value in args.items():
        if not isinstance(value, str) or not value.strip():
            return f"{field} must be a non-empty string"
    if name == "convert_temperature" and args["unit"].strip().lower() not in {
        "celsius",
        "fahrenheit",
    }:
        return "unit must be celsius or fahrenheit"
    return None


# 【🟢 必读】派发前的安全边界：即使 schema 是坏的，这里也一步不能省。
# 未知工具 / 坏 JSON / 非对象 / 验证失败 → 全部返回结构化 error（不抛异常）。
def execute_tool(name: str, raw_arguments: str) -> tuple[dict, object]:
    """Even a deliberately bad schema still needs a safe application boundary."""
    if name not in TOOL_IMPL:
        return {}, {"error": "tool not allowed"}
    try:
        args = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError):
        return {}, {"error": "arguments must be valid JSON"}
    if not isinstance(args, dict):
        return {}, {"error": "arguments must be a JSON object"}
    validation_error = _validate_args(name, args)
    if validation_error:
        return args, {"error": validation_error}
    try:
        return args, TOOL_IMPL[name](args)
    except (KeyError, TypeError, ValueError) as exc:
        return args, {"error": f"invalid arguments: {exc}"}


# 【🟢 必读】单轮选择器：发一次请求 → 只接受恰好一个 tool_call → 执行 → 返回 dict。
# 两个对照组（bad/good）的这个函数**完全相同**，差异只在 TOOLS_SPEC。
def select_and_run(question: str, client: Any = None) -> dict:
    client = client or OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    resp = client.chat.completions.create(
        model=MODEL,
        tools=TOOLS_SPEC,
        messages=[{"role": "user", "content": question}],
    )
    msg = resp.choices[0].message
    tool_calls = msg.tool_calls or []
    if not tool_calls:
        return {"tool": None, "tool_input": {}, "observation": None}
    if len(tool_calls) != 1:
        return {"tool": None, "tool_input": {}, "observation": {"error": f"expected one tool call; received {len(tool_calls)}"}}
    call = tool_calls[0]
    args, observation = execute_tool(call.function.name, call.function.arguments)
    return {"tool": call.function.name, "tool_input": args, "observation": observation, 'context':msg.content}


# 【⚪ 可跳过】真机自检：刻意宽松 —— 坏 schema 不保证选对，只保证"能产出一次选择"。
if __name__ == "__main__":
    question = "Convert 32 Celsius to Fahrenheit."
    print(f"❓ 問題：{question}（using Ollama {MODEL}、BAD schema）")
    result = select_and_run(question)
    print(f"   tool: {result['tool']}")
    print(f"   observation: {result['observation']}")
    print(f"   context(点单时通常为空): {result.get('context')}")
    # 寬鬆驗證：bad schema 不保證選對 tool、但至少要產出一個 tool call
    assert result["tool"] is not None, "even bad schema should produce an observable selection"
    assert result["observation"] is not None, "selected tool should produce an observable result"
    if result["tool"] != "convert_temperature":
        print("⚠ 這次壞 schema 挑錯了。對照 starter6_good.py，並用固定題目重跑多次。")
    print("✅ Bad schema starter 跑通 — 對照 starter6_good.py")
