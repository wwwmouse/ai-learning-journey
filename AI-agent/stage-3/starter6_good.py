"""練習 6：Schema 設計 — Good schema（Path A、Ollama 默認）。

【这个文件在做什么】
    与 starter6_bad.py 组成**对照组**：同一道题、同一套程序，**只改 TOOLS_SPEC**。
    本脚本只做一次询问，拿到模型的点单后直接退出分析点单工具，不存在agent_loop
    这一版按五条规则把 schema 写"好"：
      · description 说清"何时用 + 何时不要用"
      · 参数用正确类型（number / array，而不是一律 string）
      · required 列全
      · enum 收敛取值（unit 只能 celsius / fahrenheit）
      · additionalProperties: false 锁死字段

【链路】__main__ → select_and_run(question)
                        ├─ create(tools=TOOLS_SPEC)   ← 唯一一次模型调用
                        └─ execute_tool → _validate_args → 工具实现
【状态】无
【出口】3 个：没有 tool_calls / 多单拒绝 / 正常执行

【和 bad 版的关系 —— 控制变量】
    两个文件 90% 相同（工具实现、验证、选择器都一模一样），
    差异**全部集中在 TOOLS_SPEC**。这样真机上的行为变化，才能归因到 schema。

【跑法】（$0，走本机 Ollama）
    ollama serve
    python starter6_good.py                   # 真机：预期选对 convert_temperature
    python test6_schema_design.py             # 契约测试，不调模型
    对照：python starter6_bad.py
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


# 【⚪ 可跳过】工具实现：与 bad 版不同，这里的签名是真实的类型（list[dict] + operation）。
# 但工具怎么实现不是本练习的重点 —— 重点是模型**能不能从 schema 看出该用哪个**。
def process_data(data: list[dict], operation: str) -> dict:
    if operation == "count_rows":
        return {"rows": len(data)}
    if operation == "list_columns":
        return {"columns": sorted({key for row in data for key in row})}
    return {"error": "unknown operation", "retry_hint": "use count_rows or list_columns"}


# 【🟡 看关键】温度换算工具 —— 计算逻辑不值得看（就是公式），
# 值得看的是失败时**返回结构化 dict + retry_hint**（与练习 5 一致的做法）。
def convert_temperature(value: float, unit: str) -> dict:
    if unit == "celsius":
        return {"value": round(value * 9 / 5 + 32, 2), "unit": "fahrenheit"}
    if unit == "fahrenheit":
        return {"value": round((value - 32) * 5 / 9, 2), "unit": "celsius"}
    return {"error": "unsupported unit", "retry_hint": "unit must be celsius or fahrenheit"}


# 【⭐ 核心 —— 本练习唯一要研究的东西】
# 好在哪里，逐条对照 bad 版：
#   ① description 写明"何时用 / 何时不要用"（process_data 那句 "Do not use for ..."）
#   ② value 是 number，data 是 array of object —— 类型即约束
#   ③ required 列全（value + unit / data + operation）
#   ④ enum 收敛 unit 与 operation 的取值空间
#   ⑤ additionalProperties: false 禁止多余字段
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "process_data",
            "description": "仅用于汇总结构化 JSON 表格行；不要用于温度换算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "array", "items": {"type": "object"}, "description": "Rows to inspect"},
                    "operation": {"type": "string", "enum": ["count_rows", "list_columns"]},
                },
                "required": ["data", "operation"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_temperature",
            "description": "当用户要求在华氏度与摄氏度之间换算温度时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "需要换算的温度数值"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "输入数值所使用的温度单位"},
                },
                "required": ["value", "unit"],
                "additionalProperties": False,
            },
        },
    },
]


# 【⚪ 可跳过】allowlist：名字 → 执行函数。注意这里用 i["value"] 下标（字段已验证必存在）。
TOOL_IMPL = {
    "process_data": lambda i: process_data(i["data"], i["operation"]),
    "convert_temperature": lambda i: convert_temperature(i["value"], i["unit"]),
}


# 【🟢 必读】程序侧参数验证 —— 与 bad 版同样存在，且**跟着好 schema 的严格度走**：
#   process_data: data 必须是 list[dict]，operation 必须在枚举内
#   convert_temperature: value 必须是数字（且排除 bool），unit 必须在枚举内
# 注意 bool 检查：Python 里 True 是 int 的子类，不排除的话会被当成合法数字。
def _validate_args(name: str, args: dict) -> str | None:
    expected = {
        "process_data": {"data", "operation"},
        "convert_temperature": {"value", "unit"},
    }[name]
    if set(args) != expected:
        return "arguments contain unexpected or missing fields"
    if name == "process_data":
        if not isinstance(args["data"], list) or any(not isinstance(row, dict) for row in args["data"]):
            return "data must be a list of objects"
        if args["operation"] not in {"count_rows", "list_columns"}:
            return "operation must be count_rows or list_columns"
    else:
        if not isinstance(args["value"], (int, float)) or isinstance(args["value"], bool):
            return "value must be a number"
        if args["unit"] not in {"celsius", "fahrenheit"}:
            return "unit must be celsius or fahrenheit"
    return None


# 【🟢 必读】派发前的安全边界：schema 写得再好，这里也不能省。
# 未知工具 / 坏 JSON / 非对象 / 验证失败 → 全部返回结构化 error（不抛异常）。
def execute_tool(name: str, raw_arguments: str) -> tuple[dict, object]:
    """Validate model output even when the schema is clear."""
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


# 【🟢 必读】单轮选择器：与 bad 版**完全相同**（逐字一致），差异只在 TOOLS_SPEC。
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
    return {"tool": call.function.name, "tool_input": args, "observation": observation, "context": msg.content}


# 【⚪ 可跳过】真机自检：这一版断言更严 —— 好 schema 下预期选对且算对 89.6°F。
# ⚠️ 但注意：单次成功不能证明"好 schema 更好"，必须固定题目重跑多次。
if __name__ == "__main__":
    question = "将32摄氏度转化为华氏度"
    print(f"❓ 問題：{question}（using Ollama {MODEL}、GOOD schema）")
    result = select_and_run(question)
    print(f"   tool: {result['tool']}")
    print(f"   tool_input: {result.get('tool_input')}")
    print(f"   observation: {result['observation']}")
    print(f"   context(点单时通常为空): {result.get('context')}")

    assert result["tool"] == "convert_temperature", f"預期 convert_temperature、得到 {result['tool']}"
    assert result["observation"] == {"value": 89.6, "unit": "fahrenheit"}
    print("✅ Good schema starter 通過 — 這次模型在清楚的 schema 上挑對 tool、API 費 $0")
