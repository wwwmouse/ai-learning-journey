"""練習 5：Tool 錯誤處理 — Path A（Ollama 默認、本機免費）。

故意讓 fetch_weather 第一次回結構化 error、第二次才成功。觀察 ReAct loop 怎麼
把 error observation 交回 LLM、讓模型自己決定 retry / 改 query / 放棄。
重點：error 是結構化 dict（`{"error", "retry_hint"}`）、不是 Python exception。

【这个文件在做什么】
    同一套 ReAct loop，只把「工具失败时返回什么」换掉了：
      练习 3  失败 → 返回字符串 "error: ..."
      练习 5  失败 → 返回结构化 dict {"error": ..., "retry_hint": ...}
    然后观察：模型看到这条数据后，会不会照着 retry_hint 重试？

    ⚠️ 关键认知：**分流不是程序做的，是模型做的。**
    程序只负责两件事：① 把错误变成数据  ② 保证循环有界（max_iter）。
    （所以你会看到 execute_tool 的 is_error 仍被 "_" 丢弃——本练习不靠程序分流。）

【为什么错误要结构化】
    只回 "failed"，模型不知道下一步该干什么；
    回 {"error":"network timeout", "retry_hint":"try again in 1s"}，
    模型才能判断"这是暂时的、值得重试一次"。**retry_hint 就是给模型的决策依据。**

【三种失败情境】由 _failure_plan 控制
    set_weather_failures([True, False])  第一次失败、第二次成功 → 观察 retry
    set_weather_failures([True, True])   连续失败 → 观察 graceful end
    set_weather_failures([])             直接成功

【和前后练习的关系】
    starter3.py  错误是字符串 + is_error 被丢弃 → 只练「循环 + 回灌」
    starter5.py  错误是结构化 dict（带 retry_hint）→ 练「模型基于错误做决策」
    两者的 react_loop 骨架 90% 相同——差别就在"错误长什么样"，
    以及 TOOLS_SPEC 里多写了一句给模型的行为提示。

【跑法】（$0，走本机 Ollama）
    ollama serve
    python starter5.py                      # 真机自检（第一次必失败）
    python test5_error_handling.py          # 契约测试，不调模型、毫秒级

⚠️ 不同模型对 retry_hint 的反应会变：可能无视 hint、重复同一个错、或直接放弃。
   固定失败情境 + 重跑多次，观察才有意义。
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
# 【阅读图例】⭐ 核心（逐行吃透）｜🟢 必读（agent 机制）｜🟡 看关键（只看外层）｜⚪ 可跳过（业务/胶水）

# 【⚪ 可跳过】失败夹具：用一个全局列表控制"第几次调用会失败"
#   [True, False] → 第 1 次失败、第 2 次成功；[] → 一直成功
# ⚠️ 它是**全局可变状态**：每个测试都必须先 set_weather_failures()，否则互相污染。
_failure_plan: list[bool] = [True, False]


def set_weather_failures(plan: list[bool]) -> None:
    global _failure_plan
    _failure_plan = list(plan)


# 【🟡 看关键】假 weather API —— 值得看的是它**返回什么**，不是它怎么实现
# ⭐ 失败时 return 一个 dict {"error":..., "retry_hint":...}，而**不是 raise**
#    —— 这就是本练习的全部要点：错误是数据，不是异常。
def fetch_weather(city: str) -> dict:
    """假的 weather API、用 _failure_plan 決定這次該失敗還是成功。"""
    should_fail = _failure_plan.pop(0) if _failure_plan else False
    if should_fail:
        return {"error": "network timeout", "retry_hint": "try again in 1s"}
    return {"city": city, "forecast": "rain", "temperature_c": 24}


# 【🟢 必读】表①：工具说明书 —— 注意 description 里写了一句行为提示：
#   "If an error is returned, inspect retry_hint before retrying."
#   这是**写给模型看的指令**：告诉它遇到错误时先看 retry_hint 再决定下一步。
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "fetch_weather",
            "description": "Fetch current weather. If an error is returned, inspect retry_hint before retrying.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]


# 【🟢 必读】派发 + 验证，并**把所有失败都变成结构化 dict**（不抛异常、也不返回字符串）
#   未知工具 / 坏 JSON / 参数不对 → {"error": ..., "retry_hint": ...}
# 返回三元组 (args, observation_dict, is_error)：
#   · observation 是 **dict**（不是字符串）——所以回灌给模型时要 json.dumps
#   · is_error 由 `"error" in observation` 判断（注意：本练习还没用它做分流）
def execute_tool(name: str, raw_arguments: str) -> tuple[dict, dict, bool]:
    """Validate an untrusted call and always return a structured result."""
    if name != "fetch_weather":
        return {}, {"error": "tool not allowed", "retry_hint": "choose fetch_weather"}, True
    try:
        args = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError):
        return {}, {"error": "invalid arguments", "retry_hint": "send a JSON object with city"}, True
    if (
        not isinstance(args, dict)
        or set(args) != {"city"}
        or not isinstance(args.get("city"), str)
        or not args["city"].strip()
    ):
        return args if isinstance(args, dict) else {}, {"error": "invalid arguments", "retry_hint": "city must be a non-empty string"}, True
    observation = fetch_weather(args["city"].strip())
    return args, observation, "error" in observation


# 【⭐ 核心】和 starter3 的 react_loop 骨架几乎一模一样，**只改了一处**：
#     tool 消息的 content 变成 json.dumps(obs)   ← 因为 obs 现在是 dict
# 每轮流程不变：
#   问模型 → 回填 assistant → 没有 tool_calls 就收尾（要 stop 才算答案）
#   → 有就执行、把结果序列化回灌、继续；跑满 max_iter 就 truncated。
# 注意 is_error 在这里被 "_" 丢掉了：本练习**不靠程序分流**，靠模型自己看 retry_hint。
def react_loop(question: str, max_iter: int = 5, client: Any = None) -> dict:
    """OpenAI-compat ReAct loop。tool error 是結構化 JSON、放進 tool_result 給 LLM 看。"""
    client = client or OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    messages = [{"role": "user", "content": question}]
    trace: list[dict] = []

    for step in range(max_iter):
        resp = client.chat.completions.create(
            model=MODEL,
            tools=TOOLS_SPEC,
            messages=messages,
        )
        msg = resp.choices[0].message
        text = msg.content or ""
        tool_calls = msg.tool_calls or []

        assistant_entry: dict = {"role": "assistant", "content": text}
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        messages.append(assistant_entry)

        finish_reason = resp.choices[0].finish_reason
        if not tool_calls:
            trace.append({"step": step, "assistant_text": text, "tool": None, "obs": None})
            if finish_reason == "stop":
                return {"final": text, "trace": trace, "steps": step + 1}
            return {
                "final": None,
                "trace": trace,
                "steps": step + 1,
                "terminal_reason": finish_reason or "missing_tool_call",
                "truncated": finish_reason == "length",
            }

        for tc in tool_calls:
            args, obs, _ = execute_tool(tc.function.name, tc.function.arguments)
            # OpenAI-compat 的 tool message content 接受字串、把 dict 序列化
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(obs, ensure_ascii=False),
            })
            trace.append({"step": step, "assistant_text": text, "tool": tc.function.name, "tool_input": args, "obs": obs})

    return {"final": None, "trace": trace, "steps": max_iter, "truncated": True}


# 【⚪ 可跳过】真机自检：只有直接运行本文件时才执行；被 test5 导入时不触发。
# 它做的事：让第一次调用必失败，然后把 trace 打印出来 —— 用来"看见错误长什么样"。
if __name__ == "__main__":
    set_weather_failures([True, False])  # 第一次失敗、第二次成功
    print(f"❓ 問題：Will it rain in Taipei today?（using Ollama {MODEL}）")
    print("-" * 60)

    result = react_loop("Will it rain in Taipei today?")
    for entry in result["trace"]:
        if entry["tool"]:
            print(f"[step {entry['step']}] tool: {entry['tool']}({entry.get('tool_input')}) → {entry['obs']}")
    print("-" * 60)
    print(f"✅ 最終答案：{result['final']}")

    # 寬鬆驗證：loop 至少要看到結構化 error 在 trace 裡（小 model 不一定 retry）
    saw_error = any(isinstance(e["obs"], dict) and "error" in e["obs"] for e in result["trace"])
    assert saw_error, "預期至少一輪 tool 回傳結構化 error"
    assert result["steps"] <= 5, "loop 不可超過 max_iter"
    print("✅ 練習 5 通過 — 你已用本機 qwen2.5:3b 看見 tool error 在 ReAct loop 裡是 data 不是 exception、$0/run")
