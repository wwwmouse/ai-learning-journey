"""練習 5：Tool 錯誤處理 — Path A（Ollama 默認、本機免費）。

【这个文件在做什么】
    在 ReAct loop 上补一层「错误处理」：工具失败时不 raise、也不返回贫瘠的字符串，
    而是返回**结构化错误** {"error", "category", "retry_hint"}；
    再由程序按 category 分流，模型只处理该由它处理的那一类。

【责任分两层】
    · 程序层（策略）：transient 错误自动重试（有上限）；permanent 错误立即停止
    · 模型层（决策）：bad_input 错误回灌给它，由它改写参数再试
    错误对象用 category 区分这三类；循环不再丢弃 is_error。

【为什么错误要结构化】
    只回 "failed"，没人知道下一步该干什么；
    回 {"error":"network timeout","category":"transient","retry_hint":"try again in 1s"}：
      · 程序能判断"值得重试"           → 自动处理
      · 模型能知道"这是暂时的、怎么改"  → 需要时自己改

【三类错误：谁产生、怎么触发】
    category    由谁产生              默认运行   怎么触发（本项目）            真实系统来自
    bad_input   execute_tool（验证）  仅模型抽风时 测试里直接喂错输入           模型填错参数 / 调错工具
    transient   工具实现（fetch）     一定会    _failure_plan 第 1 项就是它     超时 / 429 / 连接失败
    permanent   工具实现（fetch）     不会       手动把 "permanent" 放进计划    401 / 403 / 404

    ⚠️ 关键区分：**_failure_plan 只控制工具层（fetch_weather）的失败。**
       bad_input 不经过计划——它由验证层在"模型给错参数 / 调错工具"时产生，
       所以它在默认运行里出不出现，取决于模型这一次有没有犯错（不确定）；
       而 transient 是 __main__ 写死的第 1 项失败，**每次运行必定触发**。

    ⚠️ permanent 分支是为真实系统预留的：此 demo 的默认计划里没有它。
       保留的目的是让"不可恢复错误"这条处理路径**能被测试覆盖**，
       属于有意的完整性覆盖，不是冗余代码（test5 用 ["permanent"] 手动造出来）。

【和练习 3 的关系】
    骨架相同（同一套循环、同样三个出口），差别有两处：
      ① observation 从 str 变成 dict（回灌时要 json.dumps）
      ② 执行工具改走 call_tool_with_retry —— 重试决策由程序接管

【跑法】（$0，走本机 Ollama）
    ollama serve
    python starter5.py                      # 真机自检：第一次必失败，程序自动重试
    python test5_error_handling.py          # 契约测试，不调模型、毫秒级
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

_failure_plan: list[bool | str] = [True, False]
# list[bool | str] 是类型注解（去掉不影响运行）：元素可以是 bool 或 str。
#   True / "transient" → 暂时性失败   "permanent" → 不可恢复失败   False → 成功
# ⚠️ str 这种写法**只会出现在测试里**（test5 用 ["permanent"] 造出永久错误）。
#    默认的 __main__ 只用 bool，所以正常运行永远不会出现字符串——
#    它存在的意义是"让不常见的分支也能被测试覆盖"。

# 【⚪ 可跳过】失败夹具：用一个全局列表控制"第几次调用会失败"
#   True / "transient" → 暂时性失败；"permanent" → 不可恢复失败；False → 成功
#   [True, False] → 第 1 次失败、第 2 次成功；[] → 一直成功
#   此函数的意义就是给_failure_plan全局变量一个可修改的接口
def set_weather_failures(plan: list[bool]) -> None:
    global _failure_plan # 使得后续赋值替换模块级变量
    _failure_plan = list(plan)


# 【🟡 看关键】假 weather API —— 值得看的是它**返回什么**，不是它怎么实现
# ⭐ 失败时 return 一个 dict {"error":..., "retry_hint":...}，而**不是 raise**
#    —— 这就是本练习的全部要点：错误是数据，不是异常。
def fetch_weather(city: str) -> dict:
    """假的 weather API、用 _failure_plan 決定這次該失敗還是成功。"""
    # 从计划里取一条：True/"transient"=暂时性失败，"permanent"=不可恢复失败，False=成功
    plan = _failure_plan.pop(0) if _failure_plan else False
    if plan is True or plan == "transient":
        return {"error": "network timeout", "category": "transient", "retry_hint": "try again in 1s"}
    # permanent 分支：本项目默认运行不会走到（只有test5会把 "permanent" 放进计划），
    # 保留它是为了让"不可恢复错误"这条路径可被验证。
    if plan == "permanent":
        return {"error": "unauthorized", "category": "permanent", "retry_hint": ""}
    # 写死了预测为rain 温度24
    return {"city": city, "forecast": "rain", "temperature_c": 24}


# 【🟢 必读】表①：工具说明书 —— description 里写了一句行为提示：
#   "If an error is returned, inspect retry_hint before retrying."
#   它主要帮模型处理 bad_input（transient 已由程序自动重试，模型通常看不到）。
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
#   未知工具 / 坏 JSON / 参数不对 → {"error", "category": "bad_input", "retry_hint"}
#   返回三元组 (args, observation_dict, is_error)：
#   · observation 是 **dict**（不是字符串）——所以回灌给模型时要 json.dumps
#   · is_error 由 `"error" in observation` 判断，交给 call_tool_with_retry 做分流
def execute_tool(name: str, raw_arguments: str) -> tuple[dict, dict, bool]:
    """Validate an untrusted call and always return a structured result."""
    if name != "fetch_weather": # 检查调用函数名字
        return {}, {"error": "tool not allowed", "category": "bad_input", "retry_hint": "choose fetch_weather"}, True
    try:                        # 尝试json object->python dict
        args = json.loads(raw_arguments) 
    except (TypeError, json.JSONDecodeError):   
        return {}, {"error": "invalid arguments", "category": "bad_input", "retry_hint": "send a JSON object with city"}, True
    if (
        not isinstance(args, dict) # 确认类型是dict
        or set(args) != {"city"}   # dict中参数名key必须是TOOLS_SPEC中指定的city
        or not isinstance(args.get("city"), str) # 参数值是str类型
        or not args["city"].strip() # 非空
    ):
        return args if isinstance(args, dict) else {}, {"error": "invalid arguments", "category": "bad_input", "retry_hint": "city must be a non-empty string"}, True
    # 拿到fetch_weaather里写死的天气预测和温度
    observation = fetch_weather(args["city"].strip())
    return args, observation, "error" in observation


# 【🟢 必读】程序侧的重试策略 —— **策略与机制分离**
#   机制：execute_tool 只负责"验证 + 执行一次"
#   策略：本函数决定"这个错误值不值得再试一次"
#     · transient（暂时性）→ 自动重试，最多 MAX_TRANSIENT_RETRIES 次
#     · bad_input（模型填错）→ 不重试，回灌给模型自己改
#     · permanent（不可恢复）→ 不重试，由 loop 立即停止
#       （本项目默认运行产生不了它；测试用 ["permanent"] 覆盖这条路径）
# 返回 (args, obs, is_error, attempts)：attempts 是实际尝试次数（1 = 没重试）
MAX_TRANSIENT_RETRIES = 2


def call_tool_with_retry(
    name: str,
    raw_arguments: str,
    max_retries: int = MAX_TRANSIENT_RETRIES,
) -> tuple[dict, dict, bool, int]:
    """在 execute_tool 外套一层重试策略：只对 transient 错误重试。"""
    attempts = 0
    while True:
        args, obs, is_error = execute_tool(name, raw_arguments)
        attempts += 1
        if not is_error:
            return args, obs, False, attempts
        # 只有"暂时性"错误才值得重试；其余立即返回，交给上层决定（回灌或停止）
        if obs.get("category") != "transient":
            return args, obs, True, attempts
        if attempts > max_retries:
            return args, obs, True, attempts


# 【⭐ 核心】骨架与 starter3 相同，**两处不同**：
#   ① tool 消息的 content 变成 json.dumps(obs)  ← 因为 obs 现在是 dict
#   ② 执行工具改走 call_tool_with_retry：程序先按 category 分流，再决定回灌还是停止
# 每轮流程：问模型 → 回填 assistant → 没有 tool_calls 就收尾（要 stop 才算答案）
#   → 有就执行（含程序侧重试）→ 序列化回灌 → 继续；跑满 max_iter 就 truncated。
# 除 A/B/C 三个出口外，新增第四个：permanent 工具错误 → terminal_reason="permanent_tool_error"
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
        
        # 把pydantic对象自动转成需要的dict格式
        assistant_entry: dict = {"role": "assistant", "content": text}
        # 如果模型点单了
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        # 把新message加到messages内
        messages.append(assistant_entry)

        finish_reason = resp.choices[0].finish_reason
        # 如果模型没点单
        if not tool_calls:
            # 记录一条没有工具的轨迹
            trace.append({"step": step, "assistant_text": text, "tool": None, "obs": None})
            # 正常结束退出
            if finish_reason == "stop":
                return {"final": text, "trace": trace, "steps": step + 1}
            return {
                "final": None,
                "trace": trace,
                "steps": step + 1,
                "terminal_reason": finish_reason or "missing_tool_call",
                "truncated": finish_reason == "length", # 是否长度过大导致截断
            }

        for tc in tool_calls:
            # 程序侧策略：transient 自动重试；bad_input 交给模型；permanent 立即停
            args, obs, is_error, attempts = call_tool_with_retry(
                tc.function.name, tc.function.arguments
            )
            # OpenAI-compat 的 tool message content 接受字串、把 dict 序列化
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(obs, ensure_ascii=False),
            })
            trace.append({
                "step": step,
                "assistant_text": text,
                "tool": tc.function.name,
                "tool_input": args,
                "obs": obs,
                "attempts": attempts,   # 实际尝试次数：>1 说明程序重试过
                "is_error": is_error,
            })
            # 不可恢复错误：不浪费剩余轮次，立即停止
            # （本项目默认运行走不到这里；仅测试覆盖）
            if is_error and obs.get("category") == "permanent":
                return {
                    "final": None,
                    "trace": trace,
                    "steps": step + 1,
                    "terminal_reason": "permanent_tool_error",
                }

    # 循环到上限导致截断
    return {"final": None, "trace": trace, "steps": max_iter, "truncated": True} 


# 【⚪ 可跳过】真机自检：只有直接运行本文件时才执行；被 test5 导入时不触发。
# 它做的事：让第一次调用必失败，再打印 trace —— 用来"看见程序偷偷重试了一次"。
if __name__ == "__main__":
    # 第一次调用 transient 失败 → 程序自动重试一次 → 第二次成功（不需要模型决定）
    set_weather_failures([True, False])
    print(f"❓ 問題：Will it rain in Taipei today?（using Ollama {MODEL}）")
    print("-" * 60)

    result = react_loop("Will it rain in Taipei today?")
    for entry in result["trace"]:
        print(f"{entry}")
    print("-" * 60)
    print(f"✅ 最終答案：{result['final']}")

    # 验证"程序侧重试"真的发生了：至少有一次工具调用尝试了不止一次
    saw_retry = any(entry.get("attempts", 1) > 1 for entry in result["trace"])
    assert saw_retry, "預期至少有一次工具調用被程序自動重試"
    assert result["steps"] <= 5, "loop 不可超過 max_iter"
    print("✅ 練習 5 通過 — 程式自己重試 transient 錯誤、不需要模型聽話、$0/run")
