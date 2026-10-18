"""
練習 3：從零實作 ReAct（不用 framework）— starter3.py（Path A、Ollama 默認）

用一小段 Python 把「assistant 文字 → Tool Call → Tool Result」迴圈寫出來。
不要 LangChain、不要 LangGraph，就是純 while loop。

【这个文件在做什么】
    让模型自己决定要不要再查一次，直到它不再点单为止。
    循环骨架只有四步（见下方 react_loop）：
      ① 问模型（带上当前累积的 messages + tools）
      ② 把这一轮 assistant 消息回填进 messages
         （含 tool_calls；不填，下一轮的 tool 消息就没有订单可认领）
      ③ 没有 tool_calls → 收工；但要先确认 finish_reason == "stop" 才算真答案
      ④ 有 tool_calls   → 逐个执行、结果包成 role="tool" 消息回灌、继续下一轮
    跑满 max_iter 还停不下来 → 标记 truncated=True，绝不无限绕圈。

【和练习 1 / 练习 2 的关系】
    hello_tool.py   1 工具 / 固定 2 次调用 / 结果回灌 / 无循环   → 走"深"
    starter2.py     3 工具 / 只用 1 次    / 结果不回灌 / 无循环   → 走"广"
    starter3.py     2 工具 / 1~max_iter 次 / 结果回灌 / 有界循环 → 第一次满足
                   Stage 3 的定义：模型 + 工具 + 有界循环
    练习 1 的"回填 + 配对"、练习 2 的"三张表 + 验证"在这里原样复用；
    真正新增的只有循环控制、停止条件与 trace。

【三种退出方式】（课程的完成条件就要求测试能分别证明 A 和 C）
    A 正常收尾  没有 tool_calls 且 finish_reason == "stop"    → final = 答案
    B 异常终止  没有 tool_calls 但 finish_reason 是 length 等 → final = None + terminal_reason
    C 撞上限    跑满 max_iter 还在点单                         → final = None + truncated=True

【三个新概念】
    trace      每轮的 step / assistant_text / tool / tool_input / obs 列表。
               记"可验证摘要"即可，不要求模型公开私有思维链。
    truncated  是否撞上限或被截断——提醒调用方"这是半成品，别当答案"
    terminal_reason  非正常结束的原因（length / missing_tool_call 等）

【跑法】（$0，走本机 Ollama）
    ollama serve
    python starter3.py                      # 真机自检（多步推理题）
    python test3_react_from_scratch.py      # 契约测试，不调模型、毫秒级

"""

from __future__ import annotations

import ast
import json
import math
import os
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

MODEL = os.environ.get("MODEL", "qwen2.5:3b")


# === 1. Tools 定義（含實作）===
#
# 【阅读图例】本文件按"对 agent 学习的重要度"标注，优先级从高到低：
#   ⭐ 核心   —— 必须逐行吃透（只有 react_loop 一个）
#   🟢 必读   —— agent 机制所在（schema / allowlist / 参数验证）
#   🟡 看关键 —— 只看外层几行就能拿到要点（工具错误怎么处理）
#   ⚪ 可跳过 —— 业务或胶水代码，当黑盒即可（安全计算器、假数据、自检）
#
# 提醒：全文件约 300 行，其中一半以上属于 ⚪。跳过它们不是偷懒，是分层阅读。

MAX_EXPRESSION_LENGTH = 200
MAX_AST_NODES = 50
MAX_AST_DEPTH = 12
MAX_ABS_NUMBER = 1_000_000_000_000


# 【⚪ 可跳过】计算器的数值边界检查
# 作用：整数看绝对值；浮点数额外要求"有限"（挡掉 inf / nan）。
# 它只是 calculator 的辅助函数——与 agent 机制无关，换业务就没有这段代码。
def _within_bounds(value: int | float) -> bool:
    if isinstance(value, int):
        return abs(value) <= MAX_ABS_NUMBER
    return math.isfinite(value) and abs(value) <= MAX_ABS_NUMBER


# 【⚪ 可跳过 —— 整段黑盒】安全计算器的核心：AST 白名单
# 它做的事：把表达式解析成语法树，只放行数字常量、一元正负号、+ - * / 和括号，
#           其它节点（**、//、函数调用、属性访问）一律 raise；另有四道上限防 DoS。
# 为什么可以跳过：
#   · 这是"安全算术"的实践，不是 agent 的机制（换业务就是另一套代码）
#   · 你只需要记住对外契约：输入表达式字符串 → 返回数值，失败 raise ValueError
# 想追问"为什么不能用 eval()"可以看，但别为读懂它耽误练习 3 的核心。
def _evaluate_arithmetic(expression: str) -> int | float:
    """只計算小型數學式；模型文字永遠不會變成程式碼。"""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, TypeError):
        raise ValueError("calculator only accepts basic arithmetic") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("expression has too many parts")

    def visit(node: ast.AST, depth: int = 0) -> int | float:
        if depth > MAX_AST_DEPTH:
            raise ValueError("expression is too deeply nested")
        if isinstance(node, ast.Expression):
            return visit(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
            if not _within_bounds(value):
                raise ValueError("number is out of bounds")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth + 1)
            result = value if isinstance(node.op, ast.UAdd) else -value
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = visit(node.left, depth + 1)
            right = visit(node.right, depth + 1)
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("division by zero")
                result = left / right
            elif isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            else:
                result = left * right
        else:
            raise ValueError("calculator only accepts numbers and + - * /")
        if not _within_bounds(result):
            raise ValueError("result is out of bounds")
        return result

    return visit(tree)


# 【🟡 只看外层 5 行】对外的计算器工具
# ⭐ 值得看的正是这几行：工具内部的异常被"翻译"成 "error: ..." 字符串返回，
#    而不是抛出去炸掉循环——这是 agent 知识：工具失败要以"数据"形式回到模型手里。
# ⚪ 不值得看的是它调用的 _evaluate_arithmetic（上面那个黑盒）。
# 契约：str 进 → str 出；成功是数值文本，失败是以 "error:" 开头的文本。
def tool_calculator(expression: str) -> str:
    """安全的計算器：只允許 + - * / 跟數字。"""
    try:
        return str(_evaluate_arithmetic(expression))
    except (TypeError, ValueError) as exc:
        return f"error: {exc}"


# 【⚪ 可跳过】假的事实查询工具（教学用，不联网）
# 实现就是查一个写死的字典，查不到返回 "unknown: ..."。
# 唯一值得注意的一点：它同样遵守约定——工具永远返回字符串，失败也给可读的文本。
def tool_lookup_fact(query: str) -> str:
    """假的事實查詢（教學專用、避免依賴外部 API）。"""
    facts = {
        "台北人口": "2602000",
        "紐約人口": "8336000",
        "光速": "299792458",  # m/s
    }
    return facts.get(query.strip(), f"unknown: {query}")


# OpenAI-compatible 的 tools schema wrap 在 {"type":"function", "function":{...}} 裡
# 【🟢 必读】表①：给模型看的"工具说明书"（schema）
# 每条三要素：name（模型点单时引用）/ description（何时该用）/ parameters（怎么填）
# 这份 schema 直接决定模型选得准不准——练习 2 的观察对象、练习 6 的改造对象。
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "做基本算術運算（加減乘除）。輸入是表達式字串。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "算術表達式"},
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_fact",
            "description": "查詢一個事實（人口 / 物理常數等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查詢關鍵字"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]

# 【🟢 必读】表②：allowlist —— 名字 → 真正要执行的函数
# 派发只能查这张表；绝不能用 globals()[name]() 或 eval()，
# 否则等于把"执行哪段代码"的决定权交给模型。
# 这里的 lambda 是适配层：把通用的 args 字典翻译成具体函数需要的参数。
TOOL_IMPL = {
    "calculator": lambda inp: tool_calculator(inp["expression"]),
    "lookup_fact": lambda inp: tool_lookup_fact(inp["query"]),
}

# 【🟢 必读】表③：函数名字 → 参数该有的字段名，供 execute_tool 做参数验证
# 注意它和 TOOLS_SPEC 里的字段名是"手动同步"的：改了一处忘了另一处，
# 就会出现"模型被要求填 A、验证器期望 B"的错位。
TOOL_FIELDS = {
    "calculator": "expression",
    "lookup_fact": "query",
}


# 【🟢 必读】派发前的六道验证关卡 —— agent 的核心安全边界之一
#   ① 名字在 allowlist 里吗   ② arguments 是合法 JSON 吗   ③ 解析出来是 dict 吗
#   ④ 键正好是那一个吗        ⑤ 是非空字符串吗             ⑥ 才真正调用工具
# ⭐ 和练习 2 的版本相比，这里多了第三个返回值 is_error：
#    (args, observation, is_error) 三元组是练习 5「错误分流」的接口准备。
# ⚠️ 但目前 react_loop 用 "_" 把它丢掉了 —— is_error 已生产、尚未使用。
def execute_tool(name: str, raw_arguments: str) -> tuple[dict, str, bool]:
    """Validate an untrusted tool call before dispatch."""
    if name not in TOOL_IMPL:                                   # 名字不在白名单内
        return {}, f"error: tool not allowed: {name}", True
    try:                                                        # json object → Python dict
        args = json.loads(raw_arguments)                        
    except (TypeError, json.JSONDecodeError):                   # 兜底类型错误与json解析错误
        return {}, "error: arguments must be valid JSON", True
    if not isinstance(args, dict):                              # 确定args是dict类型
        return {}, "error: arguments must be a JSON object", True
    field = TOOL_FIELDS[name]                                   # 该函数对应的参数名
    if set(args) != {field}:                                    # 确定函数对应的参数名与自带的参数名一致
        return {}, f"error: arguments must contain exactly {field}", True
    if not isinstance(args[field], str) or not args[field].strip():    # 确定args内值是str形式且非空
        return {}, f"error: {field} must be a non-empty string", True
    try:
        observation = TOOL_IMPL[name](args)                     # 执行函数，返回结果
    except (KeyError, TypeError, ValueError) as exc:            # 兜底键值错误/类型错误
        return args, f"error: invalid arguments: {exc}", True
    return args, observation, observation.startswith("error:")


# === 2. ReAct loop ===

# 【⭐ 核心 —— 全文件唯一必须逐行吃透的函数】
# 这是"模型 + 工具 + 有界循环"的完整实现，也就是 Stage 3 的全部意义。
#
# 每轮四步：
#   ① 问模型（messages 已累积 + tools）
#   ② 回填这一轮的 assistant 消息（含 tool_calls；不填则下一轮 tool 消息无订单可认领）
#   ③ 没有 tool_calls → 收工；但只有 finish_reason == "stop" 才算真答案
#   ④ 有 tool_calls   → 逐个执行，结果包成 role="tool" 消息回灌，继续下一轮
#
# 三种退出（课程完成条件要求测试能证明 A 和 C）：
#   A 正常收尾 final = 答案
#   B 异常终止 finish_reason 非 stop（如 length）→ final = None + terminal_reason
#   C 撞上限   跑满 max_iter → final = None + truncated = True
#
# 读的时候盯住三处：
#   · messages 建在循环外 —— 它就是"记忆"本身
#   · 回填 assistant 在"判断停止之前" —— 保证给出答案的那一轮也进了历史
#   · 所有中止都是 return 数据，不抛异常 —— 由调用方决定怎么处理
def react_loop(question: str, max_iter: int = 6, client: Any = None) -> dict:
    """
    OpenAI-compatible ReAct loop。每輪：
      1. 問 LLM（含 tools）
      2. finish_reason: 'tool_calls' → 執行 tool、observation 接回、繼續
                       'stop' → 結束、最後 message 是答案
    回傳 {final, trace, steps}。
    """
    client = client or OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    messages = [{"role": "user", "content": question}]
    
    # 运行轨迹：记录事后能检查的东西
    # 作用：
    #     调试：模型为什么选了 lookup_fact 而不是 calculator？看 trace
    #     评估（eval）：统计失败模式、算成功率（练习 6 的结果卡就靠这种数据）
    #     可观测性：生产系统里就是日志/审计（Stage 7 的正式课题）
    #     展示：starter3.py 的 __main__ 就是靠遍历 trace 打印每一步
    trace: list[dict] = []

    for step in range(max_iter):
        resp = client.chat.completions.create(
            model=MODEL,
            tools=TOOLS_SPEC,
            messages=messages,
        )
        msg = resp.choices[0].message
        assistant_text = msg.content or ""
        tool_calls = msg.tool_calls or []

        # assistant_entry用于翻译格式
        # 使用目前信息构造一条完整的message，也可以直接用msg.model_dump()把pydantic对象自动转成需要的dict格式
        assistant_entry: dict = {"role": "assistant", "content": assistant_text}
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
        if not tool_calls: # 模型这轮没有点单 
            # 记录一条没有工具的轨迹
            trace.append({"step": step, "assistant_text": assistant_text, "tool": None, "obs": None})
            if finish_reason == "stop": # 正常结束
                return {"final": assistant_text, "trace": trace, "steps": step + 1}
            return { # 非正常结束
                "final": None,
                "trace": trace,
                "steps": step + 1,
                "terminal_reason": finish_reason or "missing_tool_call",
                "truncated": finish_reason == "length",
            }

        # 執行 tool calls、observation 接回（OpenAI 用 role="tool"）
        last_obs = ""
        for tc in tool_calls:
            args, obs, _ = execute_tool(tc.function.name, tc.function.arguments)
            last_obs = obs
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": obs,
            })

            trace.append({
                "step": step,                       # 第几轮
                "assistant_text": assistant_text,   # 说了什么
                "tool": tc.function.name,           # 点单的工具
                "tool_input": args,                 # 参数输入
                "obs": last_obs,                    # 拿到的结果
            })

    return {"final": None, "trace": trace, "steps": max_iter, "truncated": True}


# === 3. 自我驗證 ===

# 【⚪ 可跳过】真机自检：只有直接运行本文件时才执行
# 它跑一道多步推理题并把 trace 逐行打印——用来"看看循环长什么样"，不是核心逻辑。
# 被 test3 导入时不会触发（靠的就是这个 __main__ 守卫）。
if __name__ == "__main__":
    question = "'台北人口' 除以 '紐約人口'、答案保留 4 位小數。" # 可改成光速是多少进行测试
    print(f"❓ 問題：{question}（using Ollama {MODEL}）")
    print("-" * 60)

    my_step=int(input("请输入Agent Loop最大轮数:"))
    result = react_loop(question, max_iter=my_step)

    for entry in result["trace"]:
        print(f"[step {entry['step']}] assistant: {(entry['assistant_text'] or '')[:80]}...")
        if entry["tool"]:
            print(f"           tool: {entry['tool']}({entry.get('tool_input')}) → {entry['obs']}")
    print("-" * 60)
    print(f"✅ 最終答案：{result['final']}")
    print(f"   共 {result['steps']} 輪")

    # 寬鬆驗證（小 model 不一定精確到 4 位小數）
    assert result.get("final") is not None or result.get("truncated"), "loop 應收尾或顯式 truncate"
    assert result["steps"] <= my_step, "loop 不可超過 max_iter"
    print("✅ 練習 3 通過 — 你已用本機 qwen2.5:3b 跑通 ReAct + tool use、$0/run")
