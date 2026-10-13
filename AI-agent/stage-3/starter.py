"""練習 2：多工具選擇 — Path A（Ollama 默認、本機免費）。

横向铺开 3 个工具,讓本機 qwen2.5:3b 在 3 個 tool（web_search / calculator / calendar_lookup）裡選一個。
重點不是工具強不強，是觀察 schema 的 description / 參數 / required 如何引導模型選對。

跑法：
    pip install -r requirements.txt
    ollama pull qwen2.5:3b   # Stage 3+ tool-use 默認 model
    ollama serve             # 預設 port 11434
    python starter.py

驗證：
    python test.py   （用 mock、不打 API）

想看 Anthropic Claude 版本：
    python starter_anthropic.py   （需 ANTHROPIC_API_KEY；每次先保留 $0.05）
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

# 假工具 ①：网页搜索。不真的联网，只是把 query 拼进一句固定的话里。
# 它的存在是为了给模型一个"查外部信息"的选项，观察它什么时候去点这道菜。
# 返回值一定是字符串——因为工具结果要塞进 role="tool" 的 content，而 content 只能是字符串。
def web_search(query: str) -> str:
    return f"search result: {query} -> Anthropic tool use docs and examples"


# calculator 的四道上限：长度 / 节点数 / 嵌套深度 / 数值大小。
# 没有上限的话，1+1+1+... 或 10**1000 这类输入能把 CPU 拖死。
MAX_EXPRESSION_LENGTH = 200
MAX_AST_NODES = 50
MAX_AST_DEPTH = 12
MAX_ABS_NUMBER = 1_000_000_000_000


# 数值边界检查：整数看绝对值；浮点数额外要求"有限"，挡住 inf 和 nan。
# 名字以下划线开头是 Python 惯例，表示"内部函数，外部别直接调用"。
def _within_bounds(value: int | float) -> bool:
    if isinstance(value, int):
        return abs(value) <= MAX_ABS_NUMBER
    return math.isfinite(value) and abs(value) <= MAX_ABS_NUMBER


# 安全计算器的核心：把表达式解析成语法树（AST），再只放行白名单里的节点。

# 为什么不能用 eval()？因为 eval 会把模型输出的文字当代码执行，
# 一句 "__import__('os').system('...')" 就能在你的机器上跑起来。

# 白名单只认：数字常量、一元正负号、+ - * /、括号。
# 其它节点（**、//、函数调用、属性访问、变量名）撞到最后的 else 就 raise。

# 递归的 visit() 顺便兼任深度检查：每往里走一层 depth+1，超过 MAX_AST_DEPTH 就停。
def _evaluate_arithmetic(expression: str) -> int | float:
    """Evaluate only small numeric expressions; model text never becomes code."""
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


# 对外的计算器工具：把内部的异常"翻译"成字符串返回，而不是抛出去炸掉程序。
# 这是本文件错误策略的关键一步——
#   抛异常 = 程序崩；返回 "error: ..." = 调用方还有机会处理。
# （练习 5 会把这种字符串真正送回模型，让它自己换个写法重试。）
# 返回值必须是字符串：str(3.0) 得到 "3.0"，test.py 断言的就是这个。
def calculator(expression: str) -> str:
    try:
        return str(_evaluate_arithmetic(expression))
    except (TypeError, ValueError) as exc:
        return f"error: {exc}"


# 假工具 ③：查日程。同样不联网，查的是写死的表。
# date.strip() 是防御性写法——模型可能给出 " tomorrow " 这种带空格的参数。
# 查不到时返回 "no events found for ..."：这是"明确的、最小的语义错误"，
#   让模型知道"这个日期没东西"，而不是收到一个会让程序崩的异常。
def calendar_lookup(date: str) -> str:
    events = {
        "2026-05-13": "10:00 Stage 3 review, 15:00 agent study group",
        "tomorrow": "10:00 Stage 3 review, 15:00 agent study group",
    }
    return events.get(date.strip(), f"no events found for {date}")


# OpenAI-compat 的 tools schema 要包一層 {"type": "function", "function": {...}}
# Schema 工厂函数：三份工具说明书只差 名字/用途/字段名/字段说明，
# 用一个函数生成，避免复制三遍、改漏一处。
# 生成的是 OpenAI-compat 要求的形状：{"type":"function","function":{...}}。
# parameters 里三个要点：
#   type: "object"              —— 参数整体是一个对象
#   required: [field]           —— 哪个字段必填
#   additionalProperties: False —— 不许填菜单外的字段
# 但 schema 只是"给模型看的约束"，各家供应商遵守程度不同，
#   所以程序侧必须再验证一遍——就是下面 execute_tool 做的事。
def _wrap(name: str, description: str, field: str, field_description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {field: {"type": "string", "description": field_description}},
                "required": [field],
                "additionalProperties": False,
            },
        },
    }


# 表①：给模型看的"菜单"，整个列表发给 tools= 参数
TOOLS_SPEC = [
    _wrap("web_search", "Search current or external information not in the prompt.", "query", "Search query"),
    _wrap("calculator", "Evaluate basic arithmetic with +, -, *, /, and parentheses.", "expression", "Math expression"),
    _wrap("calendar_lookup", "Look up events for a specific date or relative day.", "date", "Date to inspect"),
]

# 表②：allowlist —— 名字 → 真正要执行的函数
# 派发只能查这张表，绝不能用 globals()[name]() 或 eval()：
# 那等于把"执行哪段代码"的决定权交给模型。
TOOL_IMPL = {
    "web_search": lambda args: web_search(args["query"]),
    "calculator": lambda args: calculator(args["expression"]),
    "calendar_lookup": lambda args: calendar_lookup(args["date"]),
}

# 表③：每个工具"该有的字段名"，供 execute_tool 做参数验证
TOOL_FIELDS = {
    "web_search": "query",
    "calculator": "expression",
    "calendar_lookup": "date",
}


# 派发前的验证关口。模型给的 tool call 是不可信输入，逐关检查：
#   ① 名字在 allowlist 里吗      → 不在就拒绝执行
#   ② arguments 是合法 JSON 吗   → 它本来是字符串，可能根本解析不了
#   ③ 解析出来是 dict 吗         → 也可能是数组或数字
#   ④ 字段正好是那一个吗         → 多一个少一个都拒绝，不做"猜测式修补"
#   ⑤ 是非空字符串吗             → 类型对、内容也不能是空白
#   ⑥ 才真正调用函数；执行中报错也转成字符串
# 设计要点：这个函数不往外抛异常，所有失败都以 "error: ..." 字符串返回。
# 返回 (args, observation)：args 是解析后的参数，observation 是要回给模型的结果。
# 小瑕疵：前 5 关失败时返回 {}，第 6 关失败时返回已解析的 args，返回值不太一致。
def execute_tool(name: str, raw_arguments: str) -> tuple[dict, str]:
    """Parse and validate model output before dispatching an allowlisted tool."""
    if name not in TOOL_IMPL:
        return {}, f"error: tool not allowed: {name}"
    try:
        args = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError):
        return {}, "error: arguments must be valid JSON"
    if not isinstance(args, dict):
        return {}, "error: arguments must be a JSON object"
    field = TOOL_FIELDS[name]
    if set(args) != {field}:
        return {}, f"error: arguments must contain exactly {field}"
    if not isinstance(args[field], str) or not args[field].strip():
        return {}, f"error: {field} must be a non-empty string"
    try:
        return args, TOOL_IMPL[name](args)
    except (KeyError, TypeError, ValueError) as exc:
        return args, f"error: invalid arguments: {exc}"


# === 2. 單輪 tool selection ===

# 本文件的主流程：单轮工具选择。
# 一次请求 → 看模型点了什么 → 执行一个 → 返回结果，就结束了。
# 三个分支：
#   ① 没点单     → tool=None，把模型的原话带回去
#   ② 点了多个单 → 整批拒绝（返回 error），不做"只执行第一个"这种部分执行
#   ③ 正好一个单 → 交给 execute_tool 验证并执行
# 最关键的一点：observation 在这里直接返回给调用者，并没有喂回模型。
#   所以这一轮少了 "tool result → model" 那半步，还不是完整的 Agent Loop。
# client 参数是为了可测试：test.py 会传 MagicMock 进来，这样不用真的调用模型。
# 已知盲点：没有检查 finish_reason。如果模型是因为被截断（length）才没点单，
#   这里会返回 tool=None，和"模型主动决定不用工具"长得一模一样。
def run_tool_selection(question: str, client: Any = None) -> dict:
    """單輪 call：LLM 看完 question + tools 後選一個 tool 呼叫，本地執行 observation 接回去。"""
    client = client or OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    resp = client.chat.completions.create(
        model=MODEL,
        tools=TOOLS_SPEC,
        messages=[{"role": "user", "content": question}],
    )
    msg = resp.choices[0].message
    text = msg.content or ""
    tool_calls = msg.tool_calls or []
    if not tool_calls:
        return {"tool": None, "assistant_text": text, "observation": None}
    if len(tool_calls) != 1:
        return {
            "tool": None,
            "assistant_text": text,
            "observation": f"error: expected exactly one tool call; received {len(tool_calls)}",
        }
    call = tool_calls[0]
    args, observation = execute_tool(call.function.name, call.function.arguments)
    return {
        "tool": call.function.name,
        "tool_input": args,
        "assistant_text": text,
        "observation": observation,
    }


# === 3. 自我驗證 ===

# 真机自检：只有直接运行本文件时才执行这段；
# 被 import 时不会触发——test.py 靠的就是这一点。
if __name__ == "__main__":
    question = "What is (19 * 42) - 8? Use the best available tool."
    print(f"❓ 問題：{question}（using Ollama {MODEL}）")
    result = run_tool_selection(question)
    print(f"   tool: {result['tool']}")
    print(f"   tool_input: {result.get('tool_input')}")
    print(f"   observation: {result['observation']}")

    assert result["tool"] == "calculator", f"預期 calculator、得到 {result['tool']}"
    assert result["observation"] and not result["observation"].startswith("error:")
    print("✅ 練習 2 通過 — 你已用本機 qwen2.5:3b 跑通 multi-tool selection、$0/run")
