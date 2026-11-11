r"""Stage 4 练习 4：比较有界 CodeAct 与安全的 JSON tool 算术路径。

JSON tool 只允许固定工具与固定参数；算式先转成 AST，再逐节点检查。
CodeAct 会执行模型产生的 Python，所以只在受限制的 Docker 容器里示范。
Jupyter 控制端口只绑到主机的 127.0.0.1；容器仍可能连网，因此这不是 production
sandbox。两条路都设有步数与输入边界。

【这个文件在做什么】（链路 / 状态 / 出口）
    链路（离线部分）：evaluate_arithmetic（AST 白名单求值）
                      run_json_tool_call（JSON tool 边界的校验）
                      codeact_executor_config（Docker 沙箱配置）
    链路（真机部分，需要 Docker）：build_codeact_agent → agent.run(question)
                      → 模型写 Python 代码 → DockerExecutor 在容器里执行 → 最多 4 步
    状态：没有图/没有显式 state；「状态」是工具返回值与 agent 内部的步数。
    出口：run() 回传 {"final": str}；离线函数回 float / str。
    和前面练习的关系：
      - 练习 1/3 是「JSON tool」路线：模型回一个工具调用（结构化 JSON）；
      - 这题是「CodeAct」路线：模型直接写 Python 代码，由沙箱执行；
      - 两条路都要「边界」：JSON 路线靠参数校验，CodeAct 路线靠 Docker 沙箱。

【跑法】VS Code 集成终端（PowerShell）。
    离线验证（不需要 Docker，先跑这个）：
        cd E:\VScode\AI-learning\AI-agent\stage-4\04-codeact-vs-json-tool
        .\.venv\Scripts\python.exe test.py
    真机 CodeAct（需要 Docker daemon + Ollama）：
        ollama serve                                  # 另开一个终端
        启动 Docker Desktop（Linux 容器），确认 docker version 成功
        .\.venv\Scripts\python.exe test_docker_smoke.py   # 先验证沙箱通道
        .\.venv\Scripts\python.exe starter.py

【验证】离线、不花钱（test.py 不调用模型、不执行模型代码）：
    .\.venv\Scripts\python.exe test.py

【注释约定】SM: = Smolagents 的 API；↔ = 与前面练习的对照。
"""

from __future__ import annotations

import ast
import os
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# SM: CodeAgent = CodeAct 路线（模型写代码，执行器跑）；ToolCallingAgent = JSON tool 路线（对照组）。
#     OpenAIServerModel 用 OpenAI 兼容协议连本机 Ollama；tool 把普通函数注册成工具。
from smolagents import CodeAgent, OpenAIServerModel, ToolCallingAgent, tool

MODEL = os.environ.get("MODEL", "qwen2.5:3b")
OLLAMA_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434/v1")
MAX_AST_NODES = 40
MAX_NUMBER = 1_000_000


# ⭐ 核心 1：这是「安全求值」的核心 —— 不用 eval/exec，而是把表达式解析成 AST 后
#    只允许白名单节点（数字、正负号、+ - * /），其他一律拒绝。
#    三道上限：字符串长度 ≤120、AST 节点 ≤40、数值/结果 ≤1e6；除零也直接拒绝。
#    ↔ 练习 1/3 里「模型只能调工具」，这里是「模型产出的文本先被当成数据、不是命令」。
def evaluate_arithmetic(expression: str) -> float:
    """只计算小型算术 AST；模型文字不会交给 ``eval`` 或 ``exec``。"""
    if not isinstance(expression, str) or len(expression) > 120:
        raise ValueError("Expression must be a string of at most 120 characters.")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("Expression is too complex.")

    # 递归遍历 AST：只认识下面几类节点，遇到别的（属性访问、函数调用、导入…）就报错。
    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
            if abs(value) > MAX_NUMBER:
                raise ValueError("Number magnitude is too large.")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) if isinstance(node.op, ast.UAdd) else -visit(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if right == 0: raise ValueError("Division by zero is not allowed.")
            return left / right
        raise ValueError(f"Unsupported arithmetic syntax: {type(node).__name__}")

    value = visit(tree)
    if abs(value) > MAX_NUMBER:
        raise ValueError("Result magnitude is too large.")
    return value


# 🟢 必读：@tool 把函数注册成工具；docstring 是给模型看的说明书（schema 由签名 + docstring 生成）。
#    注意出错时回 "error: ..." 字符串而不是抛异常 —— 让模型能看到错误、自己修正再试。
@tool
def calculator(expression: str) -> str:
    """Return a guarded arithmetic result for a small expression.

    Args:
        expression: Digits, parentheses, and the +, -, *, / operators only.
    """
    try:
        return str(evaluate_arithmetic(expression))
    except (SyntaxError, ValueError) as error:
        return f"error: {error}"


# 🟡 看关键：固定离线事实表。JSON 路线（run_json_tool_call）和 CodeAct 路线共用它。
@tool
def lookup_fact(query: str) -> str:
    """Return one fixed offline fact from a tiny allowlist.

    Args:
        query: Either ``Taipei population`` or ``New York population``.
    """
    facts = {"taipei population": "2602000", "new york population": "8336000"}
    return facts.get(query.strip().lower(), f"unknown: {query}")


# ⭐ 核心 2：JSON tool 路线的「边界」——模型只能回 {"name": ..., "arguments": {...}}，
#    且工具名必须白名单（只有 calculator）、参数形状必须精确匹配。
#    对比：CodeAct 路线没有这个校验，它靠 Docker 沙箱兜底（见 codeact_executor_config）。
def run_json_tool_call(call: dict[str, Any]) -> str:
    """JSON tool 边界：工具名、参数形状与算式都要先通过检查。"""
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        raise ValueError("A JSON tool call needs exactly name and arguments.")
    if call["name"] != "calculator" or not isinstance(call["arguments"], dict) or set(call["arguments"]) != {"expression"}:
        raise ValueError("Only calculator(expression=...) is allowed.")
    return calculator.forward(expression=call["arguments"]["expression"])


# 🟢 必读：Docker 沙箱的「最小权限」配置。逐项看：
#    allow_pickle=False          → 不允许反序列化 pickle（防止恶意对象）
#    host="127.0.0.1"            → Jupyter 控制端口只监听本机，不暴露到局域网
#    network_mode="bridge"       → 默认桥接网络（注意：容器仍可连网，见文件头说明）
#    mem_limit="256m"            → 内存上限
#    pids_limit=64               → 进程数上限（防 fork 炸弹）
#    cap_drop=["ALL"]            → 丢掉全部 Linux capabilities
#    security_opt=["no-new-privileges"] → 禁止通过 setuid 之类提升权限
def codeact_executor_config() -> dict[str, Any]:
    """缩小示范容器权限，并把 Jupyter 控制端口只绑到本机。"""
    return {
        "allow_pickle": False,
        "host": "127.0.0.1",
        "container_run_kwargs": {
            "network_mode": "bridge",
            "mem_limit": "256m",
            "pids_limit": 64,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges"],
        },
    }


# 🟡 看关键：CodeAct 路线的主角。三个边界：
#    max_steps=4                    → 最多 4 步（小模型容易绕圈，有界才不会无限跑）
#    additional_authorized_imports=[] → 容器里不额外授权 import
#    executor_type="docker"         → 代码只在 Docker 容器里执行，不在主机执行
#    执行器参数就是上面的 codeact_executor_config()。
def build_codeact_agent(model: Any | None = None, executor_type: str = "docker") -> CodeAgent:
    """建立最多走四步、且默认只在 Docker 里执行代码的 agent。"""
    model = model or OpenAIServerModel(model_id=MODEL, api_base=OLLAMA_BASE, api_key="ollama")
    executor_kwargs = codeact_executor_config() if executor_type == "docker" else {}
    return CodeAgent(tools=[calculator, lookup_fact], model=model, max_steps=4, additional_authorized_imports=[], executor_type=executor_type, executor_kwargs=executor_kwargs)


# 🟡 看关键：同一个工具集合、同一个模型，换成 JSON tool 路线 → 对照实验的「另一种写法」。
#    ↔ 练习 1 的 LangGraph/CrewAI 就是这条路线（模型回结构化工具调用）。
def build_json_agent(model: Any) -> ToolCallingAgent:
    """建立同工具集合的 JSON tool 对照组。"""
    return ToolCallingAgent(tools=[calculator, lookup_fact], model=model, max_steps=4)


# 🟢 必读：finally 里 cleanup() —— CodeAct 会起容器/执行器，必须确保收尾，否则进程/容器残留。
def run(question: str, model: Any | None = None) -> dict[str, str]:
    agent = build_codeact_agent(model=model)
    try:
        return {"final": str(agent.run(question))}
    finally:
        agent.cleanup()


# ⚪ 可跳过：真机示范（需要 Docker）。离线验证用 test.py；沙箱通道用 test_docker_smoke.py。
if __name__ == "__main__":
    result = run("Find Taipei population, divide it by New York population, and report the ratio.")
    print(result["final"])
    assert result["final"]
    assert run_json_tool_call({"name": "calculator", "arguments": {"expression": "2 + 3 * 4"}}) == "14.0"
