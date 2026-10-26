r"""Stage 4 练习 1 自我验证 — Path A（Ollama starter.py，LangGraph 版）。

【跑法】VS Code 集成终端（PowerShell），先 cd 到本练习目录：
    cd E:\VScode\AI-learning\AI-agent\stage-4\01-same-agent-two-frameworks
    .\.venv\Scripts\python.exe test.py

【这份验收单钉住什么】（读测试时只问一句：它在防哪种改坏？）
    ① search 的输入输出：已知 key 回内容、未知 key 回 "no entry"，永不抛错
    ② 两条路径：模型点单 → agent→tools→agent→收尾（steps ≥ 3）；
                模型不点单 → 只跑 agent（steps 恰好 2，用等号钉死＝防多跑节点）
    ③ 安全契约：未知工具名必须抛 ValueError，不能被静默执行
    ④ mock 手法：bind_tools 回自己；invoke 用 side_effect 依序吐两条假消息

【它没覆盖的】（不是缺陷，是留给后面的练习与 Stage 7）
    - tool_node 的参数校验分支：args 不是 dict、多一个键、query 不是 str
    - 撞 recursion_limit 的截断路径（会抛 GraphRecursionError，不是回半成品）
    - search 的空白／大小写归一化（例如 "  TAIPEI  "）

用 mock LLM 取代真实 API、不打网路、不需要 Ollama、不花钱。
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import AIMessage

from starter import build_graph, run, search


# 🟢 契约：工具的输入输出形状——含「查不到」也必须有稳定输出，不能抛错。
def test_search_tool_basic():
    assert "Taipei" in search.invoke({"query": "Taipei"})
    assert "no entry" in search.invoke({"query": "unknown_topic"})
    print("✅ test_search_tool_basic")


# 🟢 契约：点单路径。steps ≥ 3 是「图真的走了 tools」的证据（不只跑了 agent）。
def test_run_with_mock_llm():
    """Mock LangChain LLM — agent 第一轮叫 search、第二轮收尾。"""
    llm = MagicMock()
    # 第一轮：tool_calls (search Taipei)
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {"query": "Taipei"}, "id": "call_1", "type": "tool_call"}],
    )
    # 第二轮：纯文字 summary
    final_msg = AIMessage(content="Taipei is the capital of Taiwan, population ~2.6M.")
    # mock 的关键两行：bind_tools 必须回自己（agent_node 用的是回传值），
    # invoke 用 side_effect 依序回两条消息 = 两轮。
    llm.bind_tools.return_value = llm
    llm.invoke.side_effect = [tool_call_msg, final_msg]

    result = run("summarize Taipei", llm=llm)
    assert "Taipei" in result["final"]
    assert result["steps"] >= 3  # user + tool_call + tool_result + final
    print("✅ test_run_with_mock_llm")


# 🟢 契约：不点单时不该多跑节点。steps == 2 用「等号」，就是为了钉死这件事。
def test_run_skip_tool_when_known():
    """Mock：LLM 直接给答案、不呼叫 tool。"""
    llm = MagicMock()
    final_msg = AIMessage(content="Python is a programming language.", tool_calls=[])
    llm.bind_tools.return_value = llm
    llm.invoke.return_value = final_msg

    result = run("what is Python", llm=llm)
    assert "Python" in result["final"]
    assert result["steps"] == 2
    print("✅ test_run_skip_tool_when_known")


# 🟢 安全契约：模型给的名字不可信，必须在执行前就被拒绝（allowlist 生效）。
def test_reject_unknown_tool_name():
    """模型不能把未知名称偷偷当成 search 执行。"""
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{"name": "delete_files", "args": {"query": "Taipei"}, "id": "call_bad", "type": "tool_call"}],
    )

    try:
        run("summarize Taipei", llm=llm)
    except ValueError as error:
        assert "Unknown tool" in str(error)
    else:
        raise AssertionError("unknown tool name must be rejected")
    print("✅ test_reject_unknown_tool_name")


# ⚪ 可跳过：手写 runner（没有 pytest 依赖），失败靠 assert 抛错。
if __name__ == "__main__":
    test_search_tool_basic()
    test_run_with_mock_llm()
    test_run_skip_tool_when_known()
    test_reject_unknown_tool_name()
    print("all pass")
