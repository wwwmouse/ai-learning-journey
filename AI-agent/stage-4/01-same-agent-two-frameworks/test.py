r"""Stage 4 練習 1 自我驗證 — Path A（Ollama starter.py，LangGraph 版）。

【在哪跑】VS Code 整合終端（Ctrl+` 開啟 / 或選單「終端機 → 新增終端」），
        終端類型必須是 PowerShell（VS Code 的預設那個；Windows PowerShell 5.1 或 PowerShell 7 都行）。
        先切到本檔案所在的練習資料夾，再執行下面的指令：
        cd E:\VScode\AI-learning\AI-agent\stage-4\01-same-agent-two-frameworks

【怎麼跑】本練習已建好 .venv = Python 3.11。
    .\.venv\Scripts\python.exe test.py

"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import AIMessage

from starter import build_graph, run, search


def test_search_tool_basic():
    assert "Taipei" in search.invoke({"query": "Taipei"})
    assert "no entry" in search.invoke({"query": "unknown_topic"})
    print("✅ test_search_tool_basic")


def test_run_with_mock_llm():
    """Mock LangChain LLM — agent 第一輪叫 search、第二輪收尾。"""
    llm = MagicMock()
    # 第一輪：tool_calls (search Taipei)
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {"query": "Taipei"}, "id": "call_1", "type": "tool_call"}],
    )
    # 第二輪：純文字 summary
    final_msg = AIMessage(content="Taipei is the capital of Taiwan, population ~2.6M.")
    llm.bind_tools.return_value = llm  # 鏈式：bind_tools 回自己
    llm.invoke.side_effect = [tool_call_msg, final_msg]

    result = run("summarize Taipei", llm=llm)
    assert "Taipei" in result["final"]
    assert result["steps"] >= 3  # user + tool_call + tool_result + final
    print("✅ test_run_with_mock_llm")


def test_run_skip_tool_when_known():
    """Mock：LLM 直接給答案、不呼叫 tool。"""
    llm = MagicMock()
    final_msg = AIMessage(content="Python is a programming language.", tool_calls=[])
    llm.bind_tools.return_value = llm
    llm.invoke.return_value = final_msg

    result = run("what is Python", llm=llm)
    assert "Python" in result["final"]
    assert result["steps"] == 2
    print("✅ test_run_skip_tool_when_known")


def test_reject_unknown_tool_name():
    """模型不能把未知名稱偷偷當成 search 執行。"""
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


if __name__ == "__main__":
    test_search_tool_basic()
    test_run_with_mock_llm()
    test_run_skip_tool_when_known()
    test_reject_unknown_tool_name()
    print("all pass")
