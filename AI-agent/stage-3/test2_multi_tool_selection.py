"""练习 2 的验收单 —— starter2.py 的契约测试（全程 mock，不调用模型）。

【这个文件在做什么】
    它不演示功能，它"验收"功能：把「starter2 应该怎样表现」写成断言。
    改完 starter2 后跑它，立刻知道有没有把原来的行为改坏——这就是回归测试。
    它便宜又可靠，靠的是三件事：
      · 不联网、不调模型   → 毫秒级、$0
      · 响应是自己造的     → 结果确定，不受模型随机性影响
      · 靠 __main__ 守卫导入 starter2 → 导入时不会触发真机自检

【8 个测试各自钉住什么】
    1 test_tool_spec_contains_three_tools        工具名与顺序、calculator 基本正确
    2 test_calculator_rejects_unsafe_expressions 8 种危险表达式全部拒绝
    3 test_llm_selects_calculator                选中计算器 → observation 为 "4.0"
    4 test_llm_selects_calendar                  选中日程 → 结果含 Stage 3 review
    5 test_llm_uses_search_instead_of_calendar_for_news  外部信息类走 web_search
    6 test_no_tool_call_returns_none             不点单 → tool=None、observation=None
    7 test_untrusted_arguments_are_rejected      6 种非法参数（坏 JSON / 缺字段 / 未知工具 /
                                                 多字段 / 类型错 / 空白值）全部拦下
    8 test_multiple_calls_are_not_silently_partially_executed  多单一律整批拒绝

【它能证明什么、不能证明什么】（重要）
    能证明：纯逻辑——计算器安全性、参数验证、各分支处理。完全不依赖模型。
    能证明：管道正确性——"假设模型选了 X，我的程序处理得对吗"。
    不能证明：模型会不会真的选对。因为工具名是 mock 指定的；
              模型的选择能力只能靠真机重跑多次来评估（练习 6 的结果卡）。
    一个推论：测试的覆盖面 = 你的程序实际承担的责任。这里只有 8 条，
              正说明 starter2 承担的责任很小（它只是个单轮选择器）。

【跑法】必须在 stage-3 目录下运行（底部 import 的是同目录的 starter2）
    python test2_multi_tool_selection.py              # 直接跑，零依赖
    注意：第 2、7、8 条没有 print，所以终端只有 5 行 ✅。
          没有输出不等于没跑——没有抛异常就是通过。

"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from starter2 import TOOLS_SPEC, calculator, execute_tool, run_tool_selection


# === Helpers for building mock OpenAI-compat responses ===

def make_tool_call(call_id: str, name: str, args: dict):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def make_resp(content: str, tool_calls=None, finish_reason: str = "tool_calls"):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=msg)])


def mock_client_for(name: str, args: dict) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = make_resp(
        f"I should use {name}.",
        [make_tool_call("call_1", name, args)],
    )
    return client


# === Tests ===

def test_tool_spec_contains_three_tools():
    names = [t["function"]["name"] for t in TOOLS_SPEC]
    assert names == ["web_search", "calculator", "calendar_lookup"]
    assert calculator("2 * (3 + 4)") == "14"
    print("✅ test_tool_spec_contains_three_tools")


def test_calculator_rejects_unsafe_expressions():
    expressions = [
        "2 ** 3", "7 // 2", "10 ** 1000", "1000000000001",
        "-" * 20 + "1", "1+" * 100 + "1",
        "__import__('os').system('ls')", "1 / 0",
    ]
    for expression in expressions:
        assert calculator(expression).startswith("error:"), expression


def test_llm_selects_calculator():
    client = mock_client_for("calculator", {"expression": "20 / 5"})
    result = run_tool_selection("What is 20 divided by 5?", client=client)
    assert result["tool"] == "calculator"
    assert result["observation"] == "4.0"
    print("✅ test_llm_selects_calculator")


def test_llm_selects_calendar():
    client = mock_client_for("calendar_lookup", {"date": "tomorrow"})
    result = run_tool_selection("Do I have anything tomorrow?", client=client)
    assert result["tool"] == "calendar_lookup"
    assert "Stage 3 review" in result["observation"]
    print("✅ test_llm_selects_calendar")


def test_llm_uses_search_instead_of_calendar_for_news():
    client = mock_client_for("web_search", {"query": "latest Claude tool use examples"})
    result = run_tool_selection("Find recent examples of Claude tool use.", client=client)
    assert result["tool"] == "web_search"
    assert "latest Claude tool use examples" in result["observation"]
    print("✅ test_llm_uses_search_instead_of_calendar_for_news")


def test_no_tool_call_returns_none():
    client = MagicMock()
    client.chat.completions.create.return_value = make_resp(
        "I don't think any of these tools applies.", tool_calls=None, finish_reason="stop",
    )
    result = run_tool_selection("Tell me a joke.", client=client)
    assert result["tool"] is None
    assert result["observation"] is None
    print("✅ test_no_tool_call_returns_none")


def test_untrusted_arguments_are_rejected():
    _, malformed = execute_tool("calculator", "{not-json")
    _, missing = execute_tool("calculator", "{}")
    _, unknown = execute_tool("delete_everything", "{}")
    _, extra = execute_tool("calculator", json.dumps({"expression": "2 + 2", "unexpected": "x"}))
    _, wrong_type = execute_tool("calculator", json.dumps({"expression": 4}))
    _, empty = execute_tool("calculator", json.dumps({"expression": "  "}))
    assert malformed.startswith("error:")
    assert missing.startswith("error:")
    assert "not allowed" in unknown
    assert extra.startswith("error:")
    assert wrong_type.startswith("error:")
    assert empty.startswith("error:")


def test_multiple_calls_are_not_silently_partially_executed():
    client = MagicMock()
    client.chat.completions.create.return_value = make_resp(
        "two calls",
        [
            make_tool_call("call_1", "calculator", {"expression": "2 + 2"}),
            make_tool_call("call_2", "calendar_lookup", {"date": "tomorrow"}),
        ],
    )
    result = run_tool_selection("Do both", client=client)
    assert result["tool"] is None
    assert "exactly one" in result["observation"]


if __name__ == "__main__":
    test_tool_spec_contains_three_tools()
    test_calculator_rejects_unsafe_expressions()
    test_llm_selects_calculator()
    test_llm_selects_calendar()
    test_llm_uses_search_instead_of_calendar_for_news()
    test_no_tool_call_returns_none()
    test_untrusted_arguments_are_rejected()
    test_multiple_calls_are_not_silently_partially_executed()
    print("\n🎉 全部通過 — Ollama path tool selection 邏輯正確")
