"""练习 5 的验收单 —— starter5.py 的契约测试（全程 mock，不调用模型）。

【这个文件在做什么】
    验证三件事：
      ① 失败计划（_failure_plan）按顺序消耗、不串味
      ② 结构化错误能被回灌，且"重试一次后成功"的路径走得通
      ③ 连续失败 / 未知工具 / 坏参数 / 截断 —— 都不 crash，各有明确归宿
    和 test2/test3 一样：不联网、$0、毫秒级、结果确定。

【和 test3 的关键差别】
    test3 的假响应里 obs 是**字符串**；
    test5 的假响应里 obs 是**结构化 dict**，断言要查 error / retry_hint / forecast 键。
    另外 set_weather_failures() 是**全局状态**——每个测试必须先设置它，
    否则测试之间会互相污染（这是全局可变状态的代价，真实项目里应尽量避免）。

【6 个测试各自钉住什么】
    1 test_fetch_weather_failure_plan                   失败计划按顺序消耗
    2 test_retry_then_success                           ⭐核心：错误 → 看 retry_hint
                                                        → 重试 → 成功 → 给出答案
    3 test_repeated_errors_can_end_gracefully           连续失败也能收尾，不 crash
    4 test_unknown_tool_returns_structured_error        未知工具 → 结构化 error 给模型看
    5 test_malformed_arguments_return_structured_error  坏 JSON / 空 city → 结构化 error
    6 test_length_is_not_a_final_answer                 截断不能当答案

【能证明什么、不能证明什么】
    能证明：错误是数据、循环不因错误崩溃、三种失败情境都有明确归宿。
    不能证明：真模型会不会**照着 retry_hint 重试**。mock 里"重试"是你写死的；
              真模型的反应只能真机重跑多次来评估。
    一个推论：这里验证的是"错误处理**机制**"，不是"模型**反应**"。

【跑法】必须在 stage-3 目录下运行（顶部 import 的是同目录的 starter5）
    python test5_error_handling.py
    注意：第 5、6 条没有 print，所以终端只有 4 行 ✅ —— 没有输出不等于没跑。

【怎么改它来学】
    · set_weather_failures([True, False]) 改成 [True, True] → 第 2 条会怎么失败？
    · 改成 [] → 第一次就成功，第 2 条还成立吗？
    · 把 fetch_weather 返回的 retry_hint 删掉 → 错误信息还够模型做决策吗？
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from starter5 import execute_tool, fetch_weather, react_loop, set_weather_failures


def make_tool_call(call_id: str, name: str, args: dict):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def make_resp(finish_reason: str, content: str = "", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=msg)])


def test_fetch_weather_failure_plan():
    set_weather_failures([True, False])
    first = fetch_weather("Taipei")
    second = fetch_weather("Taipei")
    assert first["error"] == "network timeout"
    assert second["forecast"] == "rain"
    print("✅ test_fetch_weather_failure_plan")


def test_retry_then_success():
    set_weather_failures([True, False])
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        make_resp("tool_calls", "Check weather.", [make_tool_call("t1", "fetch_weather", {"city": "Taipei"})]),
        make_resp("tool_calls", "Got retry_hint, retry once.", [make_tool_call("t2", "fetch_weather", {"city": "Taipei"})]),
        make_resp("stop", "Taipei is rainy now."),
    ]
    result = react_loop("Will it rain in Taipei?", client=client)
    tools = [entry["tool"] for entry in result["trace"] if entry["tool"]]
    assert tools == ["fetch_weather", "fetch_weather"]
    assert result["trace"][0]["obs"]["retry_hint"] == "try again in 1s"
    assert result["trace"][1]["obs"]["forecast"] == "rain"
    assert result["final"] == "Taipei is rainy now."
    print("✅ test_retry_then_success")


def test_repeated_errors_can_end_gracefully():
    set_weather_failures([True, True])
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        make_resp("tool_calls", "Try weather API.", [make_tool_call("t1", "fetch_weather", {"city": "Taipei"})]),
        make_resp("tool_calls", "Retry once.", [make_tool_call("t2", "fetch_weather", {"city": "Taipei"})]),
        make_resp("stop", "Weather is unavailable after retry; please try again later."),
    ]
    result = react_loop("Will it rain in Taipei?", client=client)
    assert result["trace"][0]["obs"]["error"] == "network timeout"
    assert result["trace"][1]["obs"]["error"] == "network timeout"
    assert "unavailable" in result["final"]
    print("✅ test_repeated_errors_can_end_gracefully")


def test_unknown_tool_returns_structured_error():
    """模擬 LLM 呼叫不存在的 tool、確認 loop 不 crash、而是把 error 給 LLM 看。"""
    set_weather_failures([])
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        make_resp("tool_calls", "I'll try a weird tool.", [make_tool_call("t1", "magic_tool", {"city": "Taipei"})]),
        make_resp("stop", "OK, no magic tool here."),
    ]
    result = react_loop("Try magic.", client=client)
    assert result["trace"][0]["obs"]["error"] == "tool not allowed"
    print("✅ test_unknown_tool_returns_structured_error")


def test_malformed_arguments_return_structured_error():
    _, malformed, malformed_error = execute_tool("fetch_weather", "{not-json")
    _, empty, empty_error = execute_tool("fetch_weather", json.dumps({"city": ""}))
    assert malformed_error is True and malformed["error"] == "invalid arguments"
    assert empty_error is True and empty["error"] == "invalid arguments"


def test_length_is_not_a_final_answer():
    client = MagicMock()
    client.chat.completions.create.return_value = make_resp("length", "unfinished")
    result = react_loop("too long", client=client)
    assert result["final"] is None
    assert result["terminal_reason"] == "length"


if __name__ == "__main__":
    test_fetch_weather_failure_plan()
    test_retry_then_success()
    test_repeated_errors_can_end_gracefully()
    test_unknown_tool_returns_structured_error()
    test_malformed_arguments_return_structured_error()
    test_length_is_not_a_final_answer()
    print("\n🎉 全部通過 — Ollama path tool error handling 邏輯正確")
