"""练习 5 的验收单 —— starter5.py 的契约测试（全程 mock，不调用模型）。

【这个文件在做什么】
    验证「错误分类 + 三层分流」这套机制：
      ① transient 错误 → 程序自动重试（有上限），不需要模型决定
      ② bad_input 错误 → 不重试，回灌给模型自己改
      ③ permanent 错误 → 不重试，loop 立即停止
      ④ 连续失败 / 坏参数 / 截断 —— 都不 crash，各有明确归宿
    和 test2/test3 一样：不联网、$0、毫秒级、结果确定。

【9 个测试各自钉住什么】
    1 test_fetch_weather_failure_plan              失败计划按顺序消耗（含 category）
    2 test_transient_retry_succeeds_within_quota   暂时性错误 → 自动重试后成功
    3 test_transient_retry_is_bounded              一直失败 → 尝试次数被上限卡住
    4 test_permanent_error_is_not_retried          不可恢复错误 → 一次都不重试
    5 test_bad_input_is_not_retried                模型填错 → 程序不重试（交给模型改）
    6 test_react_loop_retries_transient_then_answers  loop 里真的走了重试并拿到答案
    7 test_permanent_error_stops_the_loop          永久错误 → 第一轮就停（terminal_reason）
    8 test_malformed_arguments_return_structured_error  坏 JSON / 空 city → 结构化 error
    9 test_length_is_not_a_final_answer            截断不能当答案

【关于"本项目里不会自然出现"的分支】
    第 4、7 条测试是**手动造出** permanent 错误的（set_weather_failures(["permanent"])），
    因为默认运行里 fetch_weather 只会产生 transient。
    第 5、8 条则直接构造 bad_input：前者传一个不存在的工具名，后者传坏 JSON。
    注意 bad_input **不经过 _failure_plan**——它由 execute_tool 的验证层产生，
    所以只能靠"喂错输入"来覆盖。
    这类"造出来才能测"的分支是有意保留的：真实系统里 401/403 很常见，
    本地 demo 只是没有对应场景。测试负责保证这条路径不会写坏。

【能证明什么、不能证明什么】
    能证明：错误分类、重试配额、permanent 停止——全部是确定性的程序行为。
    不能证明：真模型遇到 transient 错误会不会配合。**但现在已经不需要它配合**，
              这就是这次改动的意义。

【跑法】必须在 stage-3 目录下运行（顶部 import 的是同目录的 starter5）
    python test5_error_handling.py
    注意：第 8、9 条没有 print，所以终端只有 7 行 ✅ —— 没有输出不等于没跑。

【怎么改它来学】
    · 把 MAX_TRANSIENT_RETRIES 改成 0 → 第 2 条会怎么变？
    · set_weather_failures([True, True, True]) → 第 3 条的 attempts 是多少？
    · 把第 4 条的 "permanent" 改成 "transient" → 第 7 条还会立即停吗？
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from starter5 import (
    MAX_TRANSIENT_RETRIES,
    call_tool_with_retry,
    execute_tool,
    fetch_weather,
    react_loop,
    set_weather_failures,
)

CITY_ARGS = json.dumps({"city": "Taipei"})


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
    assert first["category"] == "transient"
    assert second["forecast"] == "rain"
    print("✅ test_fetch_weather_failure_plan")


def test_transient_retry_succeeds_within_quota():
    """暂时性错误 → 程序自己重试一次 → 成功。"""
    set_weather_failures([True, False])
    _, obs, is_error, attempts = call_tool_with_retry("fetch_weather", CITY_ARGS)
    assert is_error is False
    assert obs["forecast"] == "rain"
    assert attempts == 2, f"应当重试 1 次（共 2 次尝试），实际 {attempts} 次"
    print("✅ test_transient_retry_succeeds_within_quota")


def test_transient_retry_is_bounded():
    """一直失败 → 尝试次数被 MAX_TRANSIENT_RETRIES 卡住，不会无限重试。"""
    set_weather_failures([True] * 10)
    _, obs, is_error, attempts = call_tool_with_retry("fetch_weather", CITY_ARGS)
    assert is_error is True
    assert obs["category"] == "transient"
    assert attempts == MAX_TRANSIENT_RETRIES + 1, (
        f"上限应卡在 {MAX_TRANSIENT_RETRIES + 1} 次尝试，实际 {attempts} 次"
    )
    print("✅ test_transient_retry_is_bounded")

# 测试脚本对于permanent（不可恢复）错误的处理，验证其完整性
def test_permanent_error_is_not_retried():
    """不可恢复错误 → 一次都不重试。"""
    set_weather_failures(["permanent"])
    _, obs, is_error, attempts = call_tool_with_retry("fetch_weather", CITY_ARGS)
    assert is_error is True
    assert obs["category"] == "permanent"
    assert attempts == 1, "永久错误不应重试"
    print("✅ test_permanent_error_is_not_retried")


def test_bad_input_is_not_retried():
    """模型填错参数 → 程序不重试（重试也没用，要模型自己改）。"""
    set_weather_failures([])
    _, obs, is_error, attempts = call_tool_with_retry("magic_tool", CITY_ARGS)
    assert is_error is True
    assert obs["category"] == "bad_input"
    assert attempts == 1, "bad_input 不应由程序重试"
    print("✅ test_bad_input_is_not_retried")


def test_react_loop_retries_transient_then_answers():
    """loop 里：transient 失败被程序重试掉，模型只看到成功结果。"""
    set_weather_failures([True, False])
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        make_resp("tool_calls", "Check weather.", [make_tool_call("t1", "fetch_weather", {"city": "Taipei"})]),
        make_resp("stop", "Taipei is rainy now."),
    ]
    result = react_loop("Will it rain in Taipei?", client=client)
    assert result["final"] == "Taipei is rainy now."
    assert result["steps"] == 2
    assert result["trace"][0]["attempts"] == 2, "程序应自动重试一次"
    assert result["trace"][0]["obs"]["forecast"] == "rain"
    print("✅ test_react_loop_retries_transient_then_answers")


def test_permanent_error_stops_the_loop():
    """永久错误 → 第一轮就停，不浪费剩余轮次。"""
    set_weather_failures(["permanent", "permanent"])
    client = MagicMock()
    client.chat.completions.create.return_value = make_resp(
        "tool_calls", "Call weather.", [make_tool_call("t1", "fetch_weather", {"city": "Taipei"})]
    )
    result = react_loop("Will it rain in Taipei?", client=client)
    assert result["final"] is None
    assert result["terminal_reason"] == "permanent_tool_error"
    assert result["steps"] == 1, "永久错误应第一轮就停"
    print("✅ test_permanent_error_stops_the_loop")


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
    test_transient_retry_succeeds_within_quota()
    test_transient_retry_is_bounded()
    test_permanent_error_is_not_retried()
    test_bad_input_is_not_retried()
    test_react_loop_retries_transient_then_answers()
    test_permanent_error_stops_the_loop()
    test_malformed_arguments_return_structured_error()
    test_length_is_not_a_final_answer()
    print("\n🎉 全部通過 — 錯誤分類 + 重試上限 + permanent 立即停止，邏輯正確")
