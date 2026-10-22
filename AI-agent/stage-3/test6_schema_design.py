"""练习 6 的验收单 —— starter6_bad.py / starter6_good.py 的契约测试（全程 mock）。

【这个文件在做什么】
    同时验证两个对照组：
      ① 坏 schema 被"误导"选到 process_data 时，管道仍然正确执行（记录现象）
      ② 好 schema 选到 convert_temperature，并算对 89.6°F
      ③ **直接检查两版 schema 的结构差异**（好版有 required + enum，坏版没有）
      ④ 无论 schema 好坏，程序侧验证都必须拦下非法调用
      ⑤ 多单一律整批拒绝

    ⚠️ 前两条是"用 mock 造一个模型行为"，所以它们证明的是**管道正确**，
       不是"真模型一定会这样选"。真机差异要靠固定题目重跑多次（见结果卡）。

【5 个测试各自钉住什么】
    1 test_bad_schema_can_select_wrong_tool          坏 schema 选错时，后续流程不出错
    2 test_good_schema_selects_temperature_tool      好 schema 选对 → 89.6°F
    3 test_good_schema_has_required_fields_and_enum  ⭐核心：断言两版 schema 的结构差异
    4 test_application_rejects_untrusted_calls_for_both_schemas
                                                     两版都要拦住非法调用（11 种坏输入）
    5 test_multiple_calls_are_not_partially_executed 多单整批拒绝

【能证明什么、不能证明什么】
    能证明：两版 schema 的差异确实存在；程序侧验证与 schema 质量无关；
            坏 schema 选错时后续流程不会崩。
    不能证明：真模型在 bad / good 上的**选择差异**——那要用固定题重跑多次。
            这正是本练习最后要填的「结果卡」。

【跑法】必须在 stage-3 目录下运行（顶部 import 的是同目录的两个 starter）
    python test6_schema_design.py
    注意：第 4、5 条没有 print，所以终端只有 3 行 ✅ —— 没有输出不等于没跑。

【怎么改它来学】
    · 把 good 的 enum 删掉 → 第 3 条会挂；真机行为也可能变
    · 把 bad 的 description 改清楚 → 第 3 条仍过（它只查结构），真机可能改善
      ↑ 这说明"结构测试"和"行为测试"是两件事
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import starter6_bad as bad
import starter6_good as good


def make_tool_call(call_id: str, name: str, args: dict):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def make_resp(content: str, tool_calls=None, finish_reason: str = "tool_calls"):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=msg)])


def test_bad_schema_can_select_wrong_tool():
    """模糊 schema 下、LLM 把溫度轉換誤丟給 process_data。"""
    client = MagicMock()
    client.chat.completions.create.return_value = make_resp(
        "The schemas are vague, so I will process the text.",
        [make_tool_call("t1", "process_data", {"data": "32 Celsius to Fahrenheit"})],
    )
    result = bad.select_and_run("Convert 32 Celsius to Fahrenheit.", client=client)
    assert result["tool"] == "process_data"
    assert "processed generic data" in result["observation"]
    print("✅ test_bad_schema_can_select_wrong_tool")


def test_good_schema_selects_temperature_tool():
    client = MagicMock()
    client.chat.completions.create.return_value = make_resp(
        "This is clearly a temperature conversion.",
        [make_tool_call("t1", "convert_temperature", {"value": 32, "unit": "celsius"})],
    )
    result = good.select_and_run("Convert 32 Celsius to Fahrenheit.", client=client)
    assert result["tool"] == "convert_temperature"
    assert result["observation"] == {"value": 89.6, "unit": "fahrenheit"}
    print("✅ test_good_schema_selects_temperature_tool")


def test_good_schema_has_required_fields_and_enum():
    """直接檢查 schema 結構：good 有 required + enum、bad 沒有。"""
    bad_temp = next(t for t in bad.TOOLS_SPEC if t["function"]["name"] == "convert_temperature")
    good_temp = next(t for t in good.TOOLS_SPEC if t["function"]["name"] == "convert_temperature")
    assert "required" not in bad_temp["function"]["parameters"]
    assert good_temp["function"]["parameters"]["required"] == ["value", "unit"]
    assert good_temp["function"]["parameters"]["properties"]["unit"]["enum"] == ["celsius", "fahrenheit"]
    assert good_temp["function"]["parameters"]["additionalProperties"] is False
    print("✅ test_good_schema_has_required_fields_and_enum")


def test_application_rejects_untrusted_calls_for_both_schemas():
    _, bad_unknown = bad.execute_tool("delete_everything", "{}")
    _, good_malformed = good.execute_tool("convert_temperature", "{not-json")
    _, good_missing = good.execute_tool("convert_temperature", json.dumps({"value": 32}))
    _, bad_missing = bad.execute_tool("convert_temperature", json.dumps({"value": "32"}))
    _, bad_unit = bad.execute_tool(
        "convert_temperature", json.dumps({"value": "32", "unit": "kelvin-ish"})
    )
    _, bad_extra = bad.execute_tool("process_data", json.dumps({"data": "rows", "unexpected": "x"}))
    _, bad_wrong_type = bad.execute_tool("process_data", json.dumps({"data": 42}))
    _, good_extra = good.execute_tool("convert_temperature", json.dumps({"value": 32, "unit": "celsius", "unexpected": 1}))
    _, good_wrong_type = good.execute_tool("convert_temperature", json.dumps({"value": True, "unit": "celsius"}))
    _, good_bad_rows = good.execute_tool("process_data", json.dumps({"data": ["row"], "operation": "count_rows"}))
    _, good_bad_operation = good.execute_tool("process_data", json.dumps({"data": [], "operation": "delete_rows"}))
    assert bad_unknown["error"] == "tool not allowed"
    assert good_malformed["error"] == "arguments must be valid JSON"
    assert good_missing["error"] == "arguments contain unexpected or missing fields"
    assert bad_missing["error"] == "arguments contain unexpected or missing fields"
    assert bad_unit["error"] == "unit must be celsius or fahrenheit"
    assert bad_extra["error"] == "arguments contain unexpected or missing fields"
    assert bad_wrong_type["error"] == "data must be a non-empty string"
    assert good_extra["error"] == "arguments contain unexpected or missing fields"
    assert good_wrong_type["error"] == "value must be a number"
    assert good_bad_rows["error"] == "data must be a list of objects"
    assert good_bad_operation["error"] == "operation must be count_rows or list_columns"


def test_multiple_calls_are_not_partially_executed():
    client = MagicMock()
    client.chat.completions.create.return_value = make_resp(
        "two",
        [
            make_tool_call("t1", "convert_temperature", {"value": 32, "unit": "celsius"}),
            make_tool_call("t2", "process_data", {"data": [], "operation": "count_rows"}),
        ],
    )
    result = good.select_and_run("Do both", client=client)
    assert result["tool"] is None
    assert "expected one" in result["observation"]["error"]


if __name__ == "__main__":
    test_bad_schema_can_select_wrong_tool()
    test_good_schema_selects_temperature_tool()
    test_good_schema_has_required_fields_and_enum()
    test_application_rejects_untrusted_calls_for_both_schemas()
    test_multiple_calls_are_not_partially_executed()
    print("\n🎉 全部通過 — Ollama path bad / good schema 對照邏輯正確")
