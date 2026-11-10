r"""Path A 离线行为测试：使用 Pydantic AI 官方 TestModel，不连网、不花钱。

【跑法】VS Code 集成终端（PowerShell），先 cd 到本练习目录：
    cd E:\VScode\AI-learning\AI-agent\stage-4\05-typed-agent
    .\.venv\Scripts\python.exe test.py

【这份验收单钉住什么】2 组，读的时候只问一句：它在防哪种改坏？
    ① 正常路径：用官方 TestModel 走完整的 run_sync（含 output_type 校验），
       返回的 AnswerWithConfidence 字段值来自 custom_output_args；
    ② 非法形状被拒：confidence 越界（1.1）、sources 是字符串而不是列表、
       answer 只有空白、sources 是空列表 —— 都必须抛 Pydantic ValidationError。

【它没覆盖的】（留给真实运行 / 后面的练习）
    - 真实模型的输出质量（schema 合法 ≠ 内容真实，测试证明不了真值）；
    - retries=2 的真实重试行为（TestModel 永远合法，走不到重试分支）；
    - 其他边界（超长 answer、confidence 恰好 0/1、重复 sources 等）。
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pydantic import ValidationError
from pydantic_ai.models.test import TestModel
from starter import AnswerWithConfidence, run


# 🟢 契约：官方 TestModel 走完整个 run_sync 链路，输出必须是 AnswerWithConfidence，
#          且字段值就是 custom_output_args 里给的（说明 schema 被真正用上了）。
def test_official_test_model_produces_valid_typed_output() -> None:
    model = TestModel(custom_output_args={"answer": "Offline answer", "confidence": 0.6, "sources": ["offline fixture"]})
    answer = run("Test question", model=model)
    assert answer.answer == "Offline answer"
    assert answer.sources == ["offline fixture"]


# 🟢 契约：四种坏形状都必须被 Pydantic 拦下（这正是 output_type 的价值）。
def test_invalid_contract_values_are_rejected() -> None:
    invalid = [
        {"answer": "ok", "confidence": 1.1, "sources": ["source"]},   # 数值越界
        {"answer": "ok", "confidence": 0.5, "sources": "source"},     # 类型不对（不是 list）
        {"answer": " ", "confidence": 0.5, "sources": ["source"]},    # 空白字符串（自定义校验器）
        {"answer": "ok", "confidence": 0.5, "sources": []},           # 列表为空
    ]
    for payload in invalid:
        try:
            AnswerWithConfidence.model_validate(payload)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"Expected validation failure for {payload!r}")


# ⚪ 可跳过：手写 runner（没有 pytest 依赖）。
if __name__ == "__main__":
    test_official_test_model_produces_valid_typed_output()
    test_invalid_contract_values_are_rejected()
    print("all pass")
