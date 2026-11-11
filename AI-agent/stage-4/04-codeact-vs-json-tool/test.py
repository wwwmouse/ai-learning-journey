r"""Path A 离线行为测试：不调用 CodeAgent.run，也不执行模型代码。

【跑法】VS Code 集成终端（PowerShell），先 cd 到本练习目录：
    cd E:\VScode\AI-learning\AI-agent\stage-4\04-codeact-vs-json-tool
    .\.venv\Scripts\python.exe test.py

    （不需要 Docker；Docker 相关的真机验证是 test_docker_smoke.py，那个才需要 daemon）

【这份验收单钉住什么】3 组，读的时候只问一句：它在防哪种改坏？
    ① JSON tool 边界 + AST 求值：
       - run_json_tool_call({"name":"calculator","arguments":{"expression":"2 + 3 * (4 - 1)"}}) == "11.0"
       - evaluate_arithmetic("-4 / 2") == -2.0
    ② AST 守卫（这是安全的核心）：
       - 代码注入 `__import__('os')`、除零 `2 / 0`、超大数 MAX_NUMBER+1 都必须被拒绝
    ③ 沙箱配置（不启动 Docker，只检查配置值）：
       - 不能有 additional_imports、allow_pickle=False、host=127.0.0.1、
         network_mode=bridge、cap_drop=["ALL"]、security_opt=["no-new-privileges"]

【它没覆盖的】（留给 Docker 真机验证 / 后面的练习）
    - Docker 控制通道是否真的只在 loopback 上（→ test_docker_smoke.py，需要 Docker）
    - 模型是否真的写出了合法代码、CodeAct 跑几步（→ starter.py，需要 Docker + Ollama）
    - 容器清理是否干净（→ run() 的 finally/cleanup，需要真跑才看得到）
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from starter import MAX_NUMBER, codeact_executor_config, evaluate_arithmetic, run_json_tool_call


# 🟢 契约：JSON 路线只接受「名字 + 参数」这一种形状，且算式结果要正确。
def test_json_tool_and_ast_evaluator() -> None:
    assert run_json_tool_call({"name": "calculator", "arguments": {"expression": "2 + 3 * (4 - 1)"}}) == "11.0"
    assert evaluate_arithmetic("-4 / 2") == -2.0


# 🟢 契约：AST 守卫必须挡住「代码注入 / 除零 / 超大数」这三类危险输入。
def test_ast_guards_reject_code_and_unsafe_values() -> None:
    for expression in ("__import__('os')", "2 / 0", f"{MAX_NUMBER + 1}"):
        try:
            evaluate_arithmetic(expression)
        except (SyntaxError, ValueError):
            pass
        else:
            raise AssertionError(f"Expected guarded evaluator to reject {expression!r}")


# 🟢 契约：沙箱配置必须是「最小权限」——任何一项被放宽，这里就会挂。
def test_codeact_uses_loopback_control_and_limits() -> None:
    config = codeact_executor_config()
    assert "additional_imports" not in config
    assert config["allow_pickle"] is False
    assert config["host"] == "127.0.0.1"
    assert config["container_run_kwargs"]["network_mode"] == "bridge"
    assert config["container_run_kwargs"]["cap_drop"] == ["ALL"]
    assert config["container_run_kwargs"]["security_opt"] == ["no-new-privileges"]


# ⚪ 可跳过：手写 runner（没有 pytest 依赖）。
if __name__ == "__main__":
    test_json_tool_and_ast_evaluator()
    test_ast_guards_reject_code_and_unsafe_values()
    test_codeact_uses_loopback_control_and_limits()
    print("all pass")
