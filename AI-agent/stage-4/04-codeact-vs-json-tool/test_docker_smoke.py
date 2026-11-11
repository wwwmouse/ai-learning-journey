r"""Docker 控制通道的手动冒烟测试（不属于离线测试，需要 Docker daemon）。

【什么时候跑】先确认 `docker version` 成功（Docker Desktop 已启动、Linux 容器在跑），
再跑这个；普通离线验证用 test.py，不需要 Docker。

【跑法】VS Code 集成终端（PowerShell），先 cd 到本练习目录：
    cd E:\VScode\AI-learning\AI-agent\stage-4\04-codeact-vs-json-tool
    docker version                                   # 必须先成功
    .\.venv\Scripts\python.exe test_docker_smoke.py

【它验证什么】
    1. 能按 codeact_executor_config() 建起官方 Jupyter 执行器容器；
    2. 宿主机可以执行一句很小的代码（print(2 + 2) → 日志里有 4）；
    3. Jupyter 控制端口只发布在 127.0.0.1（不会暴露到局域网）。

【它不验证什么】
    - 模型生成的代码质量（那是 starter.py + 3B 模型的事）；
    - 生产级隔离（容器仍可能连网，文件头已说明「这不是 production sandbox」）。
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# SM: DockerExecutor = Smolagents 提供的「在容器里跑代码」的执行器。
from smolagents import DockerExecutor
# SM: AgentLogger / LogLevel → 控制执行器日志安静点，只看 ERROR。
from smolagents.monitoring import AgentLogger, LogLevel

from starter import codeact_executor_config


def test_docker_control_channel() -> None:
    config = codeact_executor_config()
    executor = None
    try:
        # build_new_image=True：第一次会构建/启动官方 Jupyter 镜像（耗时长、占磁盘）
        executor = DockerExecutor(
            additional_imports=[],
            logger=AgentLogger(level=LogLevel.ERROR),
            build_new_image=True,
            **config,
        )
        # 在容器里执行一句代码，验证宿主机 → 容器的控制通道可用
        output = executor.run_code_raise_errors("print(2 + 2)")
        assert "4" in output.logs
        # 检查端口绑定：Jupyter 的 8888 必须只绑在 127.0.0.1
        executor.container.reload()
        bindings = executor.container.attrs["NetworkSettings"]["Ports"]["8888/tcp"]
        assert bindings
        assert all(binding["HostIp"] == "127.0.0.1" for binding in bindings)
    finally:
        # 无论成败都要清理容器，避免残留
        if executor is not None:
            executor.cleanup()


if __name__ == "__main__":
    test_docker_control_channel()
    print("docker smoke pass")
