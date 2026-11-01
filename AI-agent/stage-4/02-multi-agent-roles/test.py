r"""Stage 4 练习 2 的离线验收单（不调模型、不花钱）。

【跑法】VS Code 集成终端（PowerShell），先 cd 到本练习目录：
    cd E:\VScode\AI-learning\AI-agent\stage-4\02-multi-agent-roles
    .\.venv\Scripts\python.exe test.py

【这份验收单钉住什么】共 2 组，读的时候只问一句：它在防哪种改坏？
    ① simulate_handoff 的交接语义：research 来自知识库、draft 含有 research、verdict 为 PASS
    ② 真实 Crew 的结构：process=sequential、角色顺序 Researcher→Writer→Critic、
       tasks 的数量与 context 依赖链、每个 agent 都是 max_iter=4

【它没覆盖的】（不是缺陷，是留给真实运行 / 后面的练习）
    - 真实 kickoff 的输出质量（Critic 的判断可能错，测试不校验）
    - 模型是否真的用了工具、是否遵守 max_iter
    - Process.hierarchical 等其他 process 的行为
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from crewai import Process
from starter import build_crew, simulate_handoff


# 🟢 契约：离线流水线的三段产出必须对得上（这是「交接语义」的最小定义）。
def test_offline_research_writer_critic_handoff() -> None:
    result = simulate_handoff("react")
    assert result["input"] == "react"
    assert "ReAct" in result["research"]     # 资料真的来自知识库
    assert result["research"] in result["draft"]  # 稿子真的引用了资料
    assert result["verdict"] == "PASS"       # 有据可依时判 PASS
    assert result["stop_condition"] == "Three fixed roles; each Agent has max_iter=4."


# 🟢 契约：Crew 的结构必须和声明一致。逐一盯死顺序、数量、依赖、上限——
#    任何一条被改动（比如删了 context、换了 process），这里就会挂。
def test_crewai_sequential_context_structure() -> None:
    crew = build_crew("react")
    assert crew.process == Process.sequential                     # 顺序执行，不是别的 process
    assert [agent.role for agent in crew.agents] == ["Researcher", "Writer", "Critic"]  # 角色顺序
    assert len(crew.tasks) == 3                                  # 任务数量
    assert crew.tasks[1].context == [crew.tasks[0]]              # Writer 依赖 Research
    assert crew.tasks[2].context == [crew.tasks[0], crew.tasks[1]]  # Critic 依赖前两份产出
    assert all(agent.max_iter == 4 for agent in crew.agents)     # 每个角色都有迭代上限


# ⚪ 可跳过：手写 runner（没有 pytest 依赖）。
if __name__ == "__main__":
    test_offline_research_writer_critic_handoff()
    test_crewai_sequential_context_structure()
    print("all pass")
