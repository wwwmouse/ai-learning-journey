r"""Offline CrewAI comparison checks; no crew kickoff or network call.

【跑法】VS Code 集成终端（PowerShell），先 cd 到本练习目录：
    cd E:\VScode\AI-learning\AI-agent\stage-4\01-same-agent-two-frameworks
    .\.venv\Scripts\python.exe test_crewai.py

【这份验收单钉住什么】只检查 Crew 结构，不 kickoff、不联网、不花钱：
    - search 工具的两个分支（命中 / 未命中）
    - Crew 里恰好 1 个 agent（Researcher）、1 个 task，且 task 绑在这个 agent 上
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from starter_crewai import build_crew, search_knowledge_base


def test_offline_search_and_visible_crewai_structure() -> None:
    assert "Taipei" in search_knowledge_base("Taipei")
    assert "no entry" in search_knowledge_base("unknown")
    crew = build_crew("summarize Taipei")
    assert [agent.role for agent in crew.agents] == ["Researcher"]
    assert len(crew.tasks) == 1
    assert crew.tasks[0].agent.role == "Researcher"


if __name__ == "__main__":
    test_offline_search_and_visible_crewai_structure()
    print("all pass")
