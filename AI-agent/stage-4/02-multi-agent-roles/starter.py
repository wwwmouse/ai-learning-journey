r"""Stage 4 练习 2：用 CrewAI + Ollama 做有界的三角色交接。

3 个 agent 各有角色：
- Researcher 找资料（用 search tool）
- Writer 写稿（拿 researcher 的结果写成 blog 段落）
- Critic 审稿（检查 factual + tone）

这里使用 ``Process.sequential``，让三个角色像接力一样依序交接。

跑法：
    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    ollama pull qwen2.5:3b
    ollama serve
    .\.venv\Scripts\python.exe starter.py

验证：
    .\.venv\Scripts\python.exe test.py

模型 API 预算：$0（Ollama）。执行时间依硬体、模型与 prompt 而变。
对照 Anthropic 版见 ``starter_anthropic.py``。
"""

from __future__ import annotations

import os
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

MODEL = os.environ.get("MODEL", "ollama/qwen2.5:3b")
OLLAMA_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")


# === Tool ===

def search_knowledge_base(query: str) -> str:
    """查询固定的离线示范资料表，不代表真正的搜寻引擎。"""
    db = {
        "react": "ReAct (Reasoning+Acting, Yao et al. 2022) is the foundational agent pattern: think→act→observe loop.",
        "langgraph": "LangGraph is a graph-based agent orchestration framework by LangChain, focuses on state + checkpointing.",
        "crewai": "CrewAI is a role-based multi-agent framework — define Agent/Task/Crew, run with kickoff().",
    }
    return db.get(query.strip().lower(), f"no entry for {query}")


@tool("search")
def search(query: str) -> str:
    """Return one entry from the fixed offline knowledge base.

    Args:
        query: The lowercase topic key to look up, such as ``react``.
    """
    return search_knowledge_base(query)


def simulate_handoff(topic: str) -> dict[str, str]:
    """离线走完 Researcher → Writer → Critic，让交接结果可以直接测试。"""
    research = search_knowledge_base(topic)
    draft = f"{topic.title()} matters for agent builders: {research}"
    verdict = "PASS" if research != f"no entry for {topic}" and research in draft else "ISSUES: missing grounded research"
    return {"input": topic, "research": research, "draft": draft, "verdict": verdict, "stop_condition": "Three fixed roles; each Agent has max_iter=4."}


# === Crew 设计 ===

def build_crew(topic: str, llm_model: str = MODEL) -> Crew:
    """建立三个固定角色；每个 agent 最多迭代四次且不能自行委派。"""
    os.environ["OPENAI_API_BASE"] = f"{OLLAMA_BASE}/v1"
    os.environ["OPENAI_API_KEY"] = "ollama"

    researcher = Agent(
        role="Researcher",
        goal=f"Find concise factual info about {topic} from the knowledge base.",
        backstory="You search a knowledge base and return raw factual entries.",
        tools=[search],
        llm=llm_model,
        verbose=False,
        allow_delegation=False,
        max_iter=4,
    )
    writer = Agent(
        role="Writer",
        goal=f"Write a 2-sentence blog intro about {topic}.",
        backstory="You take the researcher's findings and write engaging blog copy.",
        llm=llm_model,
        verbose=False,
        allow_delegation=False,
        max_iter=4,
    )
    critic = Agent(
        role="Critic",
        goal="Verify the writer's blog intro is factually grounded in the researcher's data + check tone.",
        backstory="You're a strict editor who flags hallucinations and tone issues.",
        llm=llm_model,
        verbose=False,
        allow_delegation=False,
        max_iter=4,
    )

    research_task = Task(
        description=f"Search for `{topic}` and report what you find.",
        expected_output="A 1-2 sentence factual entry from the knowledge base.",
        agent=researcher,
    )
    write_task = Task(
        description="Write a 2-sentence blog intro using the researcher's findings.",
        expected_output="A 2-sentence intro paragraph.",
        agent=writer,
        context=[research_task],
    )
    critic_task = Task(
        description="Check if the writer's intro is factually grounded in the researcher's data. "
                    "Report PASS or list issues.",
        expected_output="Either 'PASS: [intro]' or 'ISSUES: [list]'.",
        agent=critic,
        context=[research_task, write_task],
    )

    return Crew(
        agents=[researcher, writer, critic],
        tasks=[research_task, write_task, critic_task],
        process=Process.sequential,
        verbose=False,
    )


def run(topic: str, llm_model: str = MODEL) -> dict:
    """执行 crew，并把最终结果与可观察的停止条件一起回传。"""
    crew = build_crew(topic, llm_model=llm_model)
    result = crew.kickoff()
    return {"final": str(result), "topic": topic, "stop_condition": "Three fixed roles; each Agent has max_iter=4."}


if __name__ == "__main__":
    topic = "react"
    print(f"❓ Topic: {topic}（using CrewAI + Ollama {MODEL}）")
    print(f"   3 agents: Researcher → Writer → Critic（sequential）")
    print("-" * 60)
    result = run(topic)
    print(f"✅ Final (critic's verdict):\n{result['final']}")
    assert result["final"], "expected critic to produce a verdict"
    assert "max_iter=4" in result["stop_condition"]
    print("\n✅ 练习 2 通过 — 3-agent crew 跑完、模型 API $0")
