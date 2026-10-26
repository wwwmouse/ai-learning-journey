"""Stage 4 练习 1：同一个 agent、CrewAI 版本对照（Ollama 预设）。

跟 starter.py (LangGraph) 同一个任务：search + summarize。
看程式码风格差异——CrewAI 用 Agent + Task 抽象、LangGraph 用 StateGraph + node。

跑法：
    pip install -r requirements.txt   # 含 crewai
    ollama pull qwen2.5:3b
    ollama serve
    python starter_crewai.py

模型 API 预算：$0（Ollama）；本机硬体与电力仍有成本。

⚠️ 注意：模型、prompt 与工具描述都可能改变步数；用相同题组比较，不先假设哪个 backend 较稳。
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from crewai import Agent, Crew, Task
from crewai.tools import tool

MODEL = os.environ.get("MODEL", "ollama/qwen2.5:3b")  # LiteLLM format
OLLAMA_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")


def search_knowledge_base(query: str) -> str:
    """Search a (fake, offline) knowledge base for a topic."""
    db = {
        "taipei": "Taipei is the capital of Taiwan, population ~2.6M, known for night markets.",
        "react agent": "ReAct (Reasoning + Acting) is an agent pattern: think → act → observe loop.",
    }
    return db.get(query.strip().lower(), f"no entry for {query}")


@tool("search")
def search(query: str) -> str:
    """Search the fixed offline knowledge base."""
    return search_knowledge_base(query)


def build_crew(query: str, llm_model: str = MODEL) -> Crew:
    """CrewAI 风格：一个 agent + 一个 task。"""
    os.environ["OPENAI_API_BASE"] = f"{OLLAMA_BASE}/v1"
    os.environ["OPENAI_API_KEY"] = "ollama"

    researcher = Agent(
        role="Researcher",
        goal="Find and summarize the requested topic.",
        backstory="You search a knowledge base and give concise summaries.",
        tools=[search],
        llm=llm_model,
        verbose=False,
    )
    task = Task(
        description=query,
        expected_output="A 1-2 sentence summary based on search results.",
        agent=researcher,
    )
    return Crew(agents=[researcher], tasks=[task], verbose=False)


def run(query: str, llm_model: str = MODEL) -> dict:
    crew = build_crew(query, llm_model=llm_model)
    result = crew.kickoff()
    return {"final": str(result), "steps": None}


if __name__ == "__main__":
    query = "summarize what you know about Taipei"
    print(f"❓ Query: {query}（using CrewAI + Ollama {MODEL}）")
    print("-" * 60)
    result = run(query)
    print(f"✅ Final: {result['final']}")
    assert result["final"], "expected non-empty summary"
    assert result["steps"] is None
    print("✅ CrewAI 版本通过 — 同样任务、不同 framework、模型 API $0")
    print("   对照 starter.py（LangGraph）看程式码风格差异")
