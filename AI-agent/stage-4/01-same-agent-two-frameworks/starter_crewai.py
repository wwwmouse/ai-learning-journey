r"""Stage 4 练习 1：同一个 agent、CrewAI 版本对照（Ollama 预设）。

跟 starter.py (LangGraph) 同一个任务：search + summarize。
看程式码风格差异——CrewAI 用 Agent + Task 抽象、LangGraph 用 StateGraph + node。

【跑法】VS Code 集成终端（PowerShell），先 cd 到本练习目录；ollama 另开一个终端跑：
    cd E:\VScode\AI-learning\AI-agent\stage-4\01-same-agent-two-frameworks
    ollama serve                                  # 另开一个终端，保持执行
    .\.venv\Scripts\python.exe starter_crewai.py

【验证】离线、不花钱：
    .\.venv\Scripts\python.exe test_crewai.py

【注释约定】行内标注怎么读：
    Crew: = CrewAI 的 API
    ↔     = 与 starter.py（LangGraph 版）的对应关系
    分层记法：CrewAI 把「谁做什么」声明出来（角色 + 任务 + 团队），循环/提示词/工具调度都在框架里。

模型 API 预算：$0（Ollama）；本机硬体与电力仍有成本。

⚠️ 注意：模型、prompt 与工具描述都可能改变步数；用相同题组比较，不先假设哪个 backend 较稳。
⚠️ 首次运行 CrewAI 会在 %LOCALAPPDATA%\CrewAI 下建数据目录（框架行为）；若安全软件拦截，先放行再跑。
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Crew: 就这三个类，没有别的。CrewAI 对外只暴露「角色 / 任务 / 团队」这层抽象。
#       ↔ LangGraph 版：import 的是 StateGraph / END / add_messages + ChatOpenAI / @tool / 消息类型
from crewai import Agent, Crew, Task

# Crew: 工具装饰器，需要显式给工具名。
#       ↔ LangGraph 版：langchain_core.tools 的 @tool（名字自动取函数名）
from crewai.tools import tool

# Crew: 模型用「provider/model」字符串表示（LiteLLM 风格，这个版本由 crewai-core 自己解析）。
#       ↔ LangGraph 版：MODEL 只是模型名 "qwen2.5:3b"，base_url/api_key 在 ChatOpenAI(...) 里给
MODEL = os.environ.get("MODEL", "ollama/qwen2.5:3b")  # LiteLLM format
OLLAMA_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")


# 纯函数，不是工具：方便离线测试直接调用，也避免在 @tool 里塞业务细节。
# ↔ LangGraph 版：search 的实现直接写在 @tool 函数体里
def search_knowledge_base(query: str) -> str:
    """Search a (fake, offline) knowledge base for a topic."""
    db = {
        "taipei": "Taipei is the capital of Taiwan, population ~2.6M, known for night markets.",
        "react agent": "ReAct (Reasoning + Acting) is an agent pattern: think → act → observe loop.",
    }
    return db.get(query.strip().lower(), f"no entry for {query}")


# Crew: @tool("search") = 把函数注册成名为 search 的工具；docstring 是给模型看的说明书。
#       ↔ LangGraph 版：@tool def search(...)，schema 由签名 + docstring 自动生成
@tool("search")
def search(query: str) -> str:
    """Search the fixed offline knowledge base."""
    return search_knowledge_base(query)


def build_crew(query: str, llm_model: str = MODEL) -> Crew:
    """CrewAI 风格：一个 agent + 一个 task。"""
    # Crew: 让框架内部的 HTTP 客户端指向本机 Ollama（走 OpenAI 兼容协议）。
    #       ↔ LangGraph 版：等价信息写在 ChatOpenAI(base_url=..., api_key=...) 里
    os.environ["OPENAI_API_BASE"] = f"{OLLAMA_BASE}/v1"
    os.environ["OPENAI_API_KEY"] = "ollama"

    # Crew: Agent = 「谁来做」。role/goal/backstory 会被框架拼成 system prompt。
    #       ↔ LangGraph 版：对应 agent_node + 你没有写的 system prompt
    researcher = Agent(
        role="Researcher",                                    # 角色名 → 拼进 prompt
        goal="Find and summarize the requested topic.",        # 目标 → 拼进 prompt
        backstory="You search a knowledge base and give concise summaries.",  # 背景 → 拼进 prompt
        tools=[search],          # ↔ bind_tools([search]) + 你手写的 tool_node（执行/回灌）
        llm=llm_model,           # ↔ ChatOpenAI(...) 客户端对象
        verbose=True,           # 打开后打印框架内部过程（默认关）
    )
    
    # Crew: Task = 「做什么」。description 是任务输入，expected_output 是验收标准。
    #       ↔ LangGraph 版：description ≈ HumanMessage(content=query)；expected_output ≈ 你自己在 prompt 里提的要求
    task = Task(
        description=query,
        expected_output="A 1-2 sentence summary based on search results.",
        agent=researcher,
    )
    
    # Crew: Crew = 「团队」= agents + tasks，本身就描述完了整张流程。
    #       ↔ LangGraph 版：StateGraph(State) + add_node/add_edge/compile
    return Crew(agents=[researcher], tasks=[task], verbose=True)


def run(query: str, llm_model: str = MODEL) -> dict:
    crew = build_crew(query, llm_model=llm_model)
    # Crew: kickoff() = 启动执行。内部跑 ReAct 循环（想 → 选工具 → 执行 → 观察 → …），直到产出最终答案。
    #       ↔ LangGraph 版：graph.invoke(...) + 你手写的 should_continue / 回边
    
    result = crew.kickoff()
    # Crew: kickoff() 返回 CrewOutput 对象；str() 取它的最终文本。
    #       ↔ LangGraph 版：run() 自己从 messages 里取 final，并数出 steps
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
