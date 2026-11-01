r"""Stage 4 练习 2：用 CrewAI + Ollama 做有界的三角色交接。

3 个 agent 各有角色：
- Researcher 找资料（用 search tool）
- Writer 写稿（拿 researcher 的结果写成 blog 段落）
- Critic 审稿（检查 factual + tone）

这里使用 Process.sequential，让三个角色像接力一样依序交接。

【这个文件在做什么】（链路 / 状态 / 出口）
    链路：run() → build_crew() 建 3 个 Agent + 3 个 Task → kickoff()
          → Researcher（查资料）→ Writer（拿 research 写稿）→ Critic（审稿给 PASS/ISSUES）
    状态：没有显式 state。角色的输出靠 Task.context 声明依赖，由框架在跑下一个 Task 时注入。
    出口：run() 回传 {"final": str, "topic": str, "stop_condition": str}；
          final 是最后一个 Task（Critic）的输出，不保证是"正确答案"（Critic 也是 LLM）。
    和练习 1 的关系：Agent/Task/Crew/kickoff 完全一样；本章新增 4 个东西 ——
          Process.sequential、allow_delegation、max_iter、Task(context=[...])。

【跑法】VS Code 集成终端（PowerShell），先 cd 到本练习目录；ollama 另开一个终端跑：
    cd E:\VScode\AI-learning\AI-agent\stage-4\02-multi-agent-roles
    ollama serve                                  # 另开一个终端，保持执行
    .\.venv\Scripts\python.exe starter.py

【验证】离线、不花钱：
    .\.venv\Scripts\python.exe test.py

【注释约定】Crew: = CrewAI 的 API；↔ = 与练习 1（starter.py/starter_crewai.py）的对照。
模型 API 预算：$0（Ollama）。执行时间依硬体、模型与 prompt 而变。
"""

from __future__ import annotations

import os
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Crew: 比练习 1 多了 Process（决定多任务怎么排）与 Task 的 context（声明数据依赖）。
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

# Crew: 模型用 provider/model 字符串；OLLAMA_BASE 让框架内部的 HTTP 客户端指向本机。
MODEL = os.environ.get("MODEL", "ollama/qwen2.5:3b")
OLLAMA_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")


# === Tool ===
# 🟢 必读：docstring 是给模型看的说明书；schema 由函数签名 + docstring 生成。
#          注意这里把「实现」拆成纯函数 search_knowledge_base，工具只做一层包装。
# 真实做查询的普通函数
def search_knowledge_base(query: str) -> str:
    """查询固定的离线示范资料表，不代表真正的搜寻引擎。"""
    db = {
        "react": "ReAct (Reasoning+Acting, Yao et al. 2022) is the foundational agent pattern: think→act→observe loop.",
        "langgraph": "LangGraph is a graph-based agent orchestration framework by LangChain, focuses on state + checkpointing.",
        "crewai": "CrewAI is a role-based multi-agent framework — define Agent/Task/Crew, run with kickoff().",
    }
    # 契约：永远回字符串；查不到回 "no entry for ..."，不抛错。
    return db.get(query.strip().lower(), f"no entry for {query}")


# 将普通函数用装饰器语法包装成一个 CrewAI 框架可使用的工具: 
# @tool("search") 把函数注册成名为 search 的工具；只有 Researcher 会拿到它
# 它不用于simulate_handoff函数，而是crewai框架与search_knowledge_base函数中的适配层
@tool("search")
def search(query: str) -> str:
    """Return one entry from the fixed offline knowledge base.

    Args:
        query: The lowercase topic key to look up, such as ``react``.
    """
    return search_knowledge_base(query)


# 🟢 必读：这是「不调模型的流水线副本」——用普通 Python 把三段交接走一遍。
#          作用：让离线测试能钉住「交接语义」（research→draft→verdict），不依赖模型、不花钱。
#          ↔ 真正的流程是 build_crew + kickoff；两者输出形状刻意保持一致。
def simulate_handoff(topic: str) -> dict[str, str]:
    """离线走完 Researcher → Writer → Critic，让交接结果可以直接测试。"""
    # 直接调用查询函数本身
    research = search_knowledge_base(topic)
    draft = f"{topic.title()} matters for agent builders: {research}"
    verdict = "PASS" if research != f"no entry for {topic}" and research in draft else "ISSUES: missing grounded research"
    return {"input": topic, "research": research, "draft": draft, "verdict": verdict, "stop_condition": "Three fixed roles; each Agent has max_iter=4."}


# === Crew 设计 ===

# ⭐ 核心：这一题要学的东西全在这个函数里——3 个角色怎么分工、任务怎么排队、数据靠谁传给谁。
#          ↔ 练习 1 只有 1 个 Agent + 1 个 Task，看不出「流水线」和「上下文依赖」。
def build_crew(topic: str, llm_model: str = MODEL) -> Crew:
    """建立三个固定角色；每个 agent 最多迭代四次且不能自行委派。"""
    # Crew: 让框架内部的客户端指向本机 Ollama（OpenAI 兼容协议）。
    os.environ["OPENAI_API_BASE"] = f"{OLLAMA_BASE}/v1"
    os.environ["OPENAI_API_KEY"] = "ollama"

    # 🟢 CrewAI的分工设计哲学：
    #    三个 Agent = 三个「岗位」。role/goal/backstory 会被拼成各自的 system prompt。
    #    新增两个参数（练习 1 没有）：
    #      allow_delegation=False —— 禁止把任务推给其他角色（防互相踢皮球、防跑出流水线）
    #      max_iter=4            —— 每个角色最多迭代 4 次 → 有界，不会无限绕
    researcher = Agent(
        role="Researcher",
        goal=f"Find concise factual info about {topic} from the knowledge base.",
        backstory="You search a knowledge base and return raw factual entries.",
        tools=[search],              # 只有它有工具；Writer/Critic 没有 → 权限最小化
        llm=llm_model,
        verbose=False,
        allow_delegation=False,      # ↔ 练习 1 没这个参数（单角色，没有可委派对象）
        max_iter=4,                  # ↔ 练习 1 由 recursion_limit / 框架默认值兜底
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

    # 🟢 三个 Task = 三个「工单」，各自绑一个 Agent。
    #    description 是输入，expected_output 是验收提示（框架不校验，只是喂给模型）。
    research_task = Task(
        description=f"Search for `{topic}` and report what you find.",
        expected_output="A 1-2 sentence factual entry from the knowledge base.",
        agent=researcher,
    )
    # Crew: context=[research_task] 是「数据依赖声明」——告诉框架：跑这个 Task 之前，
    #       把 research_task 的输出交给 Writer。↔ 练习 1 只有一个 Task，不需要交接。
    write_task = Task(
        description="Write a 2-sentence blog intro using the researcher's findings.",
        expected_output="A 2-sentence intro paragraph.",
        agent=writer,
        context=[research_task],
    )
    # Crew: Critic 同时依赖 research 与 write 两份产出 → 才能对比「稿子是否忠于资料」。
    critic_task = Task(
        description="Check if the writer's intro is factually grounded in the researcher's data. "
                    "Report PASS or list issues.",
        expected_output="Either 'PASS: [intro]' or 'ISSUES: [list]'.",
        agent=critic,
        context=[research_task, write_task],
    )

    # ⭐ Crew: process=Process.sequential → 按 tasks 列表顺序执行（不是让模型自己决定顺序）。
    #        ↔ 练习 1 只有 1 个 Task，process 取什么值都看不出差别；
    #          换成 Process.hierarchical 会变成「主管分派」，行为完全不同。
    return Crew(
        agents=[researcher, writer, critic],
        tasks=[research_task, write_task, critic_task],
        process=Process.sequential,
        verbose=False,
    )


# 🟡 看关键：kickoff 没有返回值类型保证——它给的是 CrewOutput；这里转成 str 并附上可观察的停止条件。
def run(topic: str, llm_model: str = MODEL) -> dict:
    """执行 crew，并把最终结果与可观察的停止条件一起回传。"""
    crew = build_crew(topic, llm_model=llm_model)
    result = crew.kickoff()
    return {"final": str(result), "topic": topic, "stop_condition": "Three fixed roles; each Agent has max_iter=4."}


# ⚪ 可跳过：示范用自检。assert 只保证「有输出」，不判断输出对不对。
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
