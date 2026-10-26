r"""Stage 4 练习 1：同一个 agent、两个 framework — LangGraph + Ollama（Path A、默认）。

任务：搜寻 + 摘要的小 agent。给一个 query、agent 用 search tool 拿资料、回一段摘要。
这份用 LangGraph。对照的 CrewAI 版本见 starter_crewai.py。

【这个练习要你比较什么】（本档 = LangGraph，对照 starter_crewai.py = CrewAI）
    同一个任务写两遍，比的是「框架替你接了什么线、又把什么藏起来」：
      - starter.py（本档）  你亲手用 node + edge 画出回圈 → 控制流看得见，但要自己接线
      - starter_crewai.py  用 Agent + Task + Crew 描述角色与任务 → 上手快，但回圈与工具接线被藏起来
    验收：test.py（LangGraph）与 test_crewai.py（CrewAI）都要跑，两份都是离线 mock。
    想直接把图印出来：show_graph.py。

【这个文件在做什么】（链路 / 状态 / 出口）
    链路：run() → build_graph() 组图 → agent_node 问模型 → should_continue 决定
          →（模型有点单时）tools: tool_node 执行 search、结果包成 ToolMessage → 回 agent_node
          → 最后一条消息没有 tool_calls → END
    状态：只有一个栏位 messages。每个节点只吐出「这一步新增的消息」，由 add_messages 追加（不是覆盖）。
    出口：run() 回传 {"final": str, "steps": int}；steps 是「消息条数」，不是 LLM 呼叫次数。
    和 Stage 3 的关系：Stage 3 用手写 while 完成同一个回圈；这份把它显式化成 node + edge，
          所以每个节点都对应 while 里的一步（问模型 / 执行工具 / 判断是否继续）。

【跑法】VS Code 集成终端（PowerShell），先 cd 到本练习目录：
    cd E:\VScode\AI-learning\AI-agent\stage-4\01-same-agent-two-frameworks
    .\.venv\Scripts\python.exe starter.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

# ⚪ 可跳过：换模型只改这个环境变数（预设对应 ollama pull qwen2.5:3b）。
MODEL = os.environ.get("MODEL", "qwen2.5:3b")


# === Tool ===
# 🟢 必读：docstring 是「给模型看的说明书」——模型只凭它决定要不要呼叫 search。
#          所以该写「查什么、回什么」，而不是写给人看的实作细节。

@tool
def search(query: str) -> str:
    """Search a (fake, offline) knowledge base for a topic."""
    db = {
        "taipei": "Taipei is the capital of Taiwan, population ~2.6M, known for night markets.",
        "react agent": "ReAct (Reasoning + Acting) is an agent pattern: think → act → observe loop.",
    }
    # 契约：永远回字串，查不到也回 "no entry for ..."；不抛错、不回 None。
    return db.get(query.strip().lower(), f"no entry for {query}")


# === LangGraph 状态定义 ===
# 🟢 必读：add_messages 是 reducer。节点回传 {"messages": [...]} 会被「追加」，
#          所以节点只需讲「这一步新产生什么」，不用自己拼历史。

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ⭐ 核心：这张图就是 Stage 3 那个 while 回圈的展开版。
#          三个节点 = while 里的三步；四条接线 = 回圈的控制流。
#          换成 CrewAI 或手写 while，这张图的形状仍然成立。
def build_graph(llm: Any) -> Any:
    """组一个 search → summarize 的图：LLM 看 query 决定要不要呼叫 search、看到 tool result 后给摘要。"""
    # bind_tools 把 search 的 schema 交给模型；不 bind，模型根本不知道有这个工具可点。
    llm_with_tools = llm.bind_tools([search])

    # 回圈第 ① 步：整段历史交给模型，回来的消息（可能带 tool_calls）就是下一步的依据。
    def agent_node(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # 回圈第 ② 步：执行 + 回灌。ToolMessage 必须带 tool_call_id，
    # 模型靠这个 id 把「结果」对回自己的「点单」；对不上就等于没收到结果。
    def tool_node(state: State):
        msg = state["messages"][-1]
        results = []
        for call in msg.tool_calls:
            name = call.get("name")
            args = call.get("args")
            # 安全机制：模型给的名字不能直接当函式执行，只认 allowlist 里的那一个。
            if name != "search":
                raise ValueError(f"Unknown tool: {name!r}")
            # 参数校验：形状不对就拒绝，不要把模型输出直接喂给函式。
            if not isinstance(args, dict) or set(args) != {"query"} or not isinstance(args["query"], str):
                raise ValueError("search needs exactly one string query.")
            obs = search.invoke(args)
            results.append(ToolMessage(content=obs, tool_call_id=call["id"]))
        return {"messages": results}

    # 回圈第 ③ 步：停止条件只看「最后一条消息有没有 tool_calls」，
    # 不看 content 空不空，也不看模型说了什么。
    def should_continue(state: State) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    # 接线：两个节点 + 一个入口 + 一条条件边 + 一条回边（共五个 add_* 呼叫）。
    # 回边（tools → agent）才是「回圈」；少了它，工具执行完就直接结束，模型看不到结果。
    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


# 🟡 看关键：只做三件事 —— 接模型、组图、跑一次。换 provider／换模型只动这几行。
def run(query: str, llm: Any = None) -> dict:
    """Public entrypoint: 跑一轮 agent、回传 final message + steps。"""
    llm = llm or ChatOpenAI(
        base_url="http://localhost:11434/v1", api_key="ollama",
        model=MODEL, temperature=0,
    )
    graph = build_graph(llm)
    # recursion_limit 是「超步」上限，不是「最多几轮对话」；撞上限会抛 GraphRecursionError，
    # 不会回传半成品。要「截断但不崩」的语义，得自己包一层（Stage 7 的事）。
    final_state = graph.invoke({"messages": [HumanMessage(content=query)]}, config={"recursion_limit": 6})
    return {
        "final": final_state["messages"][-1].content,
        "steps": len(final_state["messages"]),
    }


# ⚪ 可跳过：示范用自检，不是机制。assert 失败＝模型没给摘要，或图根本没动。
if __name__ == "__main__":
    query = "summarize what you know about Taipei"
    print(f"❓ Query: {query}（using Ollama {MODEL}）")
    print("-" * 60)
    result = run(query)
    print(f"✅ Final: {result['final']}")
    print(f"   Steps: {result['steps']}")
    assert result["final"], "expected non-empty summary"
    assert result["steps"] >= 2, "expected a visible graph trace"
    print("✅ 练习 1 通过 — LangGraph + Ollama qwen2.5:3b、模型 API $0")
