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

【注释约定】行内标注怎么读：
    LG:  = LangGraph 的 API（画图、调度、状态）
    LC:  = LangChain 的 API（模型适配器、工具、消息类型）
    ↔    = 与 starter_crewai.py（CrewAI 版）的对应关系
    分层记法：CrewAI 把「谁做什么」声明出来；LangGraph + LangChain 把同一件事拆成「图 + 模型/工具/消息」。

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

# LC: 消息类型。HumanMessage = 用户输入；ToolMessage = 工具结果。
#     ↔ CrewAI 版没有这层：消息由框架内部管理，你只给 Task.description
from langchain_core.messages import HumanMessage, ToolMessage

# LC: 工具装饰器：把普通函数变成「带 schema 的工具」（schema 来自函数签名 + docstring）。
#     ↔ CrewAI 版：crewai.tools 的 @tool("名字")，需要显式给名字
from langchain_core.tools import tool

# LC: OpenAI 兼容协议的对话模型适配器，底层用 openai SDK 发 HTTP。
#     ↔ CrewAI 版：Agent(llm="ollama/qwen2.5:3b") 一个字符串，客户端由框架自己造
from langchain_openai import ChatOpenAI

# LG: 图构建器 StateGraph + 终止节点常量 END。
#     ↔ CrewAI 版：Crew(agents=[...], tasks=[...])，团队就是它的「图」
from langgraph.graph import END, StateGraph

# LG: 状态 reducer：节点回传的 {"messages": [...]} 由它负责「追加」，不是覆盖。
#     ↔ CrewAI 版：没有显式 state，执行上下文由框架内部维护
from langgraph.graph.message import add_messages

from typing_extensions import Annotated, TypedDict

# ⚪ 可跳过：换模型只改这个环境变数（预设对应 ollama pull qwen2.5:3b）。
MODEL = os.environ.get("MODEL", "qwen2.5:3b")


# === Tool ===
# 🟢 必读：docstring 是「给模型看的说明书」——模型只凭它决定要不要呼叫 search。
#          所以该写「查什么、回什么」，而不是写给人看的实作细节。

# LC: @tool 把下面的函数注册成一个工具（作用见 import 处的注释）。
#     ↔ CrewAI 版：@tool("search") 显式命名 + 拆出 search_knowledge_base 纯函数
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

# LG: State 是整张图唯一的共享状态。普通 dict 不够，要用 TypedDict + reducer 注解。
#     ↔ CrewAI 版：没有对应的 state 定义（框架内部替你管上下文）
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ⭐ 核心：这张图就是 Stage 3 那个 while 回圈的展开版。
#          三个节点 = while 里的三步；四条接线 = 回圈的控制流。
#          换成 CrewAI 或手写 while，这张图的形状仍然成立。
def build_graph(llm: Any) -> Any:
    """组一个 search → summarize 的图：LLM 看 query 决定要不要呼叫 search、看到 tool result 后给摘要。"""
    # LC: bind_tools 把 search 的 schema 交给模型；不 bind，模型根本不知道有这个工具可点。
    #     ↔ CrewAI 版：Agent(tools=[search])，框架自动做这一步
    llm_with_tools = llm.bind_tools([search])

    # 回圈第 ① 步：整段历史交给模型，回来的消息（可能带 tool_calls）就是下一步的依据。
    # LC: llm.invoke(...) = 真正发一次模型请求。
    #     ↔ CrewAI 版：这一步在 kickoff() 内部，你看不到
    def agent_node(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # 回圈第 ② 步：执行 + 回灌。ToolMessage 必须带 tool_call_id，
    # 模型靠这个 id 把「结果」对回自己的「点单」；对不上就等于没收到结果。
    # ↔ CrewAI 版：工具执行 + 回灌全在 kickoff() 内部完成
    def tool_node(state: State):
        msg = state["messages"][-1]
        results = []
        for call in msg.tool_calls:
            name = call.get("name")
            args = call.get("args")
            # 安全机制：模型给的名字不能直接当函式执行，只认 allowlist 里的那一个。
            # ↔ CrewAI 版：调度由框架负责，你的脚本里没有这个 allowlist
            if name != "search":
                raise ValueError(f"Unknown tool: {name!r}")
            # 参数校验：形状不对就拒绝，不要把模型输出直接喂给函式。
            if not isinstance(args, dict) or set(args) != {"query"} or not isinstance(args["query"], str):
                raise ValueError("search needs exactly one string query.")
            
            # LC: search.invoke(args) = 执行工具（工具对象也有 invoke 接口）
            obs = search.invoke(args)
            
            # LC: ToolMessage = 工具结果的专用消息类型，靠 tool_call_id 与请求配对
            results.append(ToolMessage(content=obs, tool_call_id=call["id"]))
        return {"messages": results}

    # 回圈第 ③ 步：停止条件只看「最后一条消息有没有 tool_calls」，
    # 不看 content 空不空，也不看模型说了什么。
    # ↔ CrewAI 版：停止判断在框架内部（迭代上限 max_iterations），脚本里看不见
    def should_continue(state: State) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    # 接线：两个节点 + 一个入口 + 一条条件边 + 一条回边（共五个 add_* 呼叫）。
    # 回边（tools → agent）才是「回圈」；少了它，工具执行完就直接结束，模型看不到结果。

    # LG: StateGraph(State) = 用刚才定义的状态类型建一张空图。
    #     ↔ CrewAI 版：Crew(agents=[...], tasks=[...]) 一次性把「图」描述出来
    g = StateGraph(State)
    
    # LG: add_node(名字, 函数) = 把普通函数注册成图上的节点
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    
    # LG: set_entry_point = 指定从哪个节点开始跑（图的入口）
    g.set_entry_point("agent")
    
    # LG: add_conditional_edges = 条件边：由 should_continue 的返回值决定下一站
    #     ↔ CrewAI 版：「要不要继续」由框架的 ReAct 循环内部决定
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    
    # LG: add_edge = 固定边。tools → agent 这条就是「回圈」的那条回边
    #     ↔ CrewAI 版：kickoff() 内部的「执行完回到思考」那一步
    g.add_edge("tools", "agent")
    
    # LG: compile() = 把图编译成可执行对象（之后才能 invoke）
    #     ↔ CrewAI 版：Crew(...) 本身就是可执行对象，直接 kickoff()
    return g.compile()


# 🟡 看关键：只做三件事 —— 接模型、组图、跑一次。换 provider／换模型只动这几行。
def run(query: str, llm: Any = None) -> dict:
    """Public entrypoint: 跑一轮 agent、回传 final message + steps。"""
    # LC: 默认造一个模型客户端；调用方传 llm=... 时用它（starter_anthropic.py 与测试靠这个注入）
    #     ↔ CrewAI 版：Agent(llm="ollama/qwen2.5:3b")，模型以字符串形式交给框架
    llm = llm or ChatOpenAI(
        base_url="http://localhost:11434/v1", api_key="ollama",
        model=MODEL, temperature=0,
    )
    graph = build_graph(llm)
    # recursion_limit 是「超步」上限，不是「最多几轮对话」；撞上限会抛 GraphRecursionError，
    # LG: graph.invoke(输入, config) = 执行整张图，返回最终 state。
    #     ↔ CrewAI 版：crew.kickoff()（一个调用跑完整个流程）
    # LC: HumanMessage = 用户消息，作为图的初始输入
    final_state = graph.invoke({"messages": [HumanMessage(content=query)]}, config={"recursion_limit": 6})
    return {
        # 出口契约：final 是最后一条消息的文字；steps 是消息条数（能反映走了几步）
        # ↔ CrewAI 版：run() 回传 {"final": str(CrewOutput), "steps": None}（拿不到步数）
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
