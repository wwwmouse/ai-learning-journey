r"""Stage 4 练习 3：用 LangGraph 做条件分支与真正可暂停的 HITL。

流程像一条有岔路的轨道：先判断要不要查资料，再请模型写草稿，
接着用 interrupt() 停下来等人决定。只有收到 Command(resume=...) 之后，
图才会继续走到发布或拒绝。Path A 使用 Ollama。

【这个文件在做什么】（链路 / 状态 / 出口）
    链路：START → classify（关键字判断）→ should_search 决定走哪条
          →（需要时）search 查离线资料 → respond 请模型写一句草稿
          → review：interrupt() 暂停，等人类批准/拒绝
          → final：PUBLISHED / REJECTED
    状态：State 是 dict（TypedDict，total=False → 字段可以一开始不存在），
          跨节点共享：query / needs_search / search_result / draft / approved / final。
    出口：run() 回传最终 state；正常时 final 以 "PUBLISHED:" 或 "REJECTED:" 开头。
    和练习 1/2 的关系：
      - 练习 1 的图 = 一个 while 循环（agent ↔ tools）；
      - 练习 3 的图 = 带岔路 + 存档 + 暂停的流程：条件分支、checkpoint、人工批准。
        这才是课程说的 Workflow Graph 的完整能力。

【跑法】VS Code 集成终端（PowerShell），先 cd 到本练习目录；ollama 另开一个终端跑：
    cd E:\VScode\AI-learning\AI-agent\stage-4\03-graph-workflow
    ollama serve                                  # 另开一个终端，保持执行
    .\.venv\Scripts\python.exe starter.py

【验证】离线、不花钱（FakeLLM 假装模型，但真的走分支、暂停与恢复）：
    .\.venv\Scripts\python.exe test.py

【注释约定】LG: = LangGraph 的 API；LC: = LangChain 的 API；↔ = 与前面练习的对照。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Literal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# LC: OpenAI 兼容协议的模型适配器，底层用 openai SDK 发 HTTP；这里指向本机 Ollama。
from langchain_openai import ChatOpenAI
# LG: checkpointer = 「存档器」。没有它，图暂停后就无法恢复。
from langgraph.checkpoint.memory import InMemorySaver
# LG: START / END 是图的固定入口/出口常量；StateGraph 用来建图。
from langgraph.graph import END, START, StateGraph
# LG: interrupt = 暂停整张图并把 payload 交给外面的人；Command(resume=...) = 带着人的决定继续。
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

MODEL = os.environ.get("MODEL", "qwen2.5:3b")


# 🟢 必读：整张图共享的状态。total=False 表示「这些字段不是一开始就都有」，
#          节点按需往里写。↔ 练习 1 的 State 只有 messages 一个字段。
class State(TypedDict, total=False):
    query: str
    needs_search: bool
    search_result: str
    draft: str
    approved: bool
    final: str


# 🟢 必读：分支判断。注意这是「程序规则」（关键字命中），不是让模型决定。
#          ↔ 练习 2 的顺序由 Process.sequential 定；这里走哪条路由这个函数定。
def classify_node(state: State) -> dict[str, bool]:
    """用几个示范关键字决定要不要先走离线查询节点。"""
    query = state["query"].lower()
    return {"needs_search": any(word in query for word in ("population", "weather", "latest", "人口", "天气", "最新"))}


# 🟡 看关键：假的离线资料表，只为演示「分支的一边真的会去取资料」。
#            注意用到的英文短语（Taipei population…）是测试断言依赖的，别改。
def search_node(state: State) -> dict[str, str]:
    """从很小的离线资料表取值；这里不代表真正的搜索引擎。"""
    for key, value in {"taipei": "Taipei population: about 2.6 million.", "台北": "Taipei population: about 2.6 million.", "weather": "Demo weather: sunny, 25 C."}.items():
        if key in state["query"].lower():
            return {"search_result": value}
    return {"search_result": "No matching entry in the offline knowledge base."}


# 🟢 必读：条件边的「路由函数」。它的返回值必须是下面 mapping 里的 key
#          （"search" 或 "respond"），框架据此决定下一站。
def should_search(state: State) -> Literal["search", "respond"]:
    return "search" if state["needs_search"] else "respond"


# 🟡 看关键：模型接线。换 provider 只改这里（或像 test.py 那样注入 FakeLLM）。
def make_ollama_llm() -> ChatOpenAI:
    """用 OpenAI-compatible 接口连到本机 Ollama。"""
    return ChatOpenAI(model=MODEL, base_url="http://localhost:11434/v1", api_key="ollama", temperature=0)


# ⭐ 核心：这张图有 5 个节点、若干固定边、1 条条件边，外加一个存档器。
#          ↔ 练习 1 的图只有 agent/tools + 回边；这里多了「岔路」和「断点」。
def build_graph(llm: Any, checkpointer: Any | None = None) -> Any:
    """建立共用图；调用端可放入 Ollama 或 Anthropic chat model。"""
    # 🟢 模型节点：把 query + 可能的 search_result 拼成 prompt，产出一句草稿。
    #    prompt 用中文要求，避免模型被框架/上下文带成英文。
    def respond_node(state: State) -> dict[str, str]:
        context = state.get("search_result", "No lookup was needed.")
        message = llm.invoke(f"用一句谨慎的话回答：{state['query']}\n离线资料：{context}")
        draft = str(getattr(message, "content", message)).strip()
        if not draft:
            raise RuntimeError("模型返回了空白草稿。")
        return {"draft": draft}

    # ⭐ 本练习的核心：interrupt(payload) 会「暂停整张图」，把 payload 交给外面的人；
    #    它自己不会继续 —— 必须由外面用 Command(resume=...) 重新 invoke 才恢复。
    #    ↔ 练习 2 的 kickoff() 是一口气跑完，没有「停在中途、之后续跑」这个概念。
    def review_node(state: State) -> dict[str, bool]:
        decision = interrupt({"action": "publish", "draft": state["draft"], "question": "Approve this draft?"})
        if not isinstance(decision, bool):
            raise ValueError("人工审核必须用 true 或 false 恢复。")
        return {"approved": decision}

    # 🟢 出口：根据人类决定给出 PUBLISHED / REJECTED 前缀（测试会断言前缀）。
    def final_node(state: State) -> dict[str, str]:
        prefix = "PUBLISHED" if state["approved"] else "REJECTED"
        return {"final": f"{prefix}: {state['draft']}"}

    # LG: 组装图。看三类接线：
    #   add_edge               固定边（一定会走）
    #   add_conditional_edges  条件边（由路由函数的返回值决定）
    #   compile(checkpointer=) 编译成可执行图，并把「存档器」装上
    graph = StateGraph(State)
    graph.add_node("classify", classify_node)
    graph.add_node("search", search_node)
    graph.add_node("respond", respond_node)
    graph.add_node("review", review_node)
    graph.add_node("final", final_node)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", should_search, {"search": "search", "respond": "respond"})
    graph.add_edge("search", "respond")
    graph.add_edge("respond", "review")
    graph.add_edge("review", "final")
    graph.add_edge("final", END)
    # LG: InMemorySaver = 把每一步的 state 存在内存里；interrupt 之后靠它恢复。
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


# ⭐ 核心：同一个 workflow 要「跑两次」。
#    第 1 次 invoke：一路跑到 review 的 interrupt 就返回（此时 state 已存进 checkpointer）。
#    第 2 次 invoke(Command(resume=approve))：从暂停点继续，走到 final。
#    config 里的 thread_id 是「存档槽位」——两次必须用同一个 thread_id 才能接上。
#    ↔ 练习 1/2 都是一次 invoke/kickoff 跑完，这里第一次只跑到半路。
def run(query: str, approve: bool, llm: Any | None = None, thread_id: str = "stage4-demo") -> dict[str, Any]:
    """先跑到人工审核点，再用同一个 thread_id 恢复执行。"""
    graph = build_graph(llm or make_ollama_llm())
    config = {"configurable": {"thread_id": thread_id}}
    paused = graph.invoke({"query": query}, config=config)
    if "__interrupt__" not in paused:
        raise RuntimeError("发布前预期出现人工审核中断，但没有出现。")
    resumed = graph.invoke(Command(resume=approve), config=config)
    if not resumed.get("final"):
        raise RuntimeError("恢复后流程没有产出最终结果。")
    return resumed


# ⚪ 可跳过：示范用自检。一次批准、一次拒绝，都用不同的 thread_id。
if __name__ == "__main__":
    approved = run("What is the Taipei population?", approve=True)
    rejected = run("Explain Python", approve=False, thread_id="stage4-reject")
    print(approved["final"])
    print(rejected["final"])
    assert approved["final"].startswith("PUBLISHED:")
    assert rejected["final"].startswith("REJECTED:")
