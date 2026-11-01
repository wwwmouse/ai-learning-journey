r"""Path A 离线行为测试：真的走分支、暂停与恢复，但不连网。

【跑法】VS Code 集成终端（PowerShell），先 cd 到本练习目录：
    cd E:\VScode\AI-learning\AI-agent\stage-4\03-graph-workflow
    .\.venv\Scripts\python.exe test.py

【这份验收单钉住什么】2 组，读的时候只问一句：它在防哪种改坏？
    ① 分支 + 暂停/恢复：
       - classify_node 命中关键字 → needs_search=True；should_search 返回 "respond"
       - 图跑到 review 会真的停下（`__interrupt__` 出现在结果里）
       - 用 graph.get_state(config) 能看到暂停时的 draft 还在（checkpoint 有效）
       - Command(resume=True) 之后，最终结果以 PUBLISHED: 开头
    ② 拒绝路径用独立的 checkpoint 线程：
       - 用 thread_id="offline-reject" 跑一遍，结果以 REJECTED: 开头
       - 没走 search 时，draft 里含 "No lookup was needed"（默认上下文生效）

【它没覆盖的】（留给真实运行 / 后面的练习）
    - 真实模型的输出质量（FakeLLM 只回固定格式）
    - InMemorySaver 换成持久化 checkpointer（例如 SQLite）后的行为
    - 并发多个 thread_id / 中断后进程退出再恢复
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from starter import build_graph, classify_node, run, should_search


# ⚪ 可跳过：假模型。只回固定格式，让「图的行为」可重复验证，不依赖真模型。
class FakeLLM:
    """只回固定格式的假模型，让图的行为可以重复验证。"""
    def invoke(self, prompt: str) -> SimpleNamespace:
        # 把 prompt 最后一行（离线资料）拼进 draft，方便断言「资料真的被带进来了」
        return SimpleNamespace(content=f"Draft grounded in: {prompt.splitlines()[-1]}")


# 🟢 契约：分支判断 + 真的暂停 + 存档里还有 draft + 恢复后能走到 PUBLISHED。
def test_branch_choice_and_interrupt_resume() -> None:
    assert classify_node({"query": "latest Taipei population"})["needs_search"] is True
    assert should_search({"needs_search": False}) == "respond"
    graph = build_graph(FakeLLM())
    config = {"configurable": {"thread_id": "offline-approve"}}
    paused = graph.invoke({"query": "latest Taipei population"}, config=config)
    assert "__interrupt__" in paused          # 真的停在人工审核点
    checkpoint = graph.get_state(config)      # 从存档里读 state
    assert "Taipei population" in checkpoint.values["draft"]
    from langgraph.types import Command
    result = graph.invoke(Command(resume=True), config=config)   # 恢复并批准
    assert result["final"].startswith("PUBLISHED:")


# 🟢 契约：换一个 thread_id 就是另一条独立的存档，拒绝 → REJECTED。
def test_rejection_uses_a_separate_checkpoint_thread() -> None:
    result = run("Explain Python", approve=False, llm=FakeLLM(), thread_id="offline-reject")
    assert result["final"].startswith("REJECTED:")
    assert "No lookup was needed" in result["draft"]


# ⚪ 可跳过：手写 runner（没有 pytest 依赖）。
if __name__ == "__main__":
    test_branch_choice_and_interrupt_resume()
    test_rejection_uses_a_separate_checkpoint_thread()
    print("all pass")
