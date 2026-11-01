r"""把练习 3 的图画出来（离线，不调模型、不需要 Ollama）。

跑法：VS Code 集成终端（PowerShell），先 cd 到本练习目录：
    cd E:\VScode\AI-learning\AI-agent\stage-4\03-graph-workflow
    .\.venv\Scripts\python.exe show_graph.py

原理：build_graph() 只需要一个能 invoke 的 llm 来接线，组图阶段不会发请求，
      所以用下面的 _MockLLM 顶替真模型 —— 看得见图，又不花钱。
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from starter import build_graph


class _MockLLM:
    """组图阶段只需要一个带 invoke 的对象，不会被真正调用。"""

    def invoke(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content="(mock draft)")


def main() -> None:
    graph = build_graph(_MockLLM()).get_graph()

    print("=== 节点（node）===")
    for name in graph.nodes:
        print(f"  {name}")

    print("\n=== 边（edge）===")
    for edge in graph.edges:
        kind = "条件边（由 should_search 决定）" if edge.conditional else "固定边"
        print(f"  {edge.source} -> {edge.target}   [{kind}]")

    print("\n=== 结构图（Mermaid，可粘贴到 https://mermaid.live 看渲染图）===")
    print(graph.draw_mermaid())

    print("=== 用 ASCII 看（可选，需要 grandalf）===")
    try:
        print(graph.draw_ascii())
    except ImportError as error:
        print(f"  （跳过：{error}）")
        print("  想直接看图：.venv\\Scripts\\python.exe -m pip install grandalf")


if __name__ == "__main__":
    main()
