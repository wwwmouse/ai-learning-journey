r"""把 build_graph 组出来的图「印出来」（离线，不调模型、不需要 Ollama、不花钱）。

跑法（VS Code 集成终端、PowerShell、在本练习目录下）：
    .\.venv\Scripts\python.exe show_graph.py

原理：build_graph 只需要一个能 bind_tools 的 llm 来接线，组图阶段不会发请求，
      所以这里用 MagicMock() 顶替真模型 —— 看得见图，又不产生任何 API 费用。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from starter import build_graph


def main() -> None:
    graph = build_graph(MagicMock()).get_graph()

    print("=== 节点（node）===")
    for name in graph.nodes:
        print(f"  {name}")

    print("\n=== 边（edge）===")
    for edge in graph.edges:
        kind = "条件边（由 should_continue 决定）" if edge.conditional else "固定边"
        print(f"  {edge.source} -> {edge.target}   [{kind}]")

    print("\n=== 结构图（Mermaid，可粘贴到 https://mermaid.live 看渲染后的图）===")
    print(graph.draw_mermaid())

if __name__ == "__main__":
    main()
