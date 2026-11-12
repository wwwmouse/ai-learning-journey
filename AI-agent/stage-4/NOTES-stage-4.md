# Stage 4 笔记：Workflow Graph 与 Agent 框架

> 依据 `note.md` 骨架整理；引用文件名 + 函数/类名；不覆盖脚本内部已有注释。

## 0. 一句话主线

把一次 Agent 循环升级成可分支、可存档、可人工批准、可校验输出的工程化流程，并学会判断每一步该显式还是交给框架。

## 1. 练习地图

| 练习 | 框架 / 路线 | 核心机制 | 出口契约 | 离线验证方式 |
|---|---|---|---|---|
| 1 | LangGraph（`01/starter.py`）/ CrewAI（`01/starter_crewai.py`） | `StateGraph` + `add_conditional_edges`（显式循环）vs `Agent/Task/Crew` + `kickoff()`（声明式） | LangGraph `run()` → `{"final","steps"}`；CrewAI → `{"final": str(CrewOutput), "steps": None}` | `test.py`（mock LLM）、`test_crewai.py`，均不联网 |
| 2 | CrewAI（`02/starter.py`） | `Process.sequential`、`Task.context`、`allow_delegation=False`、`max_iter=4` | `run()` → `{"final","topic","stop_condition"}`；`test.py` 断言 `process/roles/context/max_iter` | `test.py` 用 `simulate_handoff`（纯 Python 流水线），不联网 |
| 3 | LangGraph（`03/starter.py`） | `add_conditional_edges` 分支、`InMemorySaver` 存档、`interrupt()`/`Command(resume=)` 暂停恢复、`thread_id` | `run()` → 最终 state；`final` 以 `PUBLISHED:` 或 `REJECTED:` 开头 | `test.py`（`FakeLLM`，真走暂停与恢复） |
| 4 | Smolagents（`04/starter.py`） | JSON tool 边界 `run_json_tool_call`；AST 守卫 `evaluate_arithmetic`；`CodeAgent` + `DockerExecutor` 沙箱 | `run_json_tool_call()` → `str`；`run()` → `{"final": str}`；`codeact_executor_config()` 配置断言 | `test.py`（不调模型、不需要 Docker）；`test_docker_smoke.py` 需 Docker |
| 5 | Pydantic AI（`05/starter.py`） | `Agent(output_type=AnswerWithConfidence, retries=2)`；`NativeOutput`；`@field_validator` | `run()` → `AnswerWithConfidence` 实例 | `test.py`（官方 `TestModel`），不联网 |

## 2. 逐练习精读

### 2.1 练习 1：同一个 agent、两个 framework

- **链路**：`starter.py` 的 `run()` → `build_graph()`：`StateGraph(State)` → `add_node("agent")` / `add_node("tools")` → `add_conditional_edges("agent", should_continue, ...)` → `compile()` → `graph.invoke({"messages":[HumanMessage(...)]}, config={"recursion_limit":6})` → 取最后一条 `AIMessage`。`starter_crewai.py` 的 `run()` → `build_crew()` → `Crew(agents=[researcher], tasks=[task]).kickoff()`。
- **状态**：LangGraph 用 `State.messages`（`add_messages` reducer）；CrewAI 没有显式 state，`Task(description=query, expected_output=...)` 就是全部输入。
- **关键机制**：同一个 agent loop 的两种写法。框架替你做了：消息合并、工具调用解析、重试、日志；你亲手做了：prompt、工具函数、停止条件（`should_continue`）。↔ 练习 2 把它扩成三角色；练习 3 把它换成带分支和存档的图。
- **坑与边界**：CrewAI 不给你 `steps`（`None`），想数步数只能用 LangGraph；`recursion_limit=6` 是超步上限，超了抛 `GraphRecursionError`，不是截断结果；CrewAI 运行后出现的 0 字节 `crewai` 文件与 `.crewai\last_run.json` 是 kickoff/tracing 产物，不是报错。

### 2.2 练习 2：多 agent 角色分配

- **链路**：`run(topic)` → `build_crew(topic)` 建三个 `Agent`（Researcher/Writer/Critic）+ 三个 `Task`（`write_task` 的 `context=[research_task]`，`critic_task` 的 `context=[research_task, write_task]`）→ `Crew(process=Process.sequential).kickoff()`。
- **状态**：无显式 state；交接靠 `Task.context` 声明，框架在跑下一个 Task 前把上游输出注入。
- **关键机制**：①`Process.sequential`（枚举，只有 `sequential`/`hierarchical`）定顺序；②`Task.context` 声明数据依赖；③`allow_delegation=False` + `max_iter=4` 定边界；④`Crew` 是运行时容器（绑定、排序、注入、聚合），`kickoff()` 是入口。↔ 练习 1 只有 1 Agent + 1 Task，`process`/`context` 都看不出差别。
- **坑与边界**：`Process.hierarchical` 必须提供 `manager_llm`/`manager_agent`，否则 `Crew` 构造报 `ValidationError`；`expected_output` 只是提示、不校验；框架提示词是英文，输出中文要在 `Task.description` 里明确要求；`test.py` 逐字断言 `stop_condition` 字符串，改文案会挂。

### 2.3 练习 3：图式 workflow（分支 + checkpoint + HITL）

- **链路**：`run(query, approve)` → `build_graph(llm)`：`classify_node` → `add_conditional_edges("classify", should_search, ...)` → `search_node` 或直接 `respond_node` → `review_node`（`interrupt(...)`）→ `final_node`；第一次 `graph.invoke({"query":...})` 停在 `__interrupt__`，第二次 `graph.invoke(Command(resume=approve))` 继续。
- **状态**：显式 `State`（`TypedDict`, `total=False`）：`query`、`needs_search`、`search_result`、`draft`、`approved`、`final`；`config={"configurable":{"thread_id":...}}` 决定存档槽。
- **关键机制**：分支靠 `add_conditional_edges` + `should_search`；存档靠 `compile(checkpointer=InMemorySaver())`；暂停靠 `interrupt()`（控制流暂停，不是 return）；恢复靠 `Command(resume=...)`。↔ 练习 1 的图是循环，这里的图是流程；练习 2 的 sequential 一口气跑完，这里能停在中途。
- **坑与边界**：无 checkpointer 时 resume 抛 `RuntimeError: Cannot use Command(resume=...) without checkpointer`；换 `thread_id` 恢复会 `KeyError: 'query'`（新线程空存档、从头重跑）；条件边返回未映射的 key 抛 `KeyError`；`InMemorySaver` 进程退出即丢，持久化存档是 Stage 6/7 的内容。

### 2.4 练习 4：CodeAct vs JSON tool

- **链路**：离线部分 `evaluate_arithmetic` → `calculator` 工具 → `run_json_tool_call`；真机部分 `run(question)` → `build_codeact_agent()` → `CodeAgent.run()` → `DockerExecutor` 在容器里执行 → `finally: agent.cleanup()`。
- **状态**：无图、无显式 state；只有工具返回值与 `max_steps=4` 的步数上限。
- **关键机制**：两条 action 路线。JSON tool 的边界在 `run_json_tool_call`（字段必须恰好 `{name, arguments}`、工具白名单、参数形状）；CodeAct 的边界在 `codeact_executor_config()`（`allow_pickle=False`、`host=127.0.0.1`、`network_mode="bridge"`、`cap_drop=["ALL"]`、`security_opt=["no-new-privileges"]`）。↔ 练习 1/3 都是 JSON tool 路线（模型只回工具调用）；这里让模型直接写代码。
- **坑与边界**：模型代码一律不可信，不能在主机执行；`evaluate_arithmetic` 故意不用 `eval/exec`，只走 AST 白名单；容器仍可能连网，不是 production sandbox；`test_docker_smoke.py` 要先 `docker version` 成功；当前状态：离线 `test.py` 已通过，真机 CodeAct **未跑（Docker daemon 未启动）→ 待确认**。

### 2.5 练习 5：类型安全 agent

- **链路**：`run(question)` → `build_agent()`：`Agent(model, output_type=..., retries=2, instructions=...)` → `run_sync(question).output` → 再做一次业务契约检查。
- **状态**：无图；"状态"就是最后那个通过校验的 `AnswerWithConfidence`。
- **关键机制**：输出契约（`Field(min_length/ge/le)` 管形状、`@field_validator` 管语义）；重试（`retries=2` = 最多 3 次尝试，把校验错误反馈给模型）；输出策略（默认 `ToolOutput` / `NativeOutput` / `PromptedOutput`，3B 模型上只有 `NativeOutput` 稳定；`TestModel` 不支持 `NativeOutput`，所以 `build_agent` 按模型类型切换）。↔ 练习 4 的边界在参数/沙箱，这里在 schema；练习 3 的出口是自由文本，这里的出口是结构化对象。
- **坑与边界**：schema 合法 ≠ 内容真实；`retries` 会花 token 和时间，仍可能失败；`Field(ge/le)` 管不了"全是空格"，要 validator；`result.usage` 是属性不是方法。

## 3. 跨练习对照表（本 stage 最重要的一节）

| 母题 | 练习 1 | 练习 2 | 练习 3 | 练习 4 | 练习 5 | 趋势 / 结论 |
|---|---|---|---|---|---|---|
| 控制流可见性 | 图摊开，能数 steps | 藏进 `kickoff()`，只给 final | 图摊开，且能停在半路 | 模型写代码，边界在容器 | 藏在 `run_sync()` 里 | 抽象越强、代码越短、可观测性越差；选型 = 选位置 |
| 状态与交接 | `State.messages` 共享 | `Task.context` 声明依赖 | 显式 `State` + 存档 | 无 state，只有工具返回值 | 无 state，输出即状态 | 隐式交接（context）够用，但中间结果不可读；要读要存就得显式 State |
| 边界与安全 | 无（单 agent 无副作用） | `max_iter`/`allow_delegation` 约束行为 | 人工批准点 = 业务边界 | AST 守卫 + Docker 最小权限 | schema 校验 = 数据边界 | 边界从"行为约束"到"沙箱隔离"到"数据校验"，层次在升高 |
| 结构化输出 | 自由文本 | 自由文本（`expected_output` 不校验） | 自由文本，靠 interrupt 人工看 | 工具返回固定字符串 | Pydantic 强校验 | 只有练习 5 把"形状"变成程序可验证的契约 |
| 可测试性 | mock LLM 测图结构 | 纯函数 `simulate_handoff` + 结构断言 | FakeLLM 真走暂停/恢复 | 不调模型，测 AST/边界/配置 | 官方 TestModel | 都能离线验证；黑盒框架靠"结构断言"补 |
| 人机协作 | 无 | 无 | `interrupt()` + `Command(resume=)` | 无 | 无 | 只有练习 3 把"人"变成流程中的一个节点 |
| 失败表现 | 循环超限抛 `GraphRecursionError` | 校验/构造期报错（`ValidationError`） | `KeyError`/`RuntimeError` 落在恢复路径 | Docker 连接失败/容器清理 | `Exceeded maximum output retries` | 越靠后，失败越"结构化"、越容易被测试抓到 |

## 4. 本 stage 的可复用模式

- **模式 A：固定路线优先**。能用 workflow 完成就别让模型决定下一步。适用：步骤稳定、验收标准清楚；不适用：探索型任务、需要动态分支。
- **模式 B：框架接循环，你只写节点/工具/停止条件**（`add_node` + `add_conditional_edges`）。适用：要 state/checkpoint/可视化；不适用：简单一次性调用，用 SDK 更透明。
- **模式 C：交接用显式依赖声明**（`Task.context=[...]` 或图的边），不要靠隐式传参。适用：多角色/多步骤；不适用：单任务单角色时是噪音。
- **模式 D：有副作用或花钱的步骤前置 `interrupt()` + checkpointer**。适用：发布、发送、删除；不适用：纯计算步骤，加了只会拖慢。
- **模式 E：结构化输出用 Pydantic 契约 + 校验 + 重试**，但别把 schema 当真实性证明。适用：下游要机器消费；不适用：自由文本对话。
- **模式 F：不可信代码一律沙箱**，默认选 JSON tool，CodeAct 只在需要灵活生成逻辑且能隔离时用。适用：数据分析/计算；不适用：沙箱不可用的环境。
- **模式 G：黑盒框架的观测三件套**——结构断言、事件日志（verbose）、状态快照（`get_state`/`usage`）。适用：框架藏起内部时；不适用：能直接读源码的简单场景。

## 5. 我可能没意识到的点

1. **`Crew` 只是容器，数据通道是 `Task.context`**。去掉 `write_task` 的 `context=[research_task]`，Writer 拿不到资料——`Crew` 本身不会替你传数据。
2. **`expected_output` 不参与校验**。去掉它，程序行为不变，但模型少了"验收提示"，输出格式更容易跑偏。
3. **`interrupt()` 是控制流暂停，不是 `return`**。改成 `return {"approved": False}` 就失去"暂停—等人—继续"的能力，流程会直接结束。
4. **`thread_id` 是存档槽位，不是身份标识**。换一个 `thread_id` 恢复，就等于另起一炉（实测 `KeyError: 'query'`）。
5. **`TestModel` 有输出模式的能力边界**。它不支持 `NativeOutput`；不按模型类型切换 `output_type`，离线测试会直接抛 `UserError`。
6. **`Crew(verbose=True)` 是全局开关**。只开 `Agent(verbose=True)` 看不到 Task/Tool/Crew 级面板，交接也没有独立事件。

## 6. 自测题

**概念辨析**

1. `Process.sequential` 和 `add_conditional_edges` 解决的是同一个问题吗？
2. `interrupt()` 和 `return` 在控制流上有何区别？为什么恢复必须用 `Command(resume=...)`？
3. `Field(ge=0, le=1)` 和 `@field_validator` 各管什么？只用前者会漏掉哪种坏数据？

**读码**

4.

```python
def review_node(state: State) -> dict[str, bool]:
    decision = interrupt({"action": "publish", "draft": state["draft"], "question": "Approve this draft?"})
    if not isinstance(decision, bool):
        raise ValueError("人工审核必须用 true 或 false 恢复。")
    return {"approved": decision}
```

问：为什么这里不能写成 `return {"approved": False}`？

5.

```python
write_task = Task(
    description="用 Researcher 的发现，写一段两句话的 blog 开头。",
    expected_output="一段两句话的开头。",
    agent=writer,
    context=[research_task],
)
```

问：去掉 `context=[research_task]` 后，运行结果会少什么？`test.py` 会怎样？

**迁移**

6. 若要给练习 3 加"两级审批"（先经理、后法务），你会改哪一层？需要几个 `interrupt()`、几次 `resume`？
7. 若换成更强的模型，练习 5 的 `build_agent` 里哪段可以删？为什么？

## 7. 遗留问题 / 下个 stage 的钩子

- **本 stage 故意没做**：持久化 checkpointer（`InMemorySaver` 进程退出即丢）、eval、预算/观测、并行分支；练习 4 的真机 CodeAct 未验证（Docker daemon 未启动）→ 待确认。
- **留给 Stage 5**：Claude Code 生态与 MCP——跨框架共享工具，正好接上"框架能不能混用"里"共享工具"那条。
- **留给 Stage 6/7**：记忆与长期 checkpoint（Stage 6）、多 agent 上线/观测/恢复（Stage 7）。
- **待确认**：qwen2.5:3b 在练习 4 的 CodeAct 上到底能走几步——依赖 Docker 环境实测。
