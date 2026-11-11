# Stage 4 笔记：从 Loop 到 Workflow Graph 与框架

> 理解向笔记：只记“脚本让它落地之后才看清的东西”。脚本内部的链路、API、跑法、排错在各自 `starter.py` 注释与 README，这里不重复。

## 1. 一句话总结

Stage 3 你自己写循环；Stage 4 先画工作地图，再选框架接线。框架会藏东西——真正的收获是**判断何时用哪种形状**，不是记住 API。

## 2. 核心概念：README 说了定义，脚本补的是落地

| 概念 | README 的说法 | 脚本里真正长什么样 | 由此得到的判断力 |
|---|---|---|---|
| Framework | 一盒接好线、但会藏细节的积木 | CrewAI：`Agent/Task/Crew` 声明完 `kickoff()` 一口跑完；LangGraph：节点和边由你拼 | 要过程/调状态 → LangGraph；快出雏形 → CrewAI |
| State | 工作中的笔记本 | LangGraph 是显式 `State`；CrewAI 无显式 state，靠 `Task.context` 声明依赖 | 状态在谁手里，决定能否读、存、回放 |
| Handoff | 把工作单交给下一个人 | 一行 `context=[上游 task]`：声明依赖，框架负责注入；没有“发消息”这个动作 | 交接是声明需要谁的输出，不是传数据 |
| Checkpoint | 游戏存档 | `compile(checkpointer=…)` + `thread_id` 槽位；每步留快照 | 没存档就不能恢复；存档粒度决定能回退到哪 |
| HITL | 暂停等人批准 | `interrupt(payload)` 停图；人的决定通过 `Command(resume=值)` 送回 | 人的决定 = resume 的返回值；停和继续是两个动作 |
| Structured Output | 类型安全 | `output_type=` + `retries=`：框架转 schema、校验、重试 | 校验只保形状；能否被满足取决于模型能力 |
| CodeAct | 写代码当 action | 模型代码交给 `DockerExecutor`；对照组靠参数校验 | action 的可信度决定安全边界放哪一层 |

两个补充：**Loop/Graph/Framework 不是一回事**（一次循环 / 画出多步 / 替你接线的积木）；**多 Agent 的门槛在练习 2 变成体感**——多一个角色，多一份不可见性和失败面。

## 3. 框架 vs SDK vs 库

| 层 | 例子 | 谁控制流程 |
|---|---|---|
| SDK / 客户端库 | `openai`：`client.chat.completions.create(...)` | 你 |
| 抽象库 | `langchain_core`（messages/tools）、`pydantic`（schema） | 你 |
| 框架 | LangGraph / CrewAI / Pydantic AI / Smolagents | 框架（它调你） |
| 协议 | OpenAI 兼容 `/v1`（Ollama 用它） | — |

- 判据：**SDK 是你调它，框架是它调你**；一个脚本可叠多层，说“框架”指控制流那一层。
- 混用：分层组合（编排 + 输出校验）常见且健康；两个编排框架深混少见（数据对象不兼容、丢 checkpoint）。真正能跨框架共享的：普通函数、工具协议（MCP）、JSON/dict。

## 4. 五个练习各自的本质

|  练习 | 框架 | 作用（做了什么） | 关键机制 | 带来的提升 / 判断力 |
|---|---|---|---|---|
| 1 | LangGraph、CrewAI | 同一“搜索+摘要”两种实现 | 循环摊开（node/edge）vs 藏进 `kickoff()` | 抽象有价格：要看过程/数步数选 LangGraph，要快选 CrewAI |
| 2 | CrewAI | Researcher→Writer→Critic 固定顺序 | `context` 声明交接；`max_iter`/`allow_delegation` 有界；黑盒靠结构断言+事件日志 | 多 Agent 不是默认答案，固定角色顺序要拿证据换 |
| 3  | LangGraph | 条件分支 + checkpoint + HITL | `checkpointer` 能存、`thread_id` 槽位、`interrupt` 停、`Command(resume)` 续 | 暂停不是 return 是控制流；要分支/停/续就用图 |
| 4 | Smolagents | JSON tool 与受限 CodeAct 各解一遍 | JSON：参数白名单+形状校验；CodeAct：Docker 最小权限沙箱 | 默认 JSON tool（可校验可审计）；CodeAct 只在能隔离时用 |
| 5  | Pydantic AI | 按 Pydantic schema 输出、校验、重试 | `Field` 管形状、`field_validator` 管语义、`retries` 反馈重试；策略与模型能力耦合 | schema 合法 ≠ 内容真实；结构化输出只是第一道闸 |

## 5. 六个横向主题

| 主题 | 出现在 | 代码表现 | 本质 / 启示 |
|---|---|---|---|
| 控制流 | 1/2/3/5 | 手写 `while` → `add_edge` 图 → `kickoff()`/`run_sync()` | 控制力递减、可观测性递减；选型就是在这条线上选位置 |
| 状态 | 1/3（显式）、2（隐式） | `State`/`messages` vs `Task.context` | 状态放哪，决定能否读、存、回放；中间结果重要就选能持久化的写法 |
| 交接 | 2/3 | `context=[上游 task]` / 图的边 | 交接三要素：给够背景、给对权限、给清输出（`expected_output` 只是提示） |
| 存档与 HITL | 3 | checkpointer + thread_id + interrupt/resume | 能存才能停；换 thread_id 等于另起一炉；生产要持久化存档与审批系统 |
| 输出契约 | 5（4 的 AST 守卫同层） | `Field`/`validator`/`retries`；AST 白名单 | 三层检查：形状（schema）、业务规则（代码）、真实性（eval/来源） |
| 观测与验证 | 全部 | 结构断言、事件日志、状态快照、改坏实验 | 看不到内部时靠这三件套；改一处→预测→跑→对照→改回 |

## 6. 我的误区 → 更正

| 误区 | 更正 |
|---|---|
| 用 openai SDK = 用 OpenAI 服务 | SDK 只是传输层，服务由 base_url 决定（我们指向 Ollama） |
| 练习 2 用了 workflow | 只是固定顺序，没有图/存档/HITL——那是练习 3 |
| HITL 在脚本里看不出来 | 程序化 HITL：人的决定是 `run(..., approve=…)` 参数，暂停在第一次 invoke |
| verbose 是 agent 属性 | `Crew(verbose)` 是全局开关；`Agent(verbose)` 只到 agent 自己的日志 |
| retries 是“重发请求” | 是把校验错误反馈给模型再试；retries=2 = 最多 3 次尝试 |
| 结构化输出换个库就能用 | 策略与模型能力耦合（3B 上默认工具模式失败、Native 模式成功） |
| CodeAct 就是“会写代码的 agent” | 关键是“不可信代码必须隔离执行” |
| 框架之间能随意换 | 数据对象不兼容；能共享的是函数、协议、普通数据 |
| 多 Agent 更准更快 | 不保证；多一份 context、测试与失败面，要 eval 证据 |
