# Stage 3 笔记：工具使用与第一个 Agent Loop

## 0. 元信息

- **stage**：Stage 3 — Tool Use / 第一个 Agent Loop｜**日期**：2026-11-12（脚本最后修改 2026-10-24）
- **目标来源**：方案给定（`stages/03-tool-use-and-hello-agent.zh-Hans.md` 学习目标 + 自检清单）｜**对齐状态**：首份未对齐（未见 Stage 2 笔记；旧骨架是否统一「待确认」）

## 1. 一句话主线

同一套五步契约，从两次固定调用演进到有界循环与错误分流，全程不用框架。

## 2. 结构图

### 2.1 演进链表

| 脚本 | 相比上一脚本新增 | 复用上一脚本的什么 | 出口/返回值 | 能否离线测 |
|---|---|---|---|---|
| `hello_tool.py` | 五步契约最小闭环（1 工具、固定 2 次调用） | — | 无 `return`；`print` + 2 条 `assert` | ❌ 模块顶层执行、无 `__main__` 守卫 |
| `starter2.py` | 三表 allowlist（`TOOLS_SPEC`/`TOOL_IMPL`/`TOOL_FIELDS`）、单轮选择、多单整批拒绝、错误变字符串 | 五步契约 | `run_tool_selection()` → `{"tool","tool_input","assistant_text","observation"}` | ✅ `test2_multi_tool_selection.py` 8 条 |
| `starter3.py` | `while` 循环 + `max_iter`、`trace`、三出口、`execute_tool` 返回三元组 | `starter2` 的三表结构与 `execute_tool` 六道关（工具改名 `tool_calculator`/`tool_lookup_fact`） | `react_loop()` → `{"final","trace","steps"}`（+`terminal_reason`/`truncated`） | ✅ `test3_react_from_scratch.py` 8 条 |
| `starter5.py` | 结构化错误 `{error,category,retry_hint}`、`call_tool_with_retry`、第 4 出口 | `starter3` 的 `react_loop` 骨架与 `execute_tool` 六道关 | 同 `starter3` + `terminal_reason="permanent_tool_error"` | ✅ `test5_error_handling.py` 9 条 |

### 2.2 对照组表

| 维度 | `starter6_bad.py` | `starter6_good.py` | 两者相同处（控制变量） |
|---|---|---|---|
| description | "Process data." / "Convert a value." | 写明场景 + 排除条件（"不要用于温度换算"） | 工具名都是 `process_data` / `convert_temperature` |
| 参数类型 | 全部 `string` | `number` / `array`（items: object） | 字段名相同（`data`/`operation`/`value`/`unit`） |
| schema 约束（`required`/`enum`/`additionalProperties`） | 全无 | 三者齐全 | — |
| 程序侧 | `execute_tool` + `_validate_args` 六道关 | **同结构** | `select_and_run` 逐字相同、单轮三分支相同 |

### 2.3 一句话读图结论

演进方向是**把原先靠人记的隐式约定（配对、验证、停止、错误归类）逐步写成代码里显式、可测试、可分流的契约**；`starter6` 的对照组则把「schema 是说明书、不是程序保证」这一条隔离出来单独验证。

## 3. 术语增量

| 术语 | 本 stage 中的定义 | 与前序是否一致 | 首次出现 |
|---|---|---|---|
| `tool_calls` | 模型返回的工具请求列表（含 `id`/`type`/`function`），是**请求**不是执行 | 一致 | `hello_tool.py` |
| `tool_call_id` | `assistant.tool_calls[i].id` 与 `role="tool"` 消息的 `tool_call_id` 的配对键 | 一致 | `hello_tool.py` |
| `finish_reason` | 本轮停止原因：`stop` / `tool_calls` / `length` 等；只有 `stop` 才算完整答案 | 一致（首次用于判断） | `starter3.py` `react_loop` |
| `allowlist` | 名字→函数的白名单表 `TOOL_IMPL`；派发只查表，不用 `globals()[name]()` | 一致 | `starter2.py` |
| `observation` | 工具执行结果。**改义**：`starter2`/`starter3` 是 `str`（`"error: ..."`），`starter5` 起是 `dict` | **改义**（见左） | `starter2.py` → `starter5.py` |
| `trace` | 每轮可验证摘要：`step`/`assistant_text`/`tool`/`tool_input`/`obs`/`attempts`/`is_error` | 新 | `starter3.py` |
| `truncated` | 是否撞 `max_iter` 或被 `length` 截断 | 新 | `starter3.py` |
| `terminal_reason` | 非正常结束原因（`length` / `missing_tool_call` / `permanent_tool_error`） | 新 | `starter3.py` |
| `bad_input` | 模型填错的错误类别，由验证层产生 | 新 | `starter5.py` `execute_tool` |
| `transient` | 暂时性错误（超时/限流），程序可自动重试 | 新 | `starter5.py` `fetch_weather` |
| `permanent` | 不可恢复错误（401/403），程序立即停止 | 新 | `starter5.py` `fetch_weather` |
| `retry_hint` | 错误 dict 里给下一步的提示（给程序判断、也给模型改参） | 新 | `starter5.py` |
| `attempts` | 同一次工具调用实际尝试次数（1=未重试） | 新 | `starter5.py` `call_tool_with_retry` |
| `required` | JSON Schema 必填字段列表 | 新 | `hello_tool.py` `TOOLS` |
| `enum` | JSON Schema 枚举，收敛取值空间 | 新 | `hello_tool.py` `TOOLS` |
| `additionalProperties` | JSON Schema 关键字，禁止多余字段 | 新 | `hello_tool.py` `TOOLS` |

## 4. 逐脚本精读

### 4.1 `hello_tool.py`

- **链路**：`messages=[user]`（L102）→ `client.chat.completions.create(tools=TOOLS)`（L107）→ `assistant = first.choices[0].message`（L113）→ 诊断 print 放在 `if assistant.tool_calls:` 内（L120）→ `messages.append(assistant.model_dump(exclude_none=True))`（L128）→ `for call in assistant.tool_calls or []` 验名验参 → `get_weather()` → `messages.append({"role":"tool","tool_call_id":call.id,...})`（L159）→ `if not assistant.tool_calls: raise RuntimeError`（L170）→ 第二次 `create`（L173）→ `messages.append(final_assistant.model_dump(...))`（L179）→ `assert`（L189-190）。
- **状态**：唯一状态是模块级 `messages`；没有 `trace`，没有循环。
- **关键机制**：【你写的胶水】**回填—配对**：`assistant`（带 `tool_calls`）必须先进历史，`role="tool"` 结果才能靠 `tool_call_id` 认领。↔ 与 `starter3` 对照：同一回圈这里手写展开两次，`starter3` 用 `for step in range(max_iter)` 收进循环。
- **边界/坑**：整段在模块顶层执行、无 `__main__` 守卫 → **不可 import、不可测**（`starter2` 第一处改进即是加函数与守卫）；诊断 print 若直接取 `tool_calls[0]`，模型不点单时先抛 `TypeError`、抢掉 `RuntimeError` 兜底（已修）。
- **母题连接**：M1（契约完整度）、M6（不可测的代价）。

### 4.2 `starter2.py`

- **链路**：`run_tool_selection(question, client=None)`（L248）→ 一次 `create(tools=TOOLS_SPEC, messages=[user])` → 三分支（无单 `tool=None` / 多单整批拒绝 / 恰一单）→ `execute_tool(name, raw_arguments)`（L217）六道关 → 返回 `{"tool","tool_input","assistant_text","observation"}`。
- **状态**：**无显式状态**——`messages` 在 `create` 前临时构造、用完即丢；`observation` 直接返回调用方，不回灌模型。
- **关键机制**：【你写的胶水】**三表派发**：`TOOLS_SPEC`（`_wrap()` 工厂生成，给模型看）/ `TOOL_IMPL`（allowlist）/ `TOOL_FIELDS`（名字→字段名，供验证）；`execute_tool` 六道关全部用返回值报错、不抛异常。↔ 与 `hello_tool.py` 对照：1 工具/2 次调用/回灌 → 3 工具/1 次调用/不回灌，深度换广度。
- **边界/坑**：`TOOL_FIELDS` 与 `TOOLS_SPEC` 字段名**手动同步**，改一处忘一处会造成"模型被要求填 A、验证器期望 B"；不查 `finish_reason` → 截断与"主动不用工具"无法区分；`execute_tool` 前 5 关失败返回 `{}`、第 6 关失败返回已解析 `args`，返回值不一致（调用方只用第 2 个返回值，故未爆）。
- **母题连接**：M2（验证边界）、M6（`client` 注入 + mock）。

### 4.3 `starter3.py`

- **链路**：`react_loop(question, max_iter=6, client=None)`（L267）→ `messages=[user]` 与 `trace=[]` 建在循环外 → `for step in range(max_iter)`：`create` → 投影 `assistant_entry`（仅 `id/type/function`）→ `append` → 无 `tool_calls` 时按 `finish_reason=="stop"` 走出口 A、否则出口 B；有 `tool_calls` 时 `for tc in tool_calls` → `execute_tool` 三元组（L225）→ `role="tool"` 回灌 → `trace.append` → 循环跑满 → 出口 C。
- **状态**：`messages` + `trace`，均**建在循环外**、逐轮增长；`steps` 由 `step + 1` 换算（撞上限时用 `max_iter`）。
- **关键机制**：【你写的胶水】**有界循环 + 三出口**：A 正常 `stop`、B 非 `stop`（`terminal_reason`）、C 撞 `max_iter`（`truncated=True`）。↔ 与 `starter2.py` 对照：从"单轮选择器"变成"多轮循环"，且 `observation` 回灌后模型能继续点单。
- **边界/坑**：成功返回的 dict **没有** `truncated` 键（失败才有）→ 调用方必须 `.get()`；`execute_tool` 已返回 `is_error` 却被 `_` 丢弃，程序此时还没有分流能力；`max_iter` 是**轮数**不是工具次数（真机 `step 0` 出现 3 条 trace）；真机实测 qwen2.5:3b 把"台北/纽约"算成 `8336000/2602000`（分子分母颠倒）。
- **母题连接**：M1、M3a（`is_error` 预留）、M4。

### 4.4 `starter5.py`

- **链路**：`react_loop(question, max_iter=5)`（L174）→ 每轮 `call_tool_with_retry(name, raw_arguments)`（L149）→ `execute_tool`（L118）→ `fetch_weather`（L79，读 `_failure_plan`）→ `json.dumps(obs)` 回灌 → `trace` 记 `attempts`/`is_error` → `obs["category"]=="permanent"` 直接返回第 4 出口。
- **状态**：模块级 `_failure_plan`（全局可变夹具，L60）+ `messages` + `trace`；前三个脚本都没有模块级可变状态。
- **关键机制**：【你写的胶水】**错误即数据 + 策略分层**：`transient` → 程序重试（`MAX_TRANSIENT_RETRIES = 2`，即最多 3 次尝试）；`bad_input` → 不重试、回灌模型；`permanent` → 不重试、立即停。**策略**（`call_tool_with_retry`）与**机制**（`execute_tool`：验证+执行一次）分离。↔ 与 `starter3.py` 对照：错误从 `"error: ..."` 字符串变成 dict，因而多一个出口、回灌多一步 `json.dumps`。
- **边界/坑**：`_failure_plan` 是全局可变状态，测试之间必须 `set_weather_failures(...)` 重置；类型 `list[bool | str]` 的 `str` 形式**只在测试里出现**（默认运行产生不了 `permanent`）；真机对比：改造前模型把问题升级给用户（"would you like to try again?"），改造后 `attempts=2`、模型只看到成功结果并答对。
- **母题连接**：M3（责任分层）、M4（第 4 出口）。

### 4.5 `starter6_bad.py` / `starter6_good.py`（对照组，两文件合一节）

- **链路**：`select_and_run(question, client=None)`（bad L138 / good L188）→ 一次 `create(tools=TOOLS_SPEC)` → 无单 / 多单拒绝 / 恰一单 → `execute_tool`（bad L117 / good L168）→ `_validate_args`（bad L97 / good L146）→ 工具实现 → 返回 `{"tool","tool_input","observation"}`。
- **状态**：无（模块级只有 `TOOLS_SPEC` 与 `TOOL_IMPL` 两张常量表）。
- **关键机制**：【你写的胶水】**控制变量对照**：两版 `execute_tool`/`_validate_args`/`select_and_run` 逐字相同，差异只在 `TOOLS_SPEC`（bad L63 / good L70，good 版内联标注了 ①description ②元素类型 ③参数描述 ④enum ⑤required ⑥additionalProperties）。↔ 与 `starter2.py` 对照：三表派发回来了，但焦点从"选哪个工具"移到"说明书怎么写"。
- **边界/坑**：mock 断言"坏 schema 会选 `process_data`"**不等于**真机行为；`test6_schema_design.py::test_good_schema_has_required_fields_and_enum` 只查结构、与模型无关；真机结果卡（固定 1 题 × 3 次）：bad **0/3**（模型一次点两个工具 → 被"只接受一单"拦下），good **3/3**（89.6°F）——失败方式与 README 预期的"选错工具"不同。
- **母题连接**：M2（schema vs 验证）、M5（schema 影响选择）。

## 5. 母题对照

| 母题 | 各脚本中的做法（按演进顺序） | 趋势/结论 |
|---|---|---|
| **M1 五步契约完整度** | `hello_tool.py`：五步全在手写两次调用里；`starter2.py`：缺 `answer`（`observation` 不回灌）；`starter3.py`：五步齐；`starter5.py`：五步齐且错误可分类 | 缺哪步就缺哪项能力：缺回灌＝模型看不到结果（练习 2）；缺 `finish_reason` 判读＝截断当答案 |
| **M2 不可信输入边界** | `hello_tool.py` 手验名字+参数；`starter2/3` 三表 allowlist + 六道关；`starter5` 同上 + 错误分类；`starter6` 对照证明 schema 好坏不改变程序侧验证 | schema 引导模型（**[模型/协议提供]**）、验证由程序兜底（**[你写的胶水]**）。schema 越好，验证越少被触发，但不能删：`starter6_bad` 的坏 schema 仍要 `_validate_args` |
| **M3 错误形态与责任分层** | 异常 `raise`（`hello_tool.py`）→ `"error: ..."` 字符串（`starter2/3`）→ 结构化 dict + `category`（`starter5`）→ 程序三路：retry / 回灌 / 停 | 错误越结构化，越能分「谁处理」：`transient` 归程序、`bad_input` 归模型、`permanent` 归程序立即停；只有字符串时模型只能猜 |
| **M3a 预留接口** | `starter3.execute_tool` 已返回 `is_error`（三元组），`react_loop` 用 `_` 丢弃；`starter5` 拾起并接进 `call_tool_with_retry` | 接口先生产、后使用是常见演进：没有三元组，`starter5` 的分流要重构整个验证层 |
| **M4 循环与停止条件** | `hello_tool` 无循环；`starter2` 单轮三分支；`starter3` 三出口 A/B/C；`starter5` 三出口 + 第 4 出口（`permanent_tool_error`） | 出口从"程序决定"变成"错误类别决定"；`max_iter` 管轮数、管不住一轮多单的成本 |
| **M5 schema 写法影响选择** | 只有 `starter6` 做了对照：bad（无 required/enum/类型）vs good（全有 + 写明何时不用） | 真机 0/3 vs 3/3；bad 的失败不是"选错"而是"两个都要"→ schema 含糊会让模型**不确定**，程序必须接住多单 |
| **M6 离线可测性** | `hello_tool` 不可测；`starter2` `client` 注入 + `__main__` 守卫；`starter3` `side_effect` 多轮；`starter5` 纯函数 `call_tool_with_retry` 可直测；`starter6` 两版同测 + 结构断言 | 可测性取决于**注入点**与**纯函数切分**；黑盒部分只能靠结构断言补 |

## 6. 可复用模式

| 模式 | 触发条件 | 写法/做法（引用函数名） | 承担方 | 不适用场景 |
|---|---|---|---|---|
| 单工具最小链路 | 仅 1 个工具、任务固定两步 | `hello_tool.py`：`TOOLS` + 两次 `create` + `model_dump` 回填 + `tool_call_id` 回执 | 你写的胶水 | 需要多轮或多工具时，手写分支迅速失控 |
| 多工具派发表（三表） | 工具 ≥2 且需 allowlist | `starter2.py` `TOOLS_SPEC`/`TOOL_IMPL`/`TOOL_FIELDS` + `execute_tool` 六道关 | 你写的胶水 | 单工具时是噪音（`starter5`/`starter6` 直接 `if name != ...`） |
| 有界循环骨架 | 模型需多次点单 | `starter3.py` `react_loop`：循环外建 `messages`/`trace`，回填 assistant 后再判停，三出口显式返回 | 你写的胶水 | 固定流程（无需模型决定下一步）时顺序调用更省 |
| 结构化错误 + 分类 | 有网络/外部依赖 | `starter5.py` `fetch_weather` 返回 `{error,category,retry_hint}`；`execute_tool` 原样透传 | 你写的胶水 | 纯本地计算且失败即 bug，分类是多余 |
| 策略与机制分离 | 重试策略会变或需单测 | `call_tool_with_retry`（策略）包 `execute_tool`（机制，只执行一次） | 你写的胶水 | 一次性脚本，多一层反而难读 |
| schema 五条规则 | 多工具、易混、模型选错 | `starter6_good.py` `TOOLS_SPEC`：description 写明场景/排除、正确类型、必填、enum、`additionalProperties: False` | 你写的胶水（说明书写法） | 单工具或工具名已足够自解释 |
| 离线测试注入点 | 要 mock 模型或多轮 | `client=` 参数注入 + `__main__` 守卫 + `MagicMock`/`side_effect`（`test3`、`test5`） | 你写的胶水 + 库提供 | 顶层脚本（`hello_tool.py` 就因此不可测） |
| 控制变量 + 结果卡 | 要回答"哪种写法更好" | `starter6` 两版逐字相同、只改 `TOOLS_SPEC`；固定题 × 重复 N 次记成功率 | 你写的胶水 | 探索期还没定型时，先写出来再比 |

## 7. 我可能没意识到的点

1. **`__main__` 里刻意放宽的断言**：`starter5.py` 断言 `saw_retry` 只看"有没有重试过"，`starter6_bad.py` 只断言"产出了选择"。去掉这层放宽（要求每次都选对）→ 单次真机大概率红；正是"小模型有随机性"迫使断言从"结果正确"退到"机制发生"。
2. **`is_error` 已生产、未使用**：`starter3.execute_tool` 返回三元组，`react_loop` 用 `_` 丢弃。去掉第三返回值（改回二元组）→ `starter5` 的分流无从接入，必须重构验证层。
3. **对照组里被刻意保持不变的部分**：`starter6` 两版 `select_and_run`/`execute_tool`/`_validate_args` 逐字相同。若"顺手"把 good 版验证写得更严 → 真机 0/3 vs 3/3 的差异无法归因到 `TOOLS_SPEC`。
4. **注释里的防御性写法**：`for call in assistant.tool_calls or []`。去掉 `or []` → 模型不点单时 `for x in None` 直接 `TypeError`，本该由后面的 `RuntimeError` 给出可读信息。
5. **测试断言隐含的契约**：`test2_multi_tool_selection.py::test_tool_spec_contains_three_tools` 断言工具名**列表顺序**。若改成 `set(...)` 比较 → 顺序变化不再报警，mock 与 schema 顺序错位时难察觉。

## 8. 自测题

### Q1 · 概念辨析 · 易

**【题干】** 模型返回了 `tool_calls`，是否等于工具已经执行？如果没执行，`messages` 里缺哪一条消息，会让下一次请求失败或让模型收不到结果？

**【参考答案】** 不等于，`tool_calls` 只是请求。缺 `role="tool"` 且带匹配 `tool_call_id` 的消息时，历史不自洽：服务端会拒绝，或模型拿不到结果继续瞎猜。

**【详解】** ①五步契约里 `call` 与 `execute` 是两步，`hello_tool.py` 的 `get_weather()` 是程序侧调用，模型无权执行；②配平要求 `assistant`（含 `tool_calls`）先于 `tool`，且一单对应一回执——`starter2.py` 的"多单整批拒绝"就是避免"只执行一部分"造成历史矛盾。

**【干扰项】** 常见误解是"模型调用工具后自己带回结果"，于是只 append 了 `tool_calls` 就发下一次请求，遇到 `400` 却去怀疑 prompt。

**【定位】** `hello_tool.py` L107–L170（两次 `create` 与 `role="tool"` 回填）；`starter2.py` `run_tool_selection` 多单分支。

**【自评标准】** 能说出"请求 vs 执行"且能指出缺回执的后果 → M1 掌握；只答"要回填" → M1 部分缺口。

---

### Q2 · 概念辨析 · 中

**【题干】** `transient` 与 `bad_input` 都是错误，为什么要分两类？若把 `transient` 也交给模型处理会怎样？

**【参考答案】** 因为处理者不同：`transient` 由程序重试（模型不知道网络状况），`bad_input` 由模型改写参数（只有模型知道想填什么）。交给模型处理 `transient` → 它可能把问题升级给用户或重复同样调用，消耗轮次而修不好。

**【详解】** ①责任分层是 `starter5.py` 的核心：`call_tool_with_retry` 只对 `transient` 重试，`bad_input` 立即返回给 `react_loop` 回灌；②真机证据：改造前模型看到 `network timeout` 回答"would you like to try again?"（推回用户），改造后 `attempts=2`、模型只看到成功结果并答对。

**【干扰项】** 容易答"因为错误信息不同"——那是表象；关键是**谁有能力修**。

**【定位】** `starter5.py` `fetch_weather`（L79）、`call_tool_with_retry`（L149）、`react_loop` 第 4 出口。

**【自评标准】** 能说清"谁有能力修 → 谁处理" → M3 掌握；只背三类名词 → M3 缺口。

---

### Q3 · 概念辨析 · 中

**【题干】** `max_iter` 和"工具调用总数上限"是一回事吗？`react_loop` 里一轮能否发生多次工具执行？

**【参考答案】** 不是。`max_iter` 限制**轮数**；一轮可点多单，`for tc in tool_calls` 会执行多次。控成本要另设"工具调用总数/费用"计数器。

**【详解】** ①真机证据：`starter3.py` 在 `step 0` 产生 3 条 `trace`（两条 `lookup_fact` + 一条 `calculator`），轮数与调用数不等；②`starter3.py`/`starter5.py` 的 `max_iter` 只兜"绕圈"，兜不住"单轮多单"放大——这也是 Stage 3 底线把"最大轮数、timeout、费用上限"并列的原因。

**【干扰项】** 常见误解是"轮数＝模型调用＝工具调用"，于是拿 `max_iter` 当预算上限。

**【定位】** `starter3.py` `react_loop` 内 `for tc in tool_calls`；真机 trace（step 0 三条记录）。

**【自评标准】** 能指出"轮数≠工具次数"并给出多单场景 → M4 掌握；认为二者等价 → M4 缺口。

---

### Q4 · 读码 · 中

**【题干】** `starter3.py` `react_loop` 片段：

```python
assistant_entry: dict = {"role": "assistant", "content": assistant_text}
if tool_calls:
    assistant_entry["tool_calls"] = [...]
messages.append(assistant_entry)

finish_reason = resp.choices[0].finish_reason
if not tool_calls:
    trace.append({...})
    if finish_reason == "stop":
        return {"final": assistant_text, "trace": trace, "steps": step + 1}
```

问：为什么 `append` 必须放在 `if not tool_calls` **之前**？调换会丢什么？

**【参考答案】** "给出答案的那一轮"也是 `assistant` 消息，先 `append` 才保证它进历史；调换后最终答案不在 `messages` 里。

**【详解】** ①历史完整性是 `messages` 的契约（每轮追加不是替换），收尾轮不例外；②`hello_tool.py` 曾漏 append `final_assistant`（或误 append 成旧的 `assistant`），是同类错误的单轮版本——顺序与对象都要对。

**【干扰项】** 容易答"因为要记录 `tool_calls`"——只对一半；无 `tool_calls` 的分支同样需要先 append。

**【定位】** `starter3.py` `react_loop`（`assistant_entry` 构造与收尾段）；对照 `hello_tool.py` L178–179。

**【自评标准】** 答出"最后一轮也要进历史 + 曾用错对象" → M1 掌握；只说"记录 tool_calls" → M1 缺口。

---

### Q5 · 读码 · 中

**【题干】** `starter2.py` `execute_tool` 片段：

```python
field = TOOL_FIELDS[name]
if set(args) != {field}:
    return {}, f"error: arguments must contain exactly {field}"
```

问：为什么用 `set(args)` 而不是 `list(args)`？为什么要求"正好一个字段"而不是"包含该字段即可"？

**【参考答案】** `set` 比较键集合、忽略顺序（模型输出键序随机）；"正好一个"是拒绝多余字段，不给"猜测式修补"留空间。

**【详解】** ①模型返回的 `arguments` 是 JSON 文本，键序不稳定（`hello_tool.py` 示例里 `unit` 在 `city` 前），用列表比较会误判；②多余字段常意味着模型在幻觉参数，默默忽略会掩盖 schema 或描述问题——`starter6_good.py` 用 `additionalProperties: False` 在 schema 层先挡一次，程序侧再挡一次。

**【干扰项】** 容易答"`set` 更快"——性能不是重点，**顺序无关**才是。

**【定位】** `starter2.py` `execute_tool`（六道关第 4 关）；`starter6_good.py` `TOOLS_SPEC` 的 `additionalProperties`。

**【自评标准】** 同时说出"顺序无关"与"拒绝多余字段" → M2 掌握；只答其一 → M2 部分缺口。

---

### Q6 · 读码（bad/good 对照） · 中

**【题干】** 把 `starter6_good.py` `TOOLS_SPEC` 里 `convert_temperature` 的 `"enum": ["celsius", "fahrenheit"]` 删掉，其余不动。问：`test6_schema_design.py` 哪条会挂？真机行为可能怎么变？

**【参考答案】** `test_good_schema_has_required_fields_and_enum` 会挂（断言 `properties["unit"]["enum"] == [...]`）；真机对 `unit` 的约束变弱，可能填出 `kelvin` 之类，但 `_validate_args` 仍会拦下并返回 `unit must be celsius or fahrenheit`。

**【详解】** ①结构测试与行为测试是两件事：结构断言直接读 `TOOLS_SPEC`，与模型无关；②即使 schema 失守，`starter6_good.py` 的 `_validate_args` 仍检查 `unit not in {"celsius","fahrenheit"}`——"schema 是说明书、程序验证是保证"（`starter6_bad.py` 注释同样强调坏 schema 也要安全边界）。

**【干扰项】** 常见误解是"schema 挂了 → 一定选错工具"；实际先挂的是**单元测试**，真机是否选错看模型，且非法参数会被程序拦下。

**【定位】** `starter6_good.py` `TOOLS_SPEC`（`unit` 的 `enum`）、`_validate_args`（L146）；`test6_schema_design.py::test_good_schema_has_required_fields_and_enum`。

**【自评标准】** 同时答出"哪条测试挂"与"程序仍拦" → M2/M5 掌握；只答一边 → 对应母题缺口。

---

### Q7 · 迁移 · 中

**【题干】** 给 `starter3.py` 的 `react_loop` 增加"工具调用总数上限 5"（不是轮数）。说明改哪几处、变量放哪、撞上限返回什么。

**【参考答案】** ①循环外初始化 `tool_calls_used = 0`；②在 `for tc in tool_calls` 内自增；③自增后判断超限，返回 `terminal_reason="tool_budget_exhausted"`、`truncated=True`、`final=None`。

**【详解】** ①`max_iter` 在 `for step in range(max_iter)` 上，管不到"一轮多单"（真机 `step 0` 曾出 3 条 trace），必须单独计数；②`terminal_reason` 要新增取值，否则调用方分不清"轮数用尽"与"预算用尽"；③`trace` 应继续记录，方便事后定位哪一步耗尽。

**【干扰项】** 常见错答是"把 `max_iter` 调小"——那限制轮数，模型一轮点多单照样超预算。

**【定位】** `starter3.py` `react_loop`（`for step` 与 `for tc in tool_calls` 两层循环）。

**【自评标准】** 把计数器放在**轮循环之外、工具循环之内**并区分两种截断原因 → M4 掌握。

---

### Q8 · 迁移 · 难

**【题干】** 把 `starter5.py` 的 `fetch_weather` 换成真实 HTTP API（如公开天气接口），至少说明要改哪四处、各自对应什么风险。

**【参考答案】** ①`timeout`（网络会挂，`max_iter` 管不了耗时）；②退避重试（现在立即重试，真实网络要 `sleep`+抖动）；③错误映射（HTTP 状态码 → `transient`/`permanent`/`bad_input`，401/403 不能重试）；④幂等性（若改成写操作，重试会重复执行）。

**【详解】** ①`starter5.py` 的重试是教学级：`fetch_weather` 由 `_failure_plan` 控制，不会超时或限流，所以 `call_tool_with_retry` 里没有 `sleep`；②责任分层不变（`transient` 归程序、`bad_input` 归模型），只是错误来源从夹具变成 HTTP 响应；③这是"机制正确 → 生产可用"的典型差距，对应底线里的 timeout 与费用上限。

**【干扰项】** 常见误解是"换掉工具函数体就行"——重试语义、超时、错误分类仍停留在教学假设。

**【定位】** `starter5.py` `fetch_weather`（L79）、`call_tool_with_retry`（L149）、`execute_tool` 错误 dict 结构。

**【自评标准】** 四处中答出三处以上 → M3 掌握；只答"改 URL" → M3 缺口。

---

### Q9 · 陷阱 · 难（看起来对其实错）

**【题干】** 看到 `trace` 某条记录是 `{'tool':'fetch_weather','obs':{...forecast: rain...},'attempts':2,'is_error':False}`，有同学判断："`attempts=2` 说明模型学会了重试，这是 prompt 优化的成果。"对否？为什么？

**【参考答案】** 不对。`attempts=2` 是**程序**在 `call_tool_with_retry` 里重试的结果；模型只看到最终成功的 observation，没有参与重试决策。

**【详解】** ①责任分层：`transient` 重试归程序（`MAX_TRANSIENT_RETRIES`），模型只处理 `bad_input`；`trace` 是给调用方的记录，模型看不到；②反证：改造前的真机运行里模型看到错误后回答"would you like to try again?"——它把决策推回用户，说明并不会自主重试。

**【干扰项/为什么像对】** `attempts` 出现在 `trace`（"执行轨迹"）里，容易以为它记录模型行为；实际记录的是**工具调用链路**，模型对调用次数没有决定权。

**【定位】** `starter5.py` `call_tool_with_retry`（L149）、`react_loop` 的 `trace.append`、`MAX_TRANSIENT_RETRIES`（L146）；真机对照记录。

**【自评标准】** 能明确区分"模型行为"与"程序策略" → M3 掌握；把 `attempts` 归因给模型 → M3/M4 缺口。

## 9. 遗留问题与钩子

| 遗留问题 | 去向 |
|---|---|
| `timeout`、重试退避、幂等键 | 【本 stage 未处理】只有教学级重试；真实网络与写操作需要 |
| 费用 / token 上限 | 【本 stage 未处理】`max_iter` 只控轮数，见 Q3 |
| 并行工具调用（一轮多单一律拒绝） | 【已知但超范围】`starter2`/`starter6` 整批拒绝，`starter3`/`starter5` 逐条执行 |
| `TOOLS_SPEC` 与 `TOOL_FIELDS` 双重维护 | 【未解决】字段名两处手写，改一处忘一处即错位；成熟做法是从单一来源（如 pydantic 模型）同时生成 schema 与校验 |
| 练习 4（多步推理）未做 | 【本 stage 未处理】已被 `starter3`（查两次+计算）与 `starter5`（重试后作答）覆盖 |
| Structured Output（`response_format` / JSON mode） | 【本 stage 未处理】只停留在概念对照 |
| 持久化 checkpoint、长期记忆 | 【交给下个 stage】`starter5` 的 `_failure_plan` 只是进程内夹具；Stage 6 讲记忆与检索 |
| 什么时候值得引入框架 | 【交给下个 stage】本 stage 全程手写，Stage 4 才比较"框架替你接了什么线、又藏了什么" |
| `starter5.py` / `starter6_good.py` 在 2026-10-24 改动后未重跑各自 mock test | 【未解决·待确认】下次改动前先跑 `test5_error_handling.py` / `test6_schema_design.py` 确认基线 |
