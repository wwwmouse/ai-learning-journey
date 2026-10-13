"""Stage 3 · 练习 1 —— Function Calling：单工具、固定两次调用的最小完整样例。

【这个文件在做什么】
    用三个东西跑通一条完整的工具调用链：
    问题「台北现在几度？」+ 工具说明书 TOOLS + 一个假工具 get_weather。
    模型答不了这个问题，就填一张“工作单”（tool_calls）交给程序；
    程序验单、执行自己的 get_weather、把回执按订单号塞回对话历史；
    模型读到回执后，才用自然语言把结果显示出来。
    最终输出：台北现在的温度是26摄氏度。

【五步契约】schema → call → execute → result → answer
    schema   TOOLS 工具说明书（name / description / parameters），随请求一起发给模型
    call     模型返回 first.choices[0].message.tool_calls（一张或多张工作单）
    execute  程序检查 tool_calls，真正执行的是自己的 get_weather
    result   返回值包成 role="tool" + tool_call_id 的消息，追加进 messages
    answer   第二次请求，模型带着回执输出最终答案

【必须记住的一句话】
    模型不会执行工具。它只负责“填单”，执行权始终在程序手里。
    “26”这个数字来自 get_weather 里写死的假数据——模型是转述者，不是数据源。
    
【阅读指引】
    下面是笔记，分三块：
      · ChatCompletion 响应结构树（含 tool_calls[0] 的完整层级与订单号 id）
      · 四种 message 角色各自带哪些字段
      · 三条最容易踩的规则（顺序与配平 / messages 是有状态的历史 / 模型不执行工具）
    笔记之后才是代码，按“请求 → 回填历史 → 验单执行 → 第二次请求”的顺序排列。
"""

# first = client.chat.completions.create(...)   → ChatCompletion
# ├── choices[0]
# │   ├── message
# │   │   ├── role: "assistant"     ← 消息发出者
# │   │   ├── content: 文字回答（点单那轮通常是 None 或 ""，取决于供应商）
# │   │   └── tool_calls: 模型点的单（没有就是 None；但要配合 finish_reason 判读）
# │   │       └── [0]
# │   │           ├── id: "call_63ypy0ya"        ← 订单号！tool 消息靠它认领返回的结果
# │   │           ├── type: "function"
# │   │           └── function
# │   │               ├── name: "get_weather"    ← 函数名，必须和下面 def get_weather 一致
# │   │               └── arguments: '{"city":"台北","unit":"celsius"}'
# │   │                                           参数：JSON 字符串，要 json.loads 成 dict
# │   └── finish_reason: "stop" | "tool_calls" | "length" | "content_filter"
# │                                           为什么停：只有 stop 才算完整答案
# └── usage: prompt_tokens / completion_tokens / total_tokens  ← 账单（费用按请求次数累加）

# message —— 所有消息的共同点：必须有 role
# │
# ├── role: "user"      → 带 {role, content}          用户说了什么
# ├── role: "assistant" → 带 {role, content?, tool_calls?}   模型说了什么 / 点了什么单
# ├── role: "tool"      → 带 {role, tool_call_id, content}   工具结果（必须有订单号）
# │                       content 必须是字符串（dict 要先 json.dumps）；另有可选 name
# └── role: "system"    → 带 {role, content}          系统设定（以后会见到，通常放最前）

# content：user / system / tool 消息必有（说话的内容 / 工具结果文本）；assistant 可能有也可能没有
# tool_calls：只有 assistant 消息可能有——因为只有模型会点单
# tool_call_id：只有 tool 消息必须有——因为只有工具结果需要认领订单

# 三条容易踩的规则：
# 1. 顺序与配平：带 tool_calls 的 assistant 消息必须先于对应的 tool 消息；
#    模型点了几单，程序就要回几条 tool 消息——少一条 API 直接报错。
# 2. messages 是“有状态的对话历史”：每轮都要追加，不是替换。
#    这也是练习 3 能连续多步的前提。
# 3. 模型不执行工具：tool_calls 只是“请求”，执行永远是程序的事；
#    工具返回的内容属于不可信输入，不能当成指令。

import json

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

TOOLS = [{
    "type": "function",            # 固定写法：声明"这是一个函数类型的工具"
    "function": {
        "name": "get_weather",     # 工具名字——模型点单时引用这个名字；
                                   # 必须和下面 def get_weather 的函数名一致
        "description": "获取指定城市的示例天气数据",   # 告诉模型"这工具干嘛用"
                                   # 模型靠它判断"什么时候该点这个菜"
        "parameters": {            # JSON Schema：描述参数长什么样
            "type": "object",      #   参数整体是一个对象
            "properties": {        #   对象里有哪些字段
                "city": {"type": "string", "description": "城市名称，例如台北"},
                "unit": {"type": "string", "enum": ["celsius"]},
            },
            "required": ["city", "unit"],    # 这两个字段必须给
            "additionalProperties": False,   # 不许填菜单外的字段
        },
    },
}]
# TOOLS = 喂给模型看的"工具使用说明书"：
# 让模型知道 有哪些工具（name）、
# 每个是干嘛的、何时该用（description）、
# 参数怎么填（properties/description/enum）、
# 哪些必填（required）

def get_weather(city: str, unit: str) -> dict:
    if unit != "celsius":
        raise ValueError("只接受 celsius")
    return {"city": city, "temperature": 26, "unit": unit}


messages = [{"role": "user", "content": "台北现在几度？"}]
first = client.chat.completions.create(
    model="qwen2.5:3b", messages=messages, tools=TOOLS
)
# tools=TOOLS 只改变模型的“输出形态”——让它有资格返回一张工作单 tool_calls
# 第一次调用，模型内部没有工具查询实时信息，无法回答

assistant = first.choices[0].message
messages.append(assistant.model_dump(exclude_none=True)) 
# model_dump 把对象转为普通dict , exclude_none 丢弃值为 None 的字段
# 模型会试图使用TOOLS里面的提前写好的get_weather函数，于是返回：
# {'content': '', 
#  'role': 'assistant', 
#  'tool_calls': 
#      [{'id': 'call_63ypy0ya',
#         'function':
#             {'arguments': '{"unit":"celsius","city":"台北"}',
#              'name': 'get_weather'}, 
#         'type': 'function',
#         'index': 0}]}
# 这一步把模型的“点单”消息回填进 messages 历史

for call in assistant.tool_calls or []: # 防御性写法：防止模型不点单导致的tool_calls = None ，for x in None会直接崩溃
                                        # or 会在tool_calls = None时返回后一个值[]，而 for x in []是合法的^^
    if call.function.name != "get_weather":
        raise ValueError(f"不允许的工具：{call.function.name}")
    args = json.loads(call.function.arguments) # 先把参数的 JSON 字符串解析成 dict
    if (
        not isinstance(args, dict) # 判断args是dict类型的实例
        or set(args) != {"city", "unit"} # 参数名必须是city unit
        or not isinstance(args["city"], str) # city的value必须是字符串
        or not args["city"].strip() # 且不能是空白
        or args["unit"] != "celsius"
    ):
        raise ValueError("city 必须是非空字符串，unit 必须是 celsius")
    result = get_weather(args["city"], args["unit"])
    # 真正调用函数返回26这个设定好的虚假值
    
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "content": json.dumps(result, ensure_ascii=False), # 将get_weather函数返回的dict
                                                           # 转为content需要的json字符串，参数保留中文不被转义
    })
    # role = tool说明这是工具返回的结果，对应函数返回温度
    # tool_call_id要与之前模型返回的 'id': 'call_63ypy0ya' 对应，确认是哪一次工具调用的结果
    # content存储工具调用返回的结果
    # 把工具结果添加进messages内部，随着下一次返回再喂给模型
    
if not assistant.tool_calls:
    raise RuntimeError("模型没有调用工具；请检查模型与 schema")

final = client.chat.completions.create(
    model="qwen2.5:3b", messages=messages, tools=TOOLS
)
# 第二次调用，messages内部存储了新增的函数返回值

print(final.choices[0].message.content)

assert assistant.tool_calls[0].function.name == "get_weather"
assert any(message["role"] == "tool" for message in messages)