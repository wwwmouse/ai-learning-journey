# first (ChatCompletion)
# ├── choices[0]
# │   ├── message
# │   │   ├── role: 消息发出者
# │   │   ├── content: 文字回答（调用工具那轮通常是空字符串或 None，取决于供应商）
# │   │   └── tool_calls: 模型点的单（本轮为 None 说明它直接说话了）
# │   │       └── [0].function
#                           └──.name       ← 函数名："get_weather"
# │   │                     └──.arguments  ← 参数：JSON 字符串，要 json.loads 解析成dict
# │   └── finish_reason: "stop" | "tool_calls" | "length"   ← 为什么停（老朋友了）
# └── usage: prompt_tokens / completion_tokens / total_tokens  ← 账单（老朋友了）

# message —— 所有消息的共同点：必须有 role
# │
# ├── role: "user"      → 带 {role, content}          用户说了什么
# ├── role: "assistant" → 带 {role, content?, tool_calls?}   模型说了什么 / 点了什么单
# ├── role: "tool"      → 带 {role, tool_call_id, content}   工具结果（必须有订单号）
# └── role: "system"    → 带 {role, content}          系统设定（以后会见到，通常放最前）

# content：几乎所有消息都有（说话的内容 / 工具结果文本）
# tool_calls：只有 assistant 消息可能有——因为只有模型会点单
# tool_call_id：只有 tool 消息必须有——因为只有工具结果需要认领订单

import json

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

TOOLS = [{
    "type": "function",            # 固定写法：声明"这是一个函数类型的工具"
    "function": {
        "name": "get_weather",     # 工具名字——模型点单时引用这个名字；
                                   # 必须和第 25 行的 def get_weather 对得上
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
# 第一次调用，模型内部没有工具查询实时信息，无法回答。

assistant = first.choices[0].message
messages.append(assistant.model_dump(exclude_none=True))
# 但模型会试图使用TOOLS里面的提前写好的get_weather函数，于是返回：
# {'content': '', 
#  'role': 'assistant', 
#  'tool_calls': 
#      [{'id': 'call_63ypy0ya',
#         'function':
#             {'arguments': '{"unit":"celsius","city":"台北"}',
#              'name': 'get_weather'}, 
#         'type': 'function',
#         'index': 0}]}
# 这里把这一段加入assistant

for call in assistant.tool_calls or []: # 防御性写法：防止模型不点单导致的tool_calls = None ，for x in None会直接崩溃
                                        # or 会在tool_calls = None时返回后值[]，而 for x in []是合法的^^
    if call.function.name != "get_weather":
        raise ValueError(f"不允许的工具：{call.function.name}")
    args = json.loads(call.function.arguments) # 先把参数的 JSON 字符串解析成 dict
    if (
        not isinstance(args, dict)
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