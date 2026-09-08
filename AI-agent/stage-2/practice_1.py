from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
prompt = """目标：将客服留言分到 billing、bug 或 other。
资料：<input_data>我被扣款两次，请帮我查。</input_data>
规则：只根据资料分类；不知道时选 other。
输出：只返回一个小写标签。"""
reply = client.chat.completions.create(
    model="gemma4:e4b",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
)
print(reply.choices[0].message.content)

# billing