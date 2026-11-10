r"""Stage 4 练习 5：用 Pydantic AI 的 output_type 检查输出形状。

类型检查像表格的格子：它能确保字段存在、类型正确、数字在范围内，
但不能证明内容是真的。真实性仍要靠可信来源与额外的语义检查。

【这个文件在做什么】（链路 / 状态 / 出口）
    链路：run(question) → build_agent() 建 Agent（output_type=AnswerWithConfidence、retries=2）
          → run_sync(question)（内部：模型输出 → Pydantic 校验 → 不合法就重试，最多 2 次）
          → 返回一个已校验的 AnswerWithConfidence 实例
    状态：没有图、没有显式 state；「状态」就是最后那个通过校验的 Pydantic 模型。
    出口：AnswerWithConfidence：answer（非空）/ confidence（0..1）/ sources（至少 1 个非空标签）。
    和前面练习的关系：
      - Stage 3 的 Structured Output 是「让模型回 JSON」；这里把「出 JSON → 校验 → 失败重试」交给框架；
      - 练习 1/3 的图保证「流程形状」；这题保证「最后一步的数据形状」。

【跑法】VS Code 集成终端（PowerShell），先 cd 到本练习目录；ollama 另开一个终端跑：
    cd E:\VScode\AI-learning\AI-agent\stage-4\05-typed-agent
    ollama serve                                  # 另开一个终端，保持执行
    .\.venv\Scripts\python.exe starter.py

【验证】离线、不花钱（用官方 TestModel，不连网）：
    .\.venv\Scripts\python.exe test.py

【注释约定】PA: = Pydantic AI 的 API；PD: = Pydantic 的 API；↔ = 与前面练习的对照。
"""

from __future__ import annotations

import os
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# PD: 数据模型 + 字段约束 + 自定义校验器。
from pydantic import BaseModel, Field, field_validator
# PA: Pydantic AI 的 Agent：把「模型调用 + 输出 schema + 校验重试」包在一起。
from pydantic_ai import Agent
# PA: 结构化输出有三种策略，小模型上差别很大：
#     ToolOutput（默认，走 tool calling）/ NativeOutput（走 provider 的 JSON schema 模式）
#     / PromptedOutput（让模型在文本里回 JSON）。下面 build_agent 用 NativeOutput。
from pydantic_ai.output import NativeOutput
# PA: 官方离线测试模型：测试里注入它，就不连网、不花钱。
from pydantic_ai.models.test import TestModel
# PA: 用 OpenAI 兼容协议连本机 Ollama（和练习 1 的 ChatOpenAI 是同一层意思）。
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

MODEL = os.environ.get("MODEL", "qwen2.5:3b")
OLLAMA_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434/v1")


# ⭐ 核心 1：这个类就是「输出契约」。框架会把它的 JSON schema 交给模型，
#    并在拿到输出后用 Pydantic 校验；校验失败会触发重试。
#    注意最后一句：schema 合法 ≠ 内容真实 —— 它只保证「形状对」。
class AnswerWithConfidence(BaseModel):
    """已校验的数据形状；它不是答案为真的证据。"""
    # Field 约束：最小长度 / 数值范围 / 列表至少一项
    answer: str = Field(min_length=1, description="一句谨慎、非空的回答。")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度，0.0 到 1.0。")
    sources: list[str] = Field(min_length=1, description="至少一个非空的来源标签。")

    # PD: Field 管不了「全是空格」这种情况，所以补一个自定义校验器。
    #     ↔ 练习 1/3 里「节点自己检查输入」，这里是「模型输出自己检查」。
    @field_validator("answer")
    @classmethod
    def answer_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer 不能是空白字符串")
        return value.strip()

    @field_validator("sources")
    @classmethod
    def sources_are_nonempty_labels(cls, value: list[str]) -> list[str]:
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError("sources 里每个来源都必须是非空字符串")
        return [item.strip() for item in value]


# 🟡 看关键：模型接线。OpenAIChatModel + OpenAIProvider(base_url=...) 指向本机 Ollama。
#    ↔ 练习 1：ChatOpenAI(base_url=...)、练习 2：Agent(llm="ollama/qwen2.5:3b") 字符串。
def make_ollama_model() -> OpenAIChatModel:
    """用目前 Pydantic AI 的 OpenAIChatModel 连到本机 Ollama。"""
    return OpenAIChatModel(MODEL, provider=OpenAIProvider(base_url=OLLAMA_BASE, api_key="ollama"))


# ⭐ 核心 2：Agent 的三个关键参数。
#    output_type：要求模型按 AnswerWithConfidence 的形状输出（框架负责转 schema、解析、校验）；
#    retries=2  ：校验失败自动重试，最多 2 次（会多花 token/时间，但能自愈格式错误）；
#    instructions：给模型的额外指令（可理解为这个 Agent 的 system prompt）。
def build_agent(model: Any | None = None) -> Agent:
    """建立会重试两次、并要求 AnswerWithConfidence 的 agent。"""
    chosen = model or make_ollama_model()
    # 两种模式按模型切换（原因见上面 import 处的注释）：
    #   - 官方 TestModel：不支持 NativeOutput（会抛 UserError）→ 用默认 ToolOutput；
    #   - 真模型（Ollama qwen2.5:3b）：默认 ToolOutput / PromptedOutput 都会超重试，
    #     只有 NativeOutput（走 JSON schema 模式）能稳定产出合法结构。
    output_type = AnswerWithConfidence if isinstance(chosen, TestModel) else NativeOutput(AnswerWithConfidence)
    return Agent(
        chosen,
        output_type=output_type,
        retries=2,
        instructions=(
            "返回一个谨慎的 AnswerWithConfidence：answer 非空、confidence 在 0 到 1 之间、"
            "sources 至少一个标签。schema 合法并不代表内容真实。"
        ),
    )


# ⭐ 核心 3：一次 run_sync 就完成「调模型 → 解析 → 校验 →（必要时）重试」。
#    拿到的 .output 已经是 AnswerWithConfidence 实例；下面那段 if 是练习自己的
#    「业务契约」再检查一遍（校验器已挡掉空白/越界，这里防的是将来改 schema 时漏掉）。
def run(question: str, model: Any | None = None) -> AnswerWithConfidence:
    """执行 agent，并在返回前再检查本练习的结果契约。"""
    output = build_agent(model).run_sync(question).output
    if not output.answer or not output.sources or not 0.0 <= output.confidence <= 1.0:
        raise RuntimeError("校验后的输出仍不符合本练习的结果契约。")
    return output


# ⚪ 可跳过：示范用自检。model_dump() 把校验后的模型打印成 dict。
if __name__ == "__main__":
    answer = run("What is the population of Taipei?")
    print(answer.model_dump())
    assert 0.0 <= answer.confidence <= 1.0
    assert answer.answer and answer.sources
