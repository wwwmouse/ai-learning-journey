# -*- coding: utf-8 -*-
"""
Stage 1 練習 5：Error Handling + Retry wrapper — Path A（Ollama 默認、本機免費）。

3 種錯誤情境 + 1 個 retry wrapper：
    1. API key 錯（401 AuthenticationError）→ 不要 retry、直接 raise
    2. Rate limit（429 RateLimitError）→ exponential backoff retry
    3. 網路錯（APIConnectionError）→ exponential backoff retry

跑法：
    pip install -r requirements.txt
    ollama pull gemma4:e4b   # Stage 1+2 預設、CPU-friendly
    ollama serve             # 預設 port 11434
    python starter.py

驗證：
    python test.py   （mock 三種錯誤、不需真的斷網）

想看 Anthropic 版本：
    python starter_anthropic.py   （需 ANTHROPIC_API_KEY）

⚠️ 注意：本機 Ollama 不會真的撞 RateLimitError（沒 quota），所以「情境 2 rate limit」
demo 看不到。但 `python test.py` 全部用 mock、retry 邏輯一樣可以完整驗證。
這恰好是 Ollama path 反而更適合理解 retry pattern——快、免費、可重現。

"""

from __future__ import annotations  # 兼容写法：让类型注解支持 int | float 这类新语法

import os      # 读环境变量（MODEL 模型名可以从环境变量覆盖）
import random  # 生成"抖动"随机数：random.uniform(0, 0.3)
import sys     # 操作 Python 运行时（这里用来重配 stdout 编码，防中文乱码）
import time    # 真睡觉 time.sleep —— 但测试时可以被替换成假 sleep
from typing import Any, Callable  # 类型注解用的工具（只是说明书，不影响运行）

# Windows 终端默认编码可能不支持中文，这段强制用 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 从 openai 库导入"错误类型"和"客户端类"
# except 是按类型接错误的，所以必须先知道有哪些类型可以接
from openai import (
    APIConnectionError,    # 网络层错误：没连上 / 超时 / 断连
    APIStatusError,        # HTTP 层错误（父类）：连上了但返回非 2xx
    AuthenticationError,   # 401：key 错了
    OpenAI,                # 客户端类：OpenAI(base_url=..., api_key=...)
    RateLimitError,        # 429：限流
)

# 模型名：默认 gemma4:e4b，也可以用环境变量 MODEL 覆盖（不用改代码就能换模型）
MODEL = os.environ.get("MODEL", "gemma4:e4b")


# ============================================================
# 核心工具：with_retry —— "自动重试机器人"
# ============================================================

# 规则一：只有这两种错误"值得"重试（白名单）
# 不在白名单里的错误（如 AuthenticationError）会直接穿透抛出去，一次都不重试
RETRIABLE = (APIConnectionError, RateLimitError)
MAX_ATTEMPTS = 4   # 最多试 4 次
BASE_DELAY = 1.0   # 第一次失败的等待起点：1 秒


def with_retry(
    fn: Callable[[], Any],              #  fn：要执行的操作（一个"无参数、有返回值"的函数）
    *,                                  # 星号：从这里开始必须用"名字="传参，防止搞错顺序
    max_attempts: int = MAX_ATTEMPTS,   # 最多试几次（默认 4）
    base_delay: float = BASE_DELAY,     # 等待起点秒数（默认 1.0）
    sleep_fn=time.sleep,                # 等待用的函数（默认真睡觉 time.sleep）
) -> Any:                               # -> Any：返回值类型"任意"（说明书而已）
    """
    把"执行一次操作 + 失败自动重试"这套逻辑封装成工具。

    关键理解：
      - fn 是"函数当参数"：调用者把自己要干的事打包成一个函数交进来
        （没括号 = 函数本身这张名片；有括号 = 真正执行它）
      - sleep_fn 是"依赖注入口"：测试时换成假 sleep，就能瞬间跑完不真等
      - 成功时 return 会立刻结束整个函数 → "成功 = 零重试"由结构保证
      - 只有白名单错误（RETRIABLE）会被接住重试，其他错误直接穿透抛出去
      - 4 轮全败后 raise，把错误交还给调用者（不能假装成功）
    """
    last_exc = None                        # 准备一个"口袋"，用来记住最后一次错误（现在空的）
    for attempt in range(max_attempts):    # attempt 依次 = 0, 1, 2, 3（共 4 轮机会）
        try:                               # 进入护栏：以下代码出错会被接住
            return fn()                    # 执行操作。一旦成功，return 立刻带着结果结束整个函数
        
        except RETRIABLE as e:             # 只接白名单错误（网络错/429）；e 是错误对象
            last_exc = e                   # 把错误装进口袋（为最后 raise 做准备）
            if attempt == max_attempts - 1:  # 是不是最后一轮（attempt == 3）？
                break                      # 是最后一轮：不再等待，跳出循环直接去 raise
            
            # 指数退避公式：base × 2^attempt，再加 0~0.3 秒随机抖动
            # attempt=0 → ~1s，attempt=1 → ~2s，attempt=2 → ~4s
            # 指数增长 = 给服务端喘息机会；抖动 = 防止多个客户端齐刷刷同时重试（惊群效应）
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.3)
            print(f"  ⚠ attempt {attempt+1}/{max_attempts} fail ({type(e).__name__}); retry in {delay:.1f}s")
            sleep_fn(delay)                # 等待 delay 秒（测试时是假 sleep，瞬间过）
    raise last_exc                         # 循环走完 = 4 轮全败：把口袋里的错误抛回给调用者


# ============================================================
# 三个 demo：展示真实世界里错误长什么样
# ============================================================

def demo_bad_key() -> None:
    """情境 1: 故意連到不存在的 Ollama port、看 APIConnectionError（网络层错误）。"""
    print("\n[情境 1] 故意連到不存在的 Ollama port")
    # 65535 是故意写错的（Ollama 真端口是 11434）
    # 注意：OpenAI(...) 只是"填了个地址"，不联网、不报错
    #       真正的网络请求发生在下面的 .create() 调用时
    client = OpenAI(base_url="http://localhost:65535/v1", api_key="ollama")
    try:
        client.chat.completions.create(    # 这里才真的去敲门 → 敲不到 → 抛错
            model=MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )
    except APIConnectionError as e:        # 只接"网络层错误"这一种类型
        print(f"  ✅ 抓到 APIConnectionError: {type(e).__name__}")
        print(f"  💡 production 處理: retry（網路錯通常是 transient）")
    # 如果抛的是其他类型（比如被代理劫持时的 502 InternalServerError）
    # → 这里接不住 → 错误一路往上抛 → 没人接就打印 traceback 崩溃（亲历过）


def demo_with_retry() -> None:
    """情境 2: 包 with_retry 跑一次正常 call、應該第 1 次就成功。"""
    print("\n[情境 2] 正常 call、with_retry 包裝（需要 Ollama 在跑）")
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # 真端口

    def call():  # 把"调一次 API"打包成一个函数 = with_retry 要的 fn
        return client.chat.completions.create(
            model=MODEL,
            max_tokens=30,
            messages=[{"role": "user", "content": "用一個 emoji 回答。"}],
        )

    try:
        msg = with_retry(call)  # 把操作交给"重试机器人"：失败它自己会重试
        print(f"  ✅ 成功、第一次就過: {msg.choices[0].message.content}")
    except APIConnectionError:
        # with_retry 4 次全败后会 raise，这里就是"接球"的人
        print("  ⚠ Ollama 沒在跑（port 11434 不通）。請先 `ollama serve`")
    # 说明：这是"快乐路径"，Ollama 正常时第一次就成功，
    # with_retry 看起来像啥也没干——它真正的考场在 test.py


def demo_too_long_prompt() -> None:
    """情境 3: 故意丟超大 prompt、看 context window（上下文窗口）滿了怎樣。"""
    print("\n[情境 3] Prompt 超過 context window（Ollama 通常會截斷或 raise）")
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    # "重複很多次的 token。" × 200_000（二十万次）≈ 160 万字符 ≈ 100 万+ tokens
    # 而 gemma4:e4b 的 context window 只有 128K tokens
    # → 输入是"桌面"的 8 倍，根本摆不下（"桌面"比喻：窗口 = 模型一次能看见的 token 总量）
    huge_prompt = "重複很多次的 token。" * 200_000  # ~1M tokens

    try:
        client.chat.completions.create(
            model=MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": huge_prompt}],
        )
        # 本地 Ollama 的行为：静默截断（装不下的部分丢掉），不报错
        print("  ⚠ Ollama 沒 raise（可能直接截斷 prompt）。Cloud API 通常會 400")
    except APIStatusError as e:
        # 云端 API 的行为：通常抛 400；APIStatusError 是父类，能接各种状态码
        print(f"  ✅ 抓到 APIStatusError: {e.status_code}")
        print(f"  💡 production 處理: 在 client 端先 count token、超過就拒、別浪費 API call")
    except APIConnectionError:
        print("  ⚠ Ollama 沒在跑")


# ============================================================
# 入口
# ============================================================

# if __name__ == "__main__" 是 Python 经典写法：
#   直接运行本文件（python starter.py）→ __name__ 是 "__main__" → 依次跑三个 demo
#   被其他文件 import（from starter import with_retry）→ __name__ 是 "starter" → 这里不执行
# 这就是 test.py 敢 import 本文件的原因：不会触发三个真实网络的 demo
if __name__ == "__main__":
    demo_bad_key()
    demo_with_retry()
    demo_too_long_prompt()

    # === 自我驗證 ===
    print("\n✅ 練習 5 通過 — 你已了解 3 種錯誤如何 raise、知道何時該 retry 何時該 stop、$0/run")


# ---- 我的运行记录（练习 5 跑通时的真实输出，供日后对照）----
# [情境 1] 故意連到不存在的 Ollama port
#   ✅ 抓到 APIConnectionError: APIConnectionError
#   💡 production 處理: retry（網路錯通常是 transient）

# [情境 2] 正常 call、with_retry 包裝（需要 Ollama 在跑）
#   ✅ 成功、第一次就過: 🤔

# [情境 3] Prompt 超過 context window（Ollama 通常會截斷或 raise）
#   ⚠ Ollama 沒 raise（可能直接截斷 prompt）。Cloud API 通常會 400

# ✅ 練習 5 通過 — 你已了解 3 種錯誤如何 raise、知道何時該 retry 何時該 stop、$0/run
