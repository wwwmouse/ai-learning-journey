# -*- coding: utf-8 -*-
"""
Stage 1 練習 5 自我驗證 — Path A（Ollama starter.py）。

跑法：
    python test.py

驗證內容：
    - with_retry 對 RETRIABLE 錯誤會 retry
    - with_retry 對 non-retriable（譬如 AuthenticationError）直接 raise、不浪費 retry
    - 超過 max_attempts 後 raise 最後一個 exception
    - sleep 確實被叫（透過 mock sleep_fn）
    - exponential backoff 真的指數增長

Anthropic 版本見 test_anthropic.py。
"""

from __future__ import annotations  # 类型注解兼容写法

import sys  # 重配 stdout 编码（中文输出）
from unittest.mock import MagicMock  # 核心武器：万能假对象
#   MagicMock 能扮演任何东西：
#   MagicMock(return_value="ok")     → 每次被调用都返回 "ok"
#   MagicMock(side_effect=[e, e, x]) → 按剧本依次"抛错、抛错、返回 x"
#   MagicMock()                      → 自动记账：call_count（被调几次）等

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 错误类型：造"货真价实的异常实例"需要它们
from openai import APIConnectionError, AuthenticationError, RateLimitError

# 把被测对象从 starter.py 借过来（只借 with_retry 这个函数）
# 因为有 if __name__ == "__main__" 保护，import 不会触发 starter.py 里的真实网络 demo
from starter import with_retry


# ============================================================
# 三个"造错误"助手：造出货真价实的异常实例
# ============================================================

def _make_connection_error():
    """openai.APIConnectionError 构造时强制要求 request 参数 → 用 MagicMock 假装。"""
    return APIConnectionError(request=MagicMock())


def _make_rate_limit_error():
    """RateLimitError 需要 message + response + body。
    MagicMock(status_code=429) = 伪造一个"假装是 429 的 HTTP 响应"。"""
    return RateLimitError(message="rate limited", response=MagicMock(status_code=429), body=None)


def _make_auth_error():
    """同上，伪造一个 401 响应。"""
    return AuthenticationError(message="bad key", response=MagicMock(status_code=401), body=None)


# ============================================================
# 六个测试 = with_retry 六条承诺的逐条验收
# ============================================================

def test_no_retry_when_success_first_time():
    # 承诺：成功时零开销（不该白费力气重试）。
    fn = MagicMock(return_value="ok")     # 剧本：每次调用都成功返回 "ok"
    sleep = MagicMock()                   # 假 sleep：什么都不做，只记录
    result = with_retry(fn, sleep_fn=sleep)  # 把两个替身塞进被测对象
    assert result == "ok"                 # 结果确实拿回来了
    assert fn.call_count == 1             # fn 只被调 1 次（成功就不该重试）
    assert sleep.call_count == 0          # 一次都没睡
    print("✅ test_no_retry_when_success_first_time")


def test_retry_on_connection_error_then_success():
    # 承诺：网络错会重试，直到成功为止。
    err = _make_connection_error()                    # 造一个连接错误
    fn = MagicMock(side_effect=[err, err, "ok"])      # 剧本：1.抛错、2.抛错、3.成功（模拟网络抖两下又好了的真实情况）
    sleep = MagicMock()
    result = with_retry(fn, max_attempts=4, sleep_fn=sleep)
    assert result == "ok"          # 最终成功拿到了结果
    assert fn.call_count == 3      # fn 被调了 3 次（2 次失败 + 1 次成功）
    assert sleep.call_count == 2   # 两次失败之间各睡了一次，共 2 次
    print("✅ test_retry_on_connection_error_then_success")


def test_retry_on_rate_limit():
    # 承诺：429（限流）会重试。
    # 这条在本地永远演示不了（Ollama 没有配额概念），只能靠 mock 验证。
    err = _make_rate_limit_error()
    fn = MagicMock(side_effect=[err, "ok"])   # 剧本：先 429，再成功
    sleep = MagicMock()
    result = with_retry(fn, sleep_fn=sleep)
    assert result == "ok"
    assert fn.call_count == 2                 # 重试了一次（429 值得等）
    print("✅ test_retry_on_rate_limit")


def test_raise_after_max_attempts():
    # 承诺：4 次全败后放弃，把最后一个错误抛出去（不能假装成功）。
    err = _make_connection_error()
    fn = MagicMock(side_effect=[err, err, err, err])   # 剧本：四次均失败
    sleep = MagicMock()
    try:
        with_retry(fn, max_attempts=4, sleep_fn=sleep)
    except APIConnectionError:                # 期待它把错误抛出来
        assert fn.call_count == 4             # 4 次机会全用光
        assert sleep.call_count == 3          # 4 次尝试之间只睡 3 次
        # 最后一轮不睡 —— 对应 with_retry 里 if attempt == max_attempts - 1: break 那行
        print("✅ test_raise_after_max_attempts")
        return
    # 如果 with_retry 出了 bug、没抛错（假装成功），会走到这里让测试失败
    raise AssertionError("應該 raise APIConnectionError")


def test_no_retry_on_auth_error():
    # 承诺：401（key 错）绝不重试 —— 最核心的规则。
    # 原理：AuthenticationError 不在 RETRIABLE 白名单里，
    # 所以 with_retry 的 except RETRIABLE 接不住它 → 第一次就穿透抛出 → fn 只被调 1 次。
    err = _make_auth_error()
    fn = MagicMock(side_effect=err)      # 剧本：永远抛 401
    sleep = MagicMock()
    try:
        with_retry(fn, sleep_fn=sleep)
    except AuthenticationError:
        assert fn.call_count == 1, "AuthenticationError 不該被 retry"   # 只调了 1 次！
        assert sleep.call_count == 0                                    # 一次都没睡！
        print("✅ test_no_retry_on_auth_error")
        return
    raise AssertionError("應該 raise AuthenticationError")


def test_exponential_backoff_delays():
    # 承诺：退避延迟真的是指数增长（base × 2^attempt）。
    # 技巧：假 sleep 不真等，只把"被要求等多久"记进 delays 列表，事后检查。
    err = _make_connection_error()
    fn = MagicMock(side_effect=[err, err, err, "ok"])   # 剧本：错、错、错、成功
    delays = []
    # lambda d: delays.append(d) = 匿名小函数：收到参数 d 就把它记进 delays 列表
    sleep = MagicMock(side_effect=lambda d: delays.append(d))
    result = with_retry(fn, max_attempts=4, base_delay=1.0, sleep_fn=sleep)
    assert result == "ok"
    # 预期等待序列：attempt 0 后约 1s、attempt 1 后约 2s、attempt 2 后约 4s
    # （< 1.5 / < 2.5 / < 4.5 是因为公式里加了 0~0.3 秒的随机抖动）
    assert 1.0 <= delays[0] < 1.5
    assert 2.0 <= delays[1] < 2.5
    assert 4.0 <= delays[2] < 4.5
    print("✅ test_exponential_backoff_delays")


# ============================================================
# 入口：手动按顺序调用 6 个测试
# 任何一行的 assert 失败会立刻抛 AssertionError 中断
# 所以"一路打印 ✅ 到底 + 🎉" = 全部通过
# ============================================================

if __name__ == "__main__":
    test_no_retry_when_success_first_time()
    test_retry_on_connection_error_then_success()
    test_retry_on_rate_limit()
    test_raise_after_max_attempts()
    test_no_retry_on_auth_error()
    test_exponential_backoff_delays()
    print("\n🎉 全部通過 — Ollama path retry wrapper 邏輯正確（RETRIABLE 才 retry、exponential backoff 有效）")
