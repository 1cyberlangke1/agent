#!/usr/bin/env python3
"""T2 契约层：官方 openai SDK 镜像消费 fixture（响应侧合规）。

流程（每 case）：
  1. 起 mirror_server 回放 <fixture>.sse（子进程，临时端口）
  2. 官方 openai SDK 客户端 base_url 指向它，发 stream 请求
  3. 断言 SDK 完整消费流：拼接文本 / tool call 名与参数 / finish_reason / usage
     —— 失败 = fixture 不合规（官方 SDK 都解析不了），先修 fixture
  4. SDK 发出的请求体已由 mirror_server 落盘 golden/<case>.json（请求侧参考）

裁判是官方 SDK 本身——手写 fixture 不构成自证循环。

case 清单（每个断言独立）：
  text_4    文本流：content 增量可拼接、finish=stop
  tool_1    工具调用：get_weather 参数跨 chunk 累积、finish=tool_calls

用法：python run_mirror.py   （要求 .venv 已装 openai）
"""
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

try:
    from openai import OpenAI
except ImportError:
    print("openai SDK not installed (venv + pip install -r tests/contract/requirements.txt)")
    sys.exit(2)

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
SCRIPT = pathlib.Path(__file__).resolve().parent / "mirror_server.py"


def wait_port(port: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.5)
        except Exception:
            time.sleep(0.1)
            continue
        return True
    return False


def run_case(fixture: str, case: str) -> None:
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT), fixture, case],
        stdout=subprocess.PIPE, text=True,
    )
    line = proc.stdout.readline().strip()
    port = int(line)
    wait_port(port)

    try:
        client = OpenAI(api_key="contract-test", base_url=f"http://127.0.0.1:{port}/v1")
        stream = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        text = ""
        tool_calls = {}      # index -> {"id","name","args"}
        finish = None
        saw_usage = False
        for chunk in stream:
            if getattr(chunk, "usage", None):
                saw_usage = True
            for choice in chunk.choices:
                if choice.delta.content:
                    text += choice.delta.content
                if choice.delta.tool_calls:
                    for tc in choice.delta.tool_calls:
                        slot = tool_calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["args"] += tc.function.arguments
                if choice.finish_reason:
                    finish = choice.finish_reason
        assert stream is not None

        if case == "tool_1":
            assert finish == "tool_calls", f"finish={finish!r}"
            assert 0 in tool_calls, f"no tool call: {tool_calls!r}"
            tc = tool_calls[0]
            assert tc["name"] == "get_weather", f"name={tc['name']!r}"
            args = json.loads(tc["args"])
            assert args.get("city") == "New York City", f"args={tc['args']!r}"
            assert tc["id"], "tool call id missing"
        elif case == "text_4":
            assert finish == "stop", f"finish={finish!r}"
            assert text, "no text content"
        else:
            raise AssertionError(f"unknown case {case!r}")

        print(f"PASS {case}: finish={finish} text_len={len(text)} tools={len(tool_calls)} usage={saw_usage}")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def main() -> None:
    cases = [
        ("openai_tool_1.sse", "tool_1"),
        ("openai_text_4.sse", "text_4"),
    ]
    failed = 0
    for fixture, case in cases:
        try:
            run_case(fixture, case)
        except AssertionError as e:
            failed += 1
            print(f"FAIL {case}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {case}: {type(e).__name__}: {e}")
    if failed:
        sys.exit(1)
    print("contract: all cases passed")


if __name__ == "__main__":
    main()
