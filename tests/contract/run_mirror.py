#!/usr/bin/env python3
"""T2 契约层：官方 SDK 镜像消费 fixture（响应侧合规）。

流程（每 case）：
  1. 起 mirror_server 回放 <fixture>.sse（子进程，临时端口）
  2. 官方 SDK 客户端（openai / anthropic）base_url 指向它，发 stream 请求
  3. 断言 SDK 完整消费流：拼接文本 / tool call 名与参数 / stop_reason / usage
     —— 失败 = fixture 不合规（官方 SDK 都解析不了），先修 fixture
  4. SDK 发出的请求体已由 mirror_server 落盘 golden/<case>.json（请求侧参考）

裁判是官方 SDK 本身——手写 fixture 不构成自证循环。

case 清单（每个断言独立）：
  openai:
    text_4    文本流：content 增量可拼接、finish=stop
    tool_1    工具调用：get_weather 参数跨 chunk 累积、finish=tool_calls
  anthropic:
    anthropic_text    文本流：text/thinking 增量、stop_reason=end_turn、usage
    anthropic_tool_1  工具调用：tool_use 块 input 累积、stop_reason=tool_use

用法：python run_mirror.py   （要求 .venv 已装 openai + anthropic）
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

try:
    from anthropic import Anthropic
except ImportError:
    print("anthropic SDK not installed (venv + pip install -r tests/contract/requirements.txt)")
    sys.exit(2)

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
SCRIPT = pathlib.Path(__file__).resolve().parent / "mirror_server.py"
SDK_RESPONSE = REPO / "build" / "contract" / "sdk_response"

# stop 映射：与 C++ 侧（test_contract_response.cpp stop_str）一致，供镜像对比
OPENAI_STOP = {"stop": "stop", "length": "length", "tool_calls": "tool_use",
               "function_call": "tool_use", "content_filter": "error"}
ANTHROPIC_STOP = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_use",
                  "pause_turn": "stop", "stop_sequence": "stop"}


def dump_sdk_response(case: str, text: str, thinking: str, tools: list,
                      stop_reason: str, usage: dict) -> None:
    """SDK 解析 fixture 的结构化结果落盘（与 C++ 侧 *_response.json 同 schema，
    compare_responses.py 据此做两边镜像对比）。"""
    SDK_RESPONSE.mkdir(parents=True, exist_ok=True)
    (SDK_RESPONSE / f"{case}_response.json").write_text(
        json.dumps({"text": text, "thinking": thinking, "tools": tools,
                    "stop_reason": stop_reason, "usage": usage},
                   indent=2, ensure_ascii=False), encoding="utf-8")


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


def run_openai_case(fixture: str, case: str) -> None:
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT), fixture, case, "openai"],
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
        usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage["input"] = getattr(chunk.usage, "prompt_tokens", 0) or 0
                usage["output"] = getattr(chunk.usage, "completion_tokens", 0) or 0
                usage["cache_read"] = getattr(getattr(chunk.usage, "prompt_tokens_details", None),
                                              "cached_tokens", 0) or 0
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

        # 镜像 dump：结构化结果（与 C++ 侧一致）
        tools = [{"id": slot["id"], "name": slot["name"],
                  "arguments": json.loads(slot["args"]) if slot["args"] else {}}
                 for slot in tool_calls.values()]
        dump_sdk_response(case, text, "", tools, OPENAI_STOP.get(finish or "", "error"), usage)
        print(f"PASS {case}: finish={finish} text_len={len(text)} tools={len(tool_calls)} "
              f"usage_in={usage['input']}")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def run_openai_image_case(fixture: str, case: str) -> None:
    """官方 openai SDK 发送含图片（image_url data URL）的请求 → golden 落盘。
    响应是普通文本流，fixture 复用 openai_text_4.sse；请求侧才是多模态验证点。"""
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT), fixture, case, "openai"],
        stdout=subprocess.PIPE, text=True,
    )
    line = proc.stdout.readline().strip()
    port = int(line)
    wait_port(port)

    png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    try:
        client = OpenAI(api_key="contract-test", base_url=f"http://127.0.0.1:{port}/v1")
        stream = client.chat.completions.create(
            model="gpt-4o-vision",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what's in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
                ],
            }],
            stream=True,
        )
        text = ""
        for chunk in stream:
            for choice in chunk.choices:
                if choice.delta.content:
                    text += choice.delta.content
        assert stream is not None
        assert text, "no text content"
        print(f"PASS {case}: image_url 请求已发送，SDK 完整消费响应 text_len={len(text)}")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def run_anthropic_case(fixture: str, case: str) -> None:
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT), fixture, case, "anthropic"],
        stdout=subprocess.PIPE, text=True,
    )
    line = proc.stdout.readline().strip()
    port = int(line)
    wait_port(port)

    try:
        client = Anthropic(api_key="contract-test", base_url=f"http://127.0.0.1:{port}")
        text = ""
        thinking = ""
        tool_uses = []      # {"name", "input"}
        stop_reason = None
        usage = None
        # 高层 MessageStream：SDK 内部累积 input_json_delta，get_final_message 返回完整解析结果
        with client.messages.stream(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            tools=[{
                "name": "get_weather", "description": "查询城市天气",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                                 "required": ["city"]},
            }],
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    text += event.delta.text
                elif event.type == "content_block_delta" and event.delta.type == "thinking_delta":
                    thinking += event.delta.thinking
            final = stream.get_final_message()
            for block in final.content:
                if block.type == "tool_use":
                    tool_uses.append({"id": block.id, "name": block.name, "input": block.input})
            stop_reason = final.stop_reason
            usage = final.usage

        assert usage is not None, "usage missing"
        if case == "anthropic_tool_1":
            assert stop_reason == "tool_use", f"stop_reason={stop_reason!r}"
            assert tool_uses, f"no tool use: {tool_uses!r}"
            assert tool_uses[0]["name"] == "get_weather", f"name={tool_uses[0]['name']!r}"
            assert tool_uses[0]["input"].get("city") == "北京", f"input={tool_uses[0]['input']!r}"
        elif case == "anthropic_text":
            assert stop_reason == "end_turn", f"stop_reason={stop_reason!r}"
            assert text, "no text content"
        else:
            raise AssertionError(f"unknown case {case!r}")

        # 镜像 dump：结构化结果（与 C++ 侧一致）
        tools = [{"id": t["id"], "name": t["name"], "arguments": t["input"]} for t in tool_uses]
        dump_sdk_response(case, text, thinking, tools,
                          ANTHROPIC_STOP.get(stop_reason or "", "error"),
                          {"input": usage.input_tokens or 0, "output": usage.output_tokens or 0,
                           "cache_read": usage.cache_read_input_tokens or 0,
                           "cache_write": usage.cache_creation_input_tokens or 0})
        print(f"PASS {case}: stop={stop_reason} text_len={len(text)} "
              f"thinking_len={len(thinking)} tools={len(tool_uses)} "
              f"usage_in={usage.input_tokens}")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def run_anthropic_image_case(fixture: str, case: str) -> None:
    """官方 anthropic SDK 发送含图片（image content block）的请求 → golden 落盘。
    响应是普通文本流，fixture 复用 anthropic_text.sse；请求侧才是多模态验证点。"""
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT), fixture, case, "anthropic"],
        stdout=subprocess.PIPE, text=True,
    )
    line = proc.stdout.readline().strip()
    port = int(line)
    wait_port(port)

    png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    try:
        client = Anthropic(api_key="contract-test", base_url=f"http://127.0.0.1:{port}")
        with client.messages.stream(
            model="claude-sonnet-4-5",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what's in this image?"},
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/png", "data": png_b64}},
                ],
            }],
            max_tokens=100,
        ) as stream:
            final = stream.get_final_message()
        assert final.content, "no content"
        print(f"PASS {case}: 图片 content block 请求已发送，SDK 完整消费响应")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def main() -> None:
    cases = [
        ("openai_tool_1.sse", "tool_1", "openai", run_openai_case),
        ("openai_text_4.sse", "text_4", "openai", run_openai_case),
        ("openai_text_4.sse", "openai_image_1", "openai", run_openai_image_case),
        ("anthropic_tool_round1.sse", "anthropic_tool_1", "anthropic", run_anthropic_case),
        ("anthropic_text.sse", "anthropic_text", "anthropic", run_anthropic_case),
        ("anthropic_text.sse", "anthropic_image_1", "anthropic", run_anthropic_image_case),
    ]
    failed = 0
    for fixture, case, provider, runner in cases:
        try:
            runner(fixture, case)
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
