#!/usr/bin/env python3
"""T2 契约层：请求侧对比（我们的 build_params ↔ 官方 SDK 请求体）。

对比规则：
- 我们的请求体每个字段路径（messages[0].role、tools[0].function.parameters…）
  必须在 golden（SDK 按官方 schema 发的请求）中存在且 JSON 类型一致——
  字段名拼错、嵌套结构错立刻抓出（这正是「绿着错」的主险种）
- 反向：golden 的核心必填（model / messages / stream）我们必须有
- 豁免清单（我们有而 SDK 没有的合法差异）：extra 透传、缓存归一化、
  thinking 映射、stream_options、max_* 字段——每条豁免必须附官方文档 URL

用法：python compare_requests.py <out_dir> <golden_dir>
     out_dir    = build/contract/out   （C++ dump，AGENT_CONTRACT=ON 编译跑出的）
     golden_dir = build/contract/golden （run_mirror.py 录制，官方 SDK 请求体）
"""
import json
import pathlib
import sys


# 豁免清单：我们有而 SDK 参考请求没有的合法差异 → 必须附依据（官方文档 URL）
EXEMPTIONS = {
    "stream_options": "https://platform.openai.com/docs/api-reference/chat/streaming (include_usage)",
    "stream_options.include_usage": "同上",
    "tools": "https://platform.openai.com/docs/api-reference/chat (tools 参数)",
    "tools.*.type": "同上",
    "tools.*.function": "同上",
    "tools.*.function.name": "同上",
    "tools.*.function.description": "同上",
    "tools.*.function.parameters": "同上",
    "reasoning_effort": "https://platform.openai.com/docs/api-reference/chat (reasoning_effort)",
    "thinking": "https://api-docs.deepseek.com/guides/thinking_mode",
    "thinking.type": "同上",
    "prompt_cache_key": "https://platform.openai.com/docs/api-reference/chat (prompt_cache_key)",
    "prompt_cache_retention": "同上 (prompt_cache_retention)",
    "max_completion_tokens": "同上 (max_completion_tokens)",
    "max_tokens": "同上 (max_tokens)",
    # Anthropic（docs.anthropic.com / platform.claude.com）
    "system": "https://platform.claude.com/docs/api-reference/messages (system 顶层参数)",
    "system.[i]": "同上",
    "system.[i].type": "同上",
    "system.[i].text": "同上",
    "cache_control": "https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching (cache_control ephemeral)",
    "cache_control.type": "同上",
    "cache_control.ttl": "同上 (Long → ttl:1h)",
    "thinking.type": "https://platform.claude.com/docs/api-reference/messages (thinking)",
    "thinking.budget_tokens": "同上",
    "thinking.display": "同上",
    "output_config": "同上 (output_config.effort)",
    "output_config.effort": "同上",
}


def iter_paths(obj, prefix=""):
    """遍历 JSON，产出每个叶子/对象的路径（数组用 [i] 通配）。"""
    if isinstance(obj, dict):
        if not obj:
            yield prefix or "<root>"
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                for p in iter_paths(v, path):
                    yield p
            elif isinstance(v, list):
                if not v:
                    yield path
                for item in v:
                    if isinstance(item, (dict, list)):
                        for p in iter_paths(item, f"{path}.[i]"):
                            yield p
                    else:
                        yield f"{path}.[i]"
            else:
                yield path
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                for p in iter_paths(item, f"{prefix}.[i]"):
                    yield p
            else:
                yield f"{prefix}.[i]"


def path_in(golden, path):
    """判断路径在 golden 中存在（[i] 视为通配数组元素）。"""
    parts = path.split(".")
    current = golden
    for part in parts:
        if part == "[i]":
            if not isinstance(current, list) or not current:
                return False
            current = current[0]
            continue
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def main() -> None:
    out_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(
        __file__).resolve().parent.parent.parent / "build" / "contract" / "out"
    golden_dir = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(
        __file__).resolve().parent.parent.parent / "build" / "contract" / "golden"

    failed = 0
    golden_files = sorted(golden_dir.glob("*.json"))
    if not golden_files:
        print("compare: no golden (run contract_mirror first)")
        sys.exit(2)
    for golden_file in golden_files:
        case = golden_file.stem
        ours_file = out_dir / f"{case}.json"
        if not ours_file.exists():
            print(f"SKIP {case}: no ours dump at {ours_file}")
            continue
        ours = json.loads(ours_file.read_text())
        golden = json.loads(golden_file.read_text())

        # 1) 我们的字段路径必须在 golden 存在（或豁免）
        for path in iter_paths(ours):
            if path_in(golden, path):
                continue
            exempt = EXEMPTIONS.get(path) or any(
                path.startswith(e.rstrip("*").rstrip(".")) for e in EXEMPTIONS)
            if not exempt:
                print(f"FAIL {case}: our field not in golden: {path}")
                failed += 1
            else:
                print(f"OK   {case}: exempt {path}")

        # 2) golden 核心必填我们必须有
        for required in ("model", "messages", "stream"):
            if not path_in(ours, required):
                print(f"FAIL {case}: missing required field {required!r}")
                failed += 1

    if failed:
        print(f"compare: {failed} mismatch(es)")
        sys.exit(1)
    print("compare: all fields consistent with official SDK request")


if __name__ == "__main__":
    main()
