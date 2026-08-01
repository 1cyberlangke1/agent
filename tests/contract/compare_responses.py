#!/usr/bin/env python3
"""T2 契约层：响应侧镜像对比（官方 SDK 解析结果 ↔ 我们的库解析结果）。

同一份 fixture 字节分别喂给：
  - 官方 SDK（run_mirror.py 落盘 build/contract/sdk_response/<case>_response.json）
  - 我们的库（test_contract_response.cpp 纯函数解析，落盘 build/contract/out/<case>_response.json）
两边解析出的结构化结果（text / thinking / tools / stop_reason / usage）应当一致——
验证「收到相同的包」：同一响应字节，官方 SDK 和我们的解析语义对齐。

用法：python compare_responses.py [out_dir] [sdk_dir]
     out_dir = build/contract/out   （C++ dump，AGENT_CONTRACT=ON 编译跑出的）
     sdk_dir = build/contract/sdk_response （run_mirror.py SDK dump）
"""
import json
import pathlib
import sys


def normalize(v):
    """统一化便于对比（0 与缺失等价、空对象规范化）。"""
    if isinstance(v, dict):
        return {k: normalize(x) for k, x in v.items() if normalize(x) not in (0, "", {})}
    if isinstance(v, list):
        return [normalize(x) for x in v]
    return v


def main() -> None:
    out_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(
        __file__).resolve().parent.parent.parent / "build" / "contract" / "out"
    sdk_dir = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(
        __file__).resolve().parent.parent.parent / "build" / "contract" / "sdk_response"

    failed = 0
    checked = 0
    sdk_files = sorted(sdk_dir.glob("*_response.json"))
    if not sdk_files:
        print("compare_responses: no sdk dumps (run contract_mirror first)")
        sys.exit(2)

    for sdk_file in sdk_files:
        case = sdk_file.name.removesuffix("_response.json")
        ours_file = out_dir / sdk_file.name
        if not ours_file.exists():
            print(f"SKIP {case}: no ours dump at {ours_file}")
            continue
        sdk = normalize(json.loads(sdk_file.read_text()))
        ours = normalize(json.loads(ours_file.read_text()))
        checked += 1

        mismatches = []
        for key in ("text", "thinking", "tools", "stop_reason", "usage"):
            if ours.get(key) != sdk.get(key):
                mismatches.append(key)
        if mismatches:
            failed += 1
            print(f"FAIL {case}: mismatch on {mismatches}")
            for key in mismatches:
                print(f"  ours[{key}] = {json.dumps(ours.get(key), ensure_ascii=False)}")
                print(f"  sdk [{key}] = {json.dumps(sdk.get(key), ensure_ascii=False)}")
        else:
            print(f"PASS {case}: 与官方 SDK 解析结果一致 "
                  f"(text_len={len(ours.get('text', ''))} tools={len(ours.get('tools', []))})")

    if failed:
        print(f"compare_responses: {failed}/{checked} mismatch(es)")
        sys.exit(1)
    print(f"compare_responses: {checked} cases 与官方 SDK 解析结果一致")


if __name__ == "__main__":
    main()
