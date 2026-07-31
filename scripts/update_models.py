#!/usr/bin/env python3
# 模型表生成器：拉取 models.dev/api.json → LLM 过滤 → 生成 include/agent/models/generated.hpp
# 用法：
#   python scripts/update_models.py                       # 默认 4 个 provider
#   python scripts/update_models.py --providers mistral   # 扩展 provider
# 产物提交 git，运行期零网络。仅用标准库，无第三方依赖。

import argparse
import json
import sys
import urllib.request
from pathlib import Path

MODELS_DEV_URL = "https://models.dev/api.json"
DEFAULT_PROVIDERS = ["deepseek", "openai", "anthropic", "google"]

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "include" / "agent" / "models" / "generated.hpp"

# 我们的 ThinkingLevel 枚举按序 0..6 = Off/Minimal/Low/Medium/High/XHigh/Max
THINKING_LEVEL_NAMES = ["minimal", "low", "medium", "high", "xhigh", "max"]


def fetch_catalog() -> dict:
    """拉取 models.dev 全量目录。需要 User-Agent（否则 403）。"""
    req = urllib.request.Request(
        MODELS_DEV_URL,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) agent-model-generator",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def is_llm(model: dict) -> bool:
    """LLM 判定：输出含 text 且不含 audio/video（排除 embedding/tts/视频/纯图像）。"""
    output = (model.get("modalities") or {}).get("output") or []
    return "text" in output and "audio" not in output and "video" not in output


def build_thinking_level_map(model: dict) -> list:
    """从 reasoning_options 生成 7 档 thinking_level_map（None = 该等级不支持）。

    - effort 型：枚举名直接进 map（与 models.dev effort values 同名，pi 同思路）
    - toggle / budget_tokens 型：无细分等级，全 None，由引擎用默认行为
    - Off(索引0) 恒 None：= 引擎不发 thinking 参数即关闭
    """
    result = [None] * 7
    if not model.get("reasoning"):
        return result
    efforts = []
    for opt in model.get("reasoning_options") or []:
        if opt.get("type") == "effort":
            efforts.extend(opt.get("values") or [])
    for idx, name in enumerate(THINKING_LEVEL_NAMES, start=1):
        if name in efforts:
            result[idx] = name
    return result


def build_model(model: dict) -> list:
    """转成 C++ 生成行所需的字段列表（顺序对应 BuiltinModel 声明）。"""
    limit = model.get("limit") or {}
    modalities = model.get("modalities") or {}
    inputs = modalities.get("input") or []
    thinking_map = build_thinking_level_map(model)
    return [
        model["id"],
        limit.get("context", 0),
        limit.get("output", 0),
        bool(model.get("reasoning")),
        "image" in inputs,
        thinking_map,
    ]


def cpp_string(value: str) -> str:
    """C++ 字符串字面量：转义引号/反斜杠/换行。"""
    out = []
    for ch in value:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def cpp_bool(value: bool) -> str:
    return "true" if value else "false"


def cpp_thinking_map(thinking_map: list) -> str:
    """std::array<std::optional<std::string_view>, 7> 聚合初始化。"""
    parts = []
    for v in thinking_map:
        if v is None:
            parts.append("std::nullopt")
        else:
            parts.append(cpp_string(v))
    return "{ " + ", ".join(parts) + " }"


def generate_header(providers: list, catalog: dict, generated_at: str) -> str:
    lines = []
    lines.append("// 自动生成：scripts/update_models.py —— 勿手改。")
    lines.append("// 数据源：https://models.dev/api.json（LLM 过滤：output 含 text 且不含 audio/video）")
    lines.append(f"// 快照时间（UTC）：{generated_at}     provider：{', '.join(providers)}")
    lines.append("// 重新生成：python scripts/update_models.py")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <array>")
    lines.append("#include <optional>")
    lines.append("#include <string_view>")
    lines.append("#include <agent/llm.hpp>")
    lines.append("")
    lines.append("namespace agent::detail {")
    lines.append("")
    lines.append("inline constexpr BuiltinModel kGeneratedModels[] = {")

    for provider in providers:
        provider_data = catalog.get(provider) or {}
        models = provider_data.get("models") or {}
        model_ids = sorted(models.keys())
        llm_ids = [mid for mid in model_ids if is_llm(models[mid])]
        lines.append(f"    // ── {provider}（{len(llm_ids)} 个 LLM）──")
        for mid in llm_ids:
            m = models[mid]
            f = build_model(m)
            lines.append("    {")
            lines.append(f"        .id = {cpp_string(f[0])},")
            lines.append(f"        .context_window = {f[1]},")
            lines.append(f"        .max_output_tokens = {f[2]},")
            lines.append(f"        .reasoning = {cpp_bool(f[3])},")
            lines.append(f"        .supports_image_input = {cpp_bool(f[4])},")
            lines.append(f"        .thinking_level_map = {cpp_thinking_map(f[5])},")
            lines.append("        .thinking_field = {},")
            lines.append("    },")

    lines.append("};")
    lines.append("")

    # kAllProviders 汇总：provider 名 → 子表。模型按 provider 分组，
    # 与 BuiltinModel 数组同序，用 span 切片（编译期长度由生成器保证）。
    lines.append("inline constexpr BuiltinProviderTable kAllProviders[] = {")
    offset = 0
    for provider in providers:
        provider_data = catalog.get(provider) or {}
        models = provider_data.get("models") or {}
        count = sum(1 for mid in models if is_llm(models[mid]))
        lines.append(f'    {{ "{provider}", std::span<const BuiltinModel>(&kGeneratedModels[{offset}], {count}) }},')
        offset += count
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace agent::detail")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model table from models.dev")
    parser.add_argument(
        "--providers",
        nargs="*",
        default=DEFAULT_PROVIDERS,
        help="provider 键列表（models.dev 顶层键），默认: " + " ".join(DEFAULT_PROVIDERS),
    )
    args = parser.parse_args()

    providers = args.providers or DEFAULT_PROVIDERS
    print(f"拉取 {MODELS_DEV_URL} ...")
    catalog = fetch_catalog()
    for provider in providers:
        if provider not in catalog:
            print(f"警告：provider '{provider}' 不在 models.dev 目录中，跳过", file=sys.stderr)
    providers = [p for p in providers if p in catalog]

    total = 0
    for provider in providers:
        models = (catalog[provider].get("models") or {})
        count = sum(1 for m in models.values() if is_llm(m))
        total += count
        print(f"  {provider}: {count} 个 LLM 模型")

    generated_at = __import__("datetime").datetime.now(timezone_utc()).strftime("%Y-%m-%d %H:%M:%S")
    header = generate_header(providers, catalog, generated_at)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(header, encoding="utf-8")
    print(f"写入 {OUTPUT_PATH}（共 {total} 模型）")


def timezone_utc():
    from datetime import timezone

    return timezone.utc


if __name__ == "__main__":
    main()
