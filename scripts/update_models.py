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

# budget_tokens 型模型的档位 → 预算 token 表（库定义归一化默认）。
# Anthropic 约束 budget_tokens ≥ 1024 且 < max_tokens（本地 SDK 源码确认）。
# 幂次表：Minimal=1024 … Max=32768。RuntimeModel 可覆盖。
BUDGET_LEVEL_TOKENS = {
    "minimal": "1024",
    "low": "2048",
    "medium": "4096",
    "high": "8192",
    "xhigh": "16384",
    "max": "32768",
}


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
    """生成 7 档 thinking_level_map。值唯一语义，无混用：

    - nullopt：模型无思考能力（reasoning=false）
    - "off"（index 0）：Off 档，关闭思考（引擎不发 thinking 参数）
    - "on"：toggle 型（支持思考但无强度细分）的启用档
    - effort 值（"low".."max"）：effort 型档位强度
    - budget 值（"1024".."32768"）：budget 型档位预算（BUDGET_LEVEL_TOKENS）

    effort 型非 Off 档预填「向下收敛最近有效值」：低于最低支持档 → 最低档值，
    中间空缺 → 向下最近档——引擎取 map[level] 非 Off 恒有值，免判空。
    """
    result = [None] * 7
    if not model.get("reasoning"):
        return result
    result[0] = "off"
    efforts = []
    has_budget = False
    for opt in model.get("reasoning_options") or []:
        if opt.get("type") == "effort":
            efforts.extend(opt.get("values") or [])
        elif opt.get("type") == "budget_tokens":
            has_budget = True

    if efforts:
        supported = {}
        for idx, name in enumerate(THINKING_LEVEL_NAMES, start=1):
            if name in efforts:
                supported[idx] = name
        if supported:
            lowest_idx = min(supported)
            last = None
            for idx in range(1, 7):
                if idx in supported:
                    last = supported[idx]
                    result[idx] = last
                else:
                    # 向下最近支持档；低于最低 → 最低支持档
                    result[idx] = last if last is not None else supported[lowest_idx]
    elif has_budget:
        for idx, name in enumerate(THINKING_LEVEL_NAMES, start=1):
            result[idx] = BUDGET_LEVEL_TOKENS[name]
    else:
        # toggle 型（仅开/关）：非 Off 档全部 "on"
        for idx in range(1, 7):
            result[idx] = "on"
    return result


def build_model(model: dict) -> list:
    """转成 C++ 生成行所需的字段列表（顺序对应 BuiltinModel 声明）。"""
    limit = model.get("limit") or {}
    modalities = model.get("modalities") or {}
    inputs = modalities.get("input") or []
    cost = model.get("cost") or {}
    thinking_map = build_thinking_level_map(model)
    return [
        model["id"],
        limit.get("context", 0),
        limit.get("output", 0),
        bool(model.get("reasoning")),
        "image" in inputs,
        thinking_map,
        cost.get("input", 0),        # 美元/百万 token
        cost.get("output", 0),
        cost.get("cache_read", 0),
        cost.get("cache_write", 0),  # tiers 多档先忽略（RuntimeModel 可覆盖）
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


def cpp_double(value) -> str:
    """C++ 浮点字面量：整数转 .0，浮点原样。"""
    number = float(value)
    if number.is_integer():
        return str(int(number)) + ".0"
    return repr(number)


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
    # 仅直接使用的类型才 include：std::nullopt（optional）、std::span（kAllProviders）；
    # BuiltinModel 的 array/string_view 成员类型由 agent/llm.hpp 提供，不重复引入。
    lines.append("#include <optional>")
    lines.append("#include <span>")
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
            lines.append(f"        .price_input = {cpp_double(f[6])},")
            lines.append(f"        .price_output = {cpp_double(f[7])},")
            lines.append(f"        .price_cache_read = {cpp_double(f[8])},")
            lines.append(f"        .price_cache_write = {cpp_double(f[9])},")
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
