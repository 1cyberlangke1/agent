// OpenAI 官方厂商实现：OpenAIThinking + 共享引擎实例化。
// 引擎模板实现在 engine/openai_completions.cpp，复用不重写。
//
// 官方文档依据：
//   reasoning_effort 值域 none/minimal/low/medium/high/xhigh/max:
//     https://platform.openai.com/docs/api-reference/chat
//   reasoning delta 字段 "reasoning":
//     https://platform.openai.com/docs/guides/reasoning

#include <agent/llm/model.hpp>
#include <agent/llm/providers/openai.hpp>

namespace agent::detail {

void OpenAIThinking::add_params(nlohmann::json& params, StreamOptions const& opts,
                                ModelView const& model)
{
    if (!model.reasoning)
        return;
    if (!opts.reasoning.has_value())
        return;   // 未指定 → 不发 reasoning_effort，模型用默认
    ThinkingLevel level = clamp_thinking_level(model, *opts.reasoning);
    if (level == ThinkingLevel::Off)
        return;   // 模型无思考能力 → 静默关闭
    auto const& value = model.thinking_level_map[static_cast<std::size_t>(level)];
    if (!value.has_value())
        return;
    if (*value == "off")
        return;
    if (*value == "on") {
        // toggle 型：官方无「开思考」的独立参数，用官方值域的中等档表达「开启」
        params["reasoning_effort"] = "medium";
        return;
    }
    params["reasoning_effort"] = *value;   // effort 值（"low".."max"）直接写
}

std::optional<std::string> OpenAIThinking::extract_delta(nlohmann::json const& delta)
{
    // OpenAI 推理模型的 thinking 增量在 delta["reasoning"]
    if (delta.contains("reasoning") && delta["reasoning"].is_string())
        return delta["reasoning"].get<std::string>();
    return std::nullopt;
}

// OpenAI 官方实例的引擎实例化在 engine/openai_completions.cpp（与成员定义同 TU）。

}  // namespace agent::detail
