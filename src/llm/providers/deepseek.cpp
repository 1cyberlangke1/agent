// DeepSeek 厂商实现：DeepSeekThinking + 共享引擎实例化（复用不重写）。
//
// 官方文档依据（URL 约束：实现字段对照文档，无依据不写）：
//   DeepSeek:  https://api-docs.deepseek.com/api/create-chat-completion
//             https://api-docs.deepseek.com/guides/thinking_mode
//             https://api-docs.deepseek.com/guides/kv_cache

#include <agent/llm/model.hpp>
#include <agent/llm/providers/deepseek.hpp>

namespace agent::detail {

void DeepSeekThinking::add_params(nlohmann::json& params, StreamOptions const& opts,
                                  ModelView const& model)
{
    if (!model.reasoning)
        return;
    if (!opts.reasoning.has_value())
        return;   // 未指定 → 不发 thinking 参数（DeepSeek 默认非思考模式）
    ThinkingLevel level = clamp_thinking_level(model, *opts.reasoning);
    if (level == ThinkingLevel::Off) {
        params["thinking"] = {{"type", "disabled"}};
        return;
    }
    // 官方文档 thinking_mode：thinking:{type:"enabled"} + reasoning_effort（low/high/max）
    params["thinking"] = {{"type", "enabled"}};
    auto const& value = model.thinking_level_map[static_cast<std::size_t>(level)];
    if (value.has_value() && *value != "off" && *value != "on")
        params["reasoning_effort"] = *value;
    // 官方文档：思考模式不支持 temperature/top_p/presence_penalty/frequency_penalty，
    // 必须移除否则 400。
    params.erase("temperature");
    params.erase("top_p");
    params.erase("presence_penalty");
    params.erase("frequency_penalty");
}

std::optional<std::string> DeepSeekThinking::extract_delta(nlohmann::json const& delta)
{
    // DeepSeek thinking 增量在 delta["reasoning_content"]（与 content 平级）
    if (delta.contains("reasoning_content") && delta["reasoning_content"].is_string())
        return delta["reasoning_content"].get<std::string>();
    return std::nullopt;
}

void DeepSeekThinking::finalize_assistant(nlohmann::json& msg, Message const& message)
{
    // 官方文档 thinking_mode：多轮回传 assistant 消息时须带 reasoning_content（可空串），
    // 否则工具调用场景返回 400。
    std::string thinking;
    for (auto const& b : message.content)
        if (auto t = std::get_if<Thinking>(&b)) thinking += t->text;
    msg["reasoning_content"] = thinking;
}

// DeepSeek 实例的引擎实例化在 engine/openai_completions.cpp（与成员定义同 TU）。

}  // namespace agent::detail
