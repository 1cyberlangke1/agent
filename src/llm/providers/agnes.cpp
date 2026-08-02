// Agnes (sglang 托管) 厂商实现：AgnesThinking + 共享引擎实例化。
// 引擎模板实现在 engine/openai_completions.cpp，复用不重写。
//
// 官方文档依据（URL 约束：实现字段对照文档，无依据不写）：
//   https://www.agnes-ai.com/zh-Hans/docs/agnes-25-flash
//   请求侧思考开关：chat_template_kwargs.enable_thinking（OpenAI 兼容扩展参数）
//   响应侧思考字段：reasoning_content（sglang 推理输出字段，与 DeepSeek 同构）
//   图像：messages[].content[].image_url（公开可访问 URL）

#include <agent/llm/model.hpp>
#include <agent/llm/providers/agnes.hpp>

namespace agent::detail {

void AgnesThinking::add_params(nlohmann::json& params, StreamOptions const& opts,
                               ModelView const& model)
{
    if (!model.reasoning)
        return;
    if (!opts.reasoning.has_value())
        return;   // 未指定 → 不发（Agnes 默认输出思考，不打扰默认行为）
    ThinkingLevel level = clamp_thinking_level(model, *opts.reasoning);
    bool enabled = level != ThinkingLevel::Off;
    // 官方文档 Thinking 模式：chat_template_kwargs.enable_thinking（布尔开关，无档位）
    params["chat_template_kwargs"] = { { "enable_thinking", enabled } };
}

std::optional<std::string> AgnesThinking::extract_delta(nlohmann::json const& delta)
{
    // Agnes (sglang) thinking 增量在 delta["reasoning_content"]（与 content 平级）
    if (delta.contains("reasoning_content") && delta["reasoning_content"].is_string())
        return delta["reasoning_content"].get<std::string>();
    return std::nullopt;
}

void AgnesThinking::finalize_assistant(nlohmann::json& msg, Message const& message)
{
    // 多轮回传 assistant 消息带 reasoning_content（sglang 与 DeepSeek 同构的思考回传约定）
    std::string thinking;
    for (auto const& b : message.content)
        if (auto t = std::get_if<Thinking>(&b)) thinking += t->text;
    msg["reasoning_content"] = thinking;
}

// Agnes 实例的引擎实例化在 engine/openai_completions.cpp（与成员定义同 TU）。

}  // namespace agent::detail
