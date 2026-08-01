#pragma once

// DeepSeek Provider。使用方式与注意事项见 DeepSeekProvider 的 Doxygen 注释。

#include <agent/core/result.hpp>
#include <agent/llm/content.hpp>
#include <agent/llm/engine/openai_completions.hpp>
#include <agent/llm/model.hpp>
#include <agent/llm/options.hpp>
#include <agent/llm/stream.hpp>
#include <agent/llm/stream_facade.hpp>
#include <agent/llm/types.hpp>

#include <asio.hpp>

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <string_view>

namespace agent {

namespace detail {

/// @brief DeepSeek 思考行为：thinking 内容走 reasoning_content，须补空字段。
struct DeepSeekThinking {
    static constexpr std::string_view reasoning_field = "reasoning_content";
    static void add_params(nlohmann::json& params, StreamOptions const& opts, ModelView const& model);
    static std::optional<std::string> extract_delta(nlohmann::json const& delta);
    static void finalize_assistant(nlohmann::json& msg, Message const& message);
};

}  // namespace detail

/// @brief DeepSeek Provider。
///
/// 四接口：stream（同步流式）/ complete（同步非流式）/
/// stream_async（异步流式）/ complete_async（异步非流式）。
///
/// @usage
/// ```
/// auto model = *ModelRegistry::find_model("deepseek-v4-flash");
/// DeepSeekProvider deepseek({.name="deepseek", .api_key=KEY, .base_url="https://api.deepseek.com"});
/// auto resp = deepseek.complete(model, ctx, {.reasoning=ThinkingLevel::High});
/// ```
///
/// @note 注意事项：
/// - base_url 为 https://api.deepseek.com（不带 /v1，引擎直接拼 /chat/completions）
/// - max_tokens 请求字段用 max_tokens（第三方兼容字段）
/// - 思考：reasoning 传统一 ThinkingLevel → thinking:{type:"enabled"} + reasoning_effort；
///   思考模式自动移除 temperature/top_p/presence_penalty/frequency_penalty（官方要求）
/// - 工具调用多轮回传时 assistant 自动补 reasoning_content（官方要求，否则 400）
/// - 缓存为官方自动（Context Caching），cache_retention 档位被忽略
/// - 认证：Authorization: Bearer <api_key>
using DeepSeekProvider =
    detail::StreamFacade<detail::OpenAICompletionsEngine<detail::DeepSeekThinking, detail::OpenAICompatibleCompat>>;

}  // namespace agent
