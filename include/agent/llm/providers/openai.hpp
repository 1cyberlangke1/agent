#pragma once

// OpenAI 官方 Provider。使用方式与注意事项见 OpenAIProvider 的 Doxygen 注释。

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

/// @brief OpenAI 官方协议字段名（字符串值，非能力位开关）。
///
/// 仅存「字段名 / role 名」这类值差异——位置一样只是名字不同。
struct OpenAICompat {
    static constexpr std::string_view max_tokens_field = "max_completion_tokens";  // OpenAI o 系列需要
    static constexpr std::string_view system_role = "system";                      // 兼容端点普遍接受
    /// session affinity 头开关（默认 false）。见 OpenAICompatibleCompat 同名字段。
    static constexpr bool send_session_affinity = false;
    static constexpr std::string_view session_affinity_format = "openai";
};

/// @brief OpenAI 官方思考行为（模板策略，静态成员函数，编译期绑定）。
struct OpenAIThinking {
    static constexpr std::string_view reasoning_field = "reasoning";
    static void add_params(nlohmann::json& params, StreamOptions const& opts, ModelView const& model);
    static std::optional<std::string> extract_delta(nlohmann::json const& delta);
    static void finalize_assistant(nlohmann::json& msg, Message const& message) { (void)msg; (void)message; }
};

}  // namespace detail

/// @brief OpenAI 官方 Provider。
///
/// 四接口：stream（同步流式）/ complete（同步非流式）/
/// stream_async（异步流式）/ complete_async（异步非流式）。
///
/// @usage
/// ```
/// auto model = *ModelRegistry::find_model("gpt-5.2");   // 查模型表拿 ModelView
/// OpenAIProvider openai({.name="openai", .api_key=KEY, .base_url="https://api.openai.com/v1"});
/// for (auto& ev : openai.stream(model, ctx, {.reasoning=ThinkingLevel::High})) ...
/// auto resp = openai.complete(model, ctx);
/// ```
///
/// @note 注意事项：
/// - base_url 需带 /v1（OpenAI 官方端点路径）
/// - max_tokens 请求字段用 max_completion_tokens（o 系列不兼容旧的 max_tokens）
/// - 思考：StreamOptions.reasoning 传统一 ThinkingLevel → 引擎映射 reasoning_effort
/// - temperature / max_tokens 不传则不上传（让上游模型用默认，绝不代填）
/// - 认证：Authorization: Bearer <api_key>；缓存：prompt_cache_key + retention
///   （OpenAI 官方 / 走代理网关时，可开 session affinity 头提升跨轮缓存命中——
///   见 Compat::send_session_affinity，默认关闭）
using OpenAIProvider =
    detail::StreamFacade<detail::OpenAICompletionsEngine<detail::OpenAIThinking, detail::OpenAICompat>>;

}  // namespace agent
