#pragma once

// Agnes (sglang 托管) Provider。使用方式与注意事项见 AgnesProvider 的 Doxygen 注释。

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

/// @brief Agnes 思考行为：请求侧走 chat_template_kwargs 布尔开关（OpenAI 兼容扩展），
///        响应侧思考走 reasoning_content（sglang 推理输出字段）。
class AgnesThinking {
public:
    static constexpr std::string_view reasoning_field = "reasoning_content";
    static void add_params(nlohmann::json& params, StreamOptions const& opts, ModelView const& model);
    static std::optional<std::string> extract_delta(nlohmann::json const& delta);
    static void finalize_assistant(nlohmann::json& msg, Message const& message);
};

}  // namespace detail

/// @brief Agnes（apihub.agnes-ai.com）Provider。
///
/// 四接口：stream（同步流式）/ complete（同步非流式）/
/// stream_async（异步流式）/ complete_async（异步非流式）。
///
/// @usage
/// ```
/// ModelRegistry::register_model(RuntimeModel{.id="agnes-2.5-flash", .reasoning=true,
///     .thinking_level_map={...非 Off 档填 "on"...}, .supports_image_input=true, ...});
/// auto model = *ModelRegistry::find_model("agnes-2.5-flash");
/// AgnesProvider agnes({.name="agnes", .api_key=KEY, .base_url="https://apihub.agnes-ai.com/v1"});
/// auto resp = agnes.complete(model, ctx);
/// ```
///
/// @note 注意事项：
/// - base_url 为 https://apihub.agnes-ai.com/v1（端点 POST /chat/completions，OpenAI 兼容）
/// - 思考：reasoning 传统一 ThinkingLevel → chat_template_kwargs.enable_thinking
///   （sglang 托管端点，无 effort 档，只有布尔开关）；模型默认即输出思考，
///   不传 reasoning 则请求体不带 thinking 参数
/// - 思考输出：响应/流式 delta 的 reasoning_content 字段（sglang 推理字段）
/// - 图像：支持 image_url（公开可访问 URL），模型能力由 supports_image_input 承载
/// - 认证：Authorization: Bearer <api_key>
using AgnesProvider =
    detail::StreamFacade<detail::OpenAICompletionsEngine<detail::AgnesThinking, detail::OpenAICompatibleCompat>>;

}  // namespace agent
