#pragma once

// Gemini（gemma / gemini）GenerateContent Provider。使用方式与注意事项见
// GeminiGenerateContentProvider 的 Doxygen 注释；引擎实现细节在 src/llm/providers/gemini.cpp。
//
// 官方文档依据（URL 约束：实现字段对照文档，无依据不写）：
//   https://ai.google.dev/api/generate-content
//   https://ai.google.dev/gemini-api/docs/text-generation

#include <agent/core/result.hpp>
#include <agent/llm/model.hpp>
#include <agent/llm/options.hpp>
#include <agent/llm/stream.hpp>
#include <agent/llm/stream_facade.hpp>
#include <agent/llm/types.hpp>

#include <asio.hpp>

#include <nlohmann/json.hpp>

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace agent {

namespace detail {

/// @brief Gemini 流式解析状态（parse_chunk 的累积状态，跨 chunk 保留）。
struct GeminiStreamState {
    std::string response_id;              // responseId
    Usage usage;
    StopReason stop_reason = StopReason::Stop;   // finishReason 映射
    std::string error_message;
    std::string text;                     // 累积正文
    std::string thinking;                 // 累积思考（thought:true part）
    /// 工具调用按 candidate.index 归组（functionCall 一次到达，无增量）。
    struct ToolSlot {
        std::string id;
        std::string name;
        nlohmann::json args;
        bool finished = false;
    };
    std::map<int, ToolSlot> tools;
};

/// @brief Gemini GenerateContent 协议引擎。
///        build_params / parse_chunk 为协议纯函数（public static，T0 测试直调）；
///        stream_async 为唯一核心协程。
class GeminiGenerateContentEngine {
public:
    using event_type = StreamEvent;
    using result_type = ChatResponse;
    static std::optional<ChatResponse> as_done(StreamEvent const& ev);
    static std::optional<Error> as_error(StreamEvent const& ev);

    explicit GeminiGenerateContentEngine(EndpointConfig config);

    /// @brief 构建请求体（协议纯函数）。
    static nlohmann::json build_params(ModelView const& model, Context const& ctx, StreamOptions const& opts);

    /// @brief 解析单个流式事件 → 事件序列（纯函数，T0 可测）。
    static std::vector<StreamEvent> parse_chunk(GeminiStreamState& state, nlohmann::json const& chunk);

    asio::awaitable<void> stream_async(ModelView const& model, Context const& ctx, StreamOptions const& opts,
                                       AsyncStream<StreamEvent> sink);

private:
    static nlohmann::json convert_contents(Context const& ctx);
    static nlohmann::json convert_tools(std::vector<ToolInfo> const& tools);
    static ChatResponse build_response(GeminiStreamState const& state);
    EndpointConfig config_;
};

}  // namespace detail

/// @brief Gemini GenerateContent Provider（gemma / gemini 模型）。
///
/// 四接口：stream（同步流式）/ complete（同步非流式）/
/// stream_async（异步流式）/ complete_async（异步非流式）。
///
/// @usage
/// ```
/// ModelRegistry::register_model(RuntimeModel{.id="gemma-4-26b-a4b-it", ...});
/// auto model = *ModelRegistry::find_model("gemma-4-26b-a4b-it");
/// GeminiGenerateContentProvider gemini({.name="gemini", .api_key=KEY,
///     .base_url="https://generativelanguage.googleapis.com"});
/// auto resp = gemini.complete(model, ctx);
/// ```
///
/// @note 注意事项：
/// - 认证：x-goog-api-key 头（非 Bearer）
/// - base_url 不带 /v1（引擎拼 /v1beta/models/{model}:streamGenerateContent?alt=sse）
/// - 思考：gemma 系列思考是自动的（thought part 自动输出 → ThinkingDelta）；gemma-4 不支持
///   thinkingBudget 参数（传了 400）——不传 StreamOptions.reasoning 即不触发，
///   但思考内容仍随响应返回；仅支持 thinkingBudget 的模型才传 reasoning
/// - 工具调用 functionCall 一次到达（无增量，直接 ToolCallEnd），多轮回传 functionResponse
/// - temperature / max_tokens 不传则不上传（上游模型默认）
using GeminiGenerateContentProvider = detail::StreamFacade<detail::GeminiGenerateContentEngine>;

}  // namespace agent
