#pragma once

// Anthropic Messages Provider。
// 官方文档依据（实现对照文档 / 官方 SDK，无依据不写）：
//   https://docs.anthropic.com/en/api/messages-streaming
//   https://platform.claude.com/docs/api-reference/messages
// 参考实现：tmp/pi/packages/ai/src/api/anthropic-messages.ts
// 事件契约（官方 SSE data 行 JSON，按 type 判别）：
//   message_start / message_delta / message_stop / content_block_start /
//   content_block_delta / content_block_stop / ping / error
// 认证：x-api-key + anthropic-version: 2023-06-01。

#include <agent/core/result.hpp>
#include <agent/llm/engine/image_support.hpp>
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

/// @brief 流式解析状态（parse_chunk 的累积状态，跨 chunk 保留）。
///        供 T0 测试与引擎 stream_async 共用。
struct AnthropicStreamState {
    std::string response_id;              // message_start.message.id
    Usage usage;                          // message_start 输入 + message_delta 累积
    StopReason stop_reason = StopReason::Stop;   // message_delta.delta.stop_reason
    std::string error_message;            // refusal / sensitive 等终止原因
    std::string text;                     // text_delta 累积
    std::string thinking;                 // thinking_delta 累积
    std::string thinking_signature;       // signature_delta 累积（thinking 块签名）
    bool redacted = false;                // redacted_thinking 块
    /// 工具调用按 content_block index 归组：id/name 在 content_block_start 补齐，
    /// partial_args 由 input_json_delta 逐段累积，content_block_stop 收尾。
    struct ToolSlot {
        std::string id;
        std::string name;
        std::string partial_args;
        bool finished = false;
    };
    std::map<int, ToolSlot> tools;
    /// content_block_start 记录的块类型（"text"/"thinking"/"redacted_thinking"/"tool_use"），
    /// content_block_stop 时据此决定是否收尾工具调用。
    std::map<int, std::string> block_kinds;
};

/// @brief Anthropic Messages 协议引擎。
///        build_params / parse_chunk 为协议纯函数（public static，T0 测试直调）；
///        stream_async 为唯一核心协程（HttpStreamReader + SseParser + parse_chunk）。
class AnthropicMessagesEngine {
public:
    using event_type = StreamEvent;
    using result_type = ChatResponse;
    static std::optional<ChatResponse> as_done(StreamEvent const& ev);
    static std::optional<Error> as_error(StreamEvent const& ev);

    explicit AnthropicMessagesEngine(EndpointConfig config);

    /// @brief 构建请求体（协议纯函数）。字段对照官方文档逐条写入，无依据不写。
    ///        Anthropic 必填 max_tokens：用户不传 → 模型表 max_output_tokens。
    ///        thinking 映射：effort 档 → adaptive + output_config.effort；
    ///        budget 档 → enabled + budget_tokens；"off" → disabled；"on" → enabled(1024)。
    ///        缓存：cache_control{type:ephemeral}（Long → ttl:"1h"）挂
    ///        system / 最后一个 tool / 最后一个 user 消息。
    static nlohmann::json build_params(ModelView const& model, Context const& ctx, StreamOptions const& opts);

    /// @brief 解析单个流式事件 → 事件序列（纯函数，T0 可测）。
    ///        返回的事件不含 Done——Done 由 stream_async 在 message_stop/EOF 聚合。
    static std::vector<StreamEvent> parse_chunk(AnthropicStreamState& state, nlohmann::json const& chunk);

    asio::awaitable<void> stream_async(ModelView const& model, Context const& ctx, StreamOptions const& opts,
                                       AsyncStream<StreamEvent> sink);

    /// session affinity 头开关（对齐 pi 的 sendSessionAffinityHeaders；默认 false）。
    /// 触发：cache_retention≠none 且给了 session_id 才发 x-session-affinity。
    static constexpr bool send_session_affinity = false;

private:
    static nlohmann::json convert_messages(Context const& ctx, nlohmann::json const* cache_control,
                                           bool supports_image);
    static nlohmann::json convert_tools(std::vector<ToolInfo> const& tools, nlohmann::json const* cache_control);
    static ChatResponse build_response(AnthropicStreamState const& state);
    EndpointConfig config_;
};

}  // namespace detail

/// @brief Anthropic Messages 协议 Provider。
///
/// 四接口：stream（同步流式）/ complete（同步非流式）/
/// stream_async（异步流式）/ complete_async（异步非流式）。
///
/// @usage
/// ```
/// auto model = *ModelRegistry::find_model("claude-sonnet-4-5");   // 查模型表拿 ModelView
/// AnthropicMessagesProvider anthropic({.name="anthropic", .api_key=KEY, .base_url="https://api.anthropic.com"});
/// for (auto& ev : anthropic.stream(model, ctx, {.reasoning=ThinkingLevel::High})) ...
/// auto resp = anthropic.complete(model, ctx);
/// ```
///
/// @note 注意事项：
/// - base_url 不带 /v1（引擎自动拼 /v1/messages）；认证 x-api-key + anthropic-version
/// - max_tokens 必填：用户不传 → 用模型表 max_output_tokens
/// - 思考：effort 档 → thinking{type:adaptive}+output_config.effort；budget 档 →
///   thinking{type:enabled,budget_tokens}；Off → thinking{type:disabled}；
///   思考开启时 temperature 不传（与 extended thinking 不兼容，官方文档明示）
/// - 缓存：cache_control{type:ephemeral} 挂 system/最后 tool/最后 user 消息；
///   Long → ttl:"1h"
/// - 流式事件：thinking 块多轮回传需带 signature（引擎自动保存，无签名降级为文本）
/// - ping 事件随时可能插入，自动跳过
using AnthropicMessagesProvider = detail::StreamFacade<detail::AnthropicMessagesEngine>;

}  // namespace agent
