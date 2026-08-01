#pragma once

// OpenAI Completions 协议引擎（共享实现，不属于任何厂商）。
// OpenAICompletionsEngine 被 OpenAI / DeepSeek / 第三方兼容端点（vLLM/Ollama/网关）复用：
// 差异在 ThinkingPolicy（思考行为）+ Compat（字段名字符串值）。
// 厂商实例：providers/openai.hpp（OpenAI 官方）、providers/deepseek.hpp（DeepSeek）等。
// 实现见 src/llm/engine/openai_completions.cpp。

#include <agent/core/result.hpp>
#include <agent/llm/engine/image_support.hpp>
#include <agent/llm/model.hpp>
#include <agent/llm/options.hpp>
#include <agent/llm/stream.hpp>
#include <agent/llm/types.hpp>

#include <asio.hpp>

#include <nlohmann/json.hpp>

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace agent::detail {

/// @brief 第三方 OpenAI 兼容端点协议字段名（字符串值）。
///        DeepSeek / vLLM / Ollama / 自建网关等通用：多数只实现旧字段 max_tokens。
struct OpenAICompatibleCompat {
    static constexpr std::string_view max_tokens_field = "max_tokens";
    static constexpr std::string_view system_role = "system";
    /// session affinity 头开关（对齐 pi 参考实现；默认 false，代理/网关场景可开）。
    /// 触发：cache_retention≠none 且给了 session_id 才发。
    static constexpr bool send_session_affinity = false;
    /// 头格式：openai = session_id + x-client-request-id + x-session-affinity；
    /// openai-nosession = x-client-request-id + x-session-affinity；
    /// openrouter = x-session-id。
    static constexpr std::string_view session_affinity_format = "openai";
};

/// @brief 添加 session affinity 头（对齐 pi 的 createClient 逻辑）。
///        触发条件（cache_retention≠none && session_id）由调用方（引擎）判断，
///        本函数只按格式把对应头追加到 headers。
/// @param headers   目标请求头（追加）
/// @param format    openai / openai-nosession / openrouter
/// @param session_id 会话标识（非空）
void add_session_affinity_headers(std::vector<std::pair<std::string, std::string>>& headers,
                                  std::string_view format, std::string_view session_id);

/// @brief 流式解析状态（parse_chunk 的累积状态，跨 chunk 保留）。
///        供 T0 测试与引擎 stream_async 共用。
struct OpenAIStreamState {
    std::string response_id;              // chunk.id（流内各 chunk 同 id）
    Usage usage;
    StopReason stop_reason = StopReason::Stop;   // finish_reason 映射
    std::string error_message;            // content_filter 等非致命终止的原因
    std::string text;                     // 累积正文
    std::string thinking;                 // 累积思考（TP::extract_delta）
    /// 工具调用按 index 归组：id/name 首次出现时补齐，partial_args 逐段累积。
    struct ToolSlot {
        std::string id;
        std::string name;
        std::string partial_args;
        bool finished = false;
    };
    std::map<int, ToolSlot> tools;
};

/// @brief OpenAI Completions 协议引擎：共享模板实现。
///        build_params / parse_chunk 为协议纯函数（public static，T0 测试直调）；
///        stream_async 为唯一核心协程（HttpStreamReader + SseParser + parse_chunk）。
template<typename ThinkingPolicy, typename Compat>
class OpenAICompletionsEngine {
public:
    using event_type = StreamEvent;
    using result_type = ChatResponse;
    static std::optional<ChatResponse> as_done(StreamEvent const& ev);
    static std::optional<Error> as_error(StreamEvent const& ev);

    explicit OpenAICompletionsEngine(EndpointConfig config);

    /// @brief 构建请求体（协议纯函数）。字段对照官方文档逐条写入，无依据不写。
    ///        Context.tools 存工具名 → 从全局 Tools 注册表 resolve 定义；
    ///        未注册工具名 → Result 错误（NotFound）。
    static Result<nlohmann::json> build_params(ModelView const& model, Context const& ctx, StreamOptions const& opts);

    /// @brief 解析单个流式 chunk → 事件序列（纯函数，T0 可测）。
    ///        返回的事件不含 Done——Done 由 stream_async 在流结束聚合。
    static std::vector<StreamEvent> parse_chunk(OpenAIStreamState& state, nlohmann::json const& chunk);

    asio::awaitable<void> stream_async(ModelView const& model, Context const& ctx, StreamOptions const& opts,
                                       AsyncStream<StreamEvent> sink);

private:
    static nlohmann::json convert_messages(Context const& ctx, bool supports_image);
    static Result<nlohmann::json> convert_tools(std::vector<std::string> const& names);
    static ChatResponse build_response(OpenAIStreamState const& state);
    EndpointConfig config_;
};

}  // namespace agent::detail
