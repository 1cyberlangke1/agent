#pragma once

// Anthropic Messages Provider（⏳ 未实现，暂不可用）。
// 抽象接口已统一（构造 + 四接口签名与 OpenAI 系一致），引擎实现待后续补齐。

#include <agent/core/result.hpp>
#include <agent/llm/model.hpp>
#include <agent/llm/options.hpp>
#include <agent/llm/stream.hpp>
#include <agent/llm/stream_facade.hpp>
#include <agent/llm/types.hpp>

#include <asio.hpp>

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <utility>

namespace agent {

namespace detail {

/// @brief Anthropic Messages 协议引擎：L5 实现。
class AnthropicMessagesEngine {
public:
    using event_type = StreamEvent;
    using result_type = ChatResponse;
    static std::optional<ChatResponse> as_done(StreamEvent const& ev);
    static std::optional<Error> as_error(StreamEvent const& ev);

    explicit AnthropicMessagesEngine(EndpointConfig config);

    asio::awaitable<void> stream_async(ModelView const& model, Context const& ctx, StreamOptions const& opts,
                                       AsyncStream<StreamEvent> sink);

private:
    nlohmann::json build_params(ModelView const& model, Context const& ctx, StreamOptions const& opts) const;
    EndpointConfig config_;
};

}  // namespace detail

/// @brief Anthropic Messages 协议 Provider。
///
/// @note ⏳ 未实现：引擎实现待后续补齐，当前构造/调用会链接失败，暂不可用。
using AnthropicMessagesProvider = detail::StreamFacade<detail::AnthropicMessagesEngine>;

}  // namespace agent
