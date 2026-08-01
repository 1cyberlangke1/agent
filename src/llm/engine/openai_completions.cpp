// OpenAI Completions 共享引擎实现（模板成员函数，不属于任何厂商）。
// 厂商实例化在 src/llm/providers/{openai,deepseek}.cpp。
//
// 官方文档依据（URL 约束：实现字段对照文档，无依据不写）：
//   OpenAI:  https://platform.openai.com/docs/api-reference/chat
//   DeepSeek: https://api-docs.deepseek.com/api/create-chat-completion
//            https://api-docs.deepseek.com/guides/thinking_mode
//   pi 参考实现：tmp/pi/packages/ai/src/api/openai-completions.ts

#include <agent/core/http_client.hpp>
#include <agent/llm/engine/openai_completions.hpp>
#include <agent/llm/providers/deepseek.hpp>
#include <agent/llm/providers/openai.hpp>

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

namespace agent::detail {

namespace {

/// @brief finish_reason → StopReason（官方值表：stop/length/tool_calls/content_filter/function_call）。
StopReason map_stop_reason(std::string_view reason)
{
    if (reason == "stop" || reason == "end") return StopReason::Stop;
    if (reason == "length") return StopReason::Length;
    if (reason == "tool_calls" || reason == "function_call") return StopReason::ToolUse;
    return StopReason::Error;
}

/// @brief usage chunk → 统一 Usage（input/output/total + 缓存命中读取/写入）。
///        缓存命中读取两种格式都处理（不互踩）：
///        - DeepSeek 官方文档格式：顶层 prompt_cache_hit_tokens / prompt_cache_miss_tokens
///        - OpenAI / DeepSeek 实际响应格式：prompt_tokens_details.cached_tokens
Usage parse_usage(nlohmann::json const& u)
{
    Usage usage;
    usage.input_tokens = u.value("prompt_tokens", 0);
    usage.output_tokens = u.value("completion_tokens", 0);
    usage.total_tokens = u.value("total_tokens", 0);
    usage.cache_read_tokens = u.value("prompt_cache_hit_tokens", 0);
    usage.cache_write_tokens = u.value("prompt_cache_miss_tokens", 0);
    if (u.contains("prompt_tokens_details") && u["prompt_tokens_details"].is_object()) {
        int cached = u["prompt_tokens_details"].value("cached_tokens", 0);
        if (cached > 0 || usage.cache_read_tokens == 0)
            usage.cache_read_tokens = cached;   // 实际格式有值则优先，避免 0 覆盖顶层值
    }
    return usage;
}

/// @brief tool arguments 增量累积后解析为 JSON；非合法 JSON → 空对象（不崩，错误由上层处理）。
nlohmann::json parse_tool_args(std::string const& partial)
{
    if (partial.empty())
        return nlohmann::json::object();
    nlohmann::json j = nlohmann::json::parse(partial, nullptr, false);
    return j.is_discarded() ? nlohmann::json::object() : j;
}

/// @brief prompt_cache_key 截断到 64 字符（官方上限，见 pi clampOpenAIPromptCacheKey）。
std::string clamp_prompt_cache_key(std::string const& key)
{
    return key.size() <= 64 ? key : key.substr(0, 64);
}

/// @brief HTTP 状态 → 统一错误码（429→RateLimited / 401,403→AuthError / 其他→ProviderError）。
Errc map_http_status(int status)
{
    if (status == 429) return Errc::RateLimited;
    if (status == 401 || status == 403) return Errc::AuthError;
    return Errc::ProviderError;
}

/// @brief 非 2xx 响应体 → 错误消息（取 JSON error.message，失败用 body 截断）。
std::string extract_error_message(std::string const& body)
{
    nlohmann::json j = nlohmann::json::parse(body, nullptr, false);
    if (!j.is_discarded() && j.contains("error") && j["error"].is_object()) {
        auto const& e = j["error"];
        if (e.contains("message") && e["message"].is_string())
            return e["message"].get<std::string>();
    }
    if (!body.empty() && body.size() <= 512)
        return body;
    return "HTTP error";
}

}  // namespace

// ───────────────────── session affinity 头（对齐 pi）─────────────────────

void add_session_affinity_headers(std::vector<std::pair<std::string, std::string>>& headers,
                                  std::string_view format, std::string_view session_id)
{
    // 对齐 pi createClient（tmp/pi/.../openai-completions.ts:647-657）：
    //   openai → session_id + x-client-request-id + x-session-affinity
    //   openai-nosession → x-client-request-id + x-session-affinity（无 session_id）
    //   openrouter → x-session-id
    if (format == "openrouter") {
        headers.emplace_back("x-session-id", std::string(session_id));
        return;
    }
    if (format == "openai")
        headers.emplace_back("session_id", std::string(session_id));
    headers.emplace_back("x-client-request-id", std::string(session_id));
    headers.emplace_back("x-session-affinity", std::string(session_id));
}

// ───────────────────── 请求体构建 ─────────────────────

template<typename ThinkingPolicy, typename Compat>
nlohmann::json OpenAICompletionsEngine<ThinkingPolicy, Compat>::convert_messages(Context const& ctx)
{
    nlohmann::json messages = nlohmann::json::array();
    if (!ctx.system_prompt.empty())
        messages.push_back({{"role", Compat::system_role}, {"content", ctx.system_prompt}});
    for (auto const& m : ctx.messages) {
        switch (m.role) {
            case Role::User: {
                std::string text;
                for (auto const& b : m.content)
                    if (auto t = std::get_if<Text>(&b)) text += t->text;
                messages.push_back({{"role", "user"}, {"content", std::move(text)}});
                break;
            }
            case Role::Assistant: {
                nlohmann::json msg{{"role", "assistant"}};
                std::string text;
                nlohmann::json tool_calls = nlohmann::json::array();
                for (auto const& b : m.content) {
                    if (auto t = std::get_if<Text>(&b)) {
                        text += t->text;
                    } else if (auto tc = std::get_if<ToolCall>(&b)) {
                        tool_calls.push_back({{"id", tc->id}, {"type", "function"},
                                              {"function", {{"name", tc->name},
                                                            {"arguments", tc->arguments.dump()}}}});
                    }
                }
                msg["content"] = text.empty() ? nlohmann::json(nullptr) : nlohmann::json(std::move(text));
                if (!tool_calls.empty())
                    msg["tool_calls"] = std::move(tool_calls);
                ThinkingPolicy::finalize_assistant(msg, m);
                messages.push_back(std::move(msg));
                break;
            }
            case Role::ToolResult: {
                for (auto const& b : m.content) {
                    if (auto tr = std::get_if<ToolResult>(&b)) {
                        messages.push_back({{"role", "tool"},
                                            {"tool_call_id", tr->tool_call_id},
                                            {"content", tr->output}});
                    }
                }
                break;
            }
        }
    }
    return messages;
}

template<typename ThinkingPolicy, typename Compat>
nlohmann::json OpenAICompletionsEngine<ThinkingPolicy, Compat>::convert_tools(
    std::vector<ToolInfo> const& tools)
{
    nlohmann::json arr = nlohmann::json::array();
    for (auto const& t : tools) {
        nlohmann::json fn{{"name", t.name}};
        if (!t.description.empty()) fn["description"] = t.description;
        if (!t.parameters.empty()) fn["parameters"] = t.parameters;
        arr.push_back({{"type", "function"}, {"function", std::move(fn)}});
    }
    return arr;
}

template<typename ThinkingPolicy, typename Compat>
nlohmann::json OpenAICompletionsEngine<ThinkingPolicy, Compat>::build_params(
    ModelView const& model, Context const& ctx, StreamOptions const& opts)
{
    nlohmann::json params;
    params["model"] = model.id;
    params["messages"] = convert_messages(ctx);
    params["stream"] = true;
    // 官方文档：include_usage 使流末尾多一个 usage chunk（choices 空）；中断时可能缺失。
    params["stream_options"] = {{"include_usage", true}};
    // max_tokens 字段名随 Compat（OpenAI o 系列 → max_completion_tokens；兼容端点 → max_tokens）。
    // 用户不传 → 不上传（由上游模型默认），绝不代填。
    if (opts.max_tokens.has_value())
        params[Compat::max_tokens_field] = *opts.max_tokens;
    if (opts.temperature.has_value())
        params["temperature"] = *opts.temperature;
    if (!ctx.tools.empty())
        params["tools"] = convert_tools(ctx.tools);
    // 缓存（官方文档 prompt_cache_key / prompt_cache_retention）：
    //   有缓存意图且给了 session_id → prompt_cache_key（截断 64）；
    //   Long → prompt_cache_retention = "24h"（官方最长档）。
    if (opts.cache_retention.has_value() && *opts.cache_retention != CacheRetention::None) {
        if (opts.session_id.has_value())
            params["prompt_cache_key"] = clamp_prompt_cache_key(*opts.session_id);
        if (*opts.cache_retention == CacheRetention::Long)
            params["prompt_cache_retention"] = "24h";
    }
    // thinking 差异（策略层：OpenAI reasoning_effort / DeepSeek thinking+reasoning_effort）
    ThinkingPolicy::add_params(params, opts, model);
    // 非公约数透传：原样并入（引擎不解释）
    if (!opts.extra.empty()) {
        for (auto it = opts.extra.begin(); it != opts.extra.end(); ++it)
            params[it.key()] = it.value();
    }
    return params;
}

// ───────────────────── 流解析 ─────────────────────

template<typename ThinkingPolicy, typename Compat>
std::vector<StreamEvent> OpenAICompletionsEngine<ThinkingPolicy, Compat>::parse_chunk(
    OpenAIStreamState& state, nlohmann::json const& chunk)
{
    std::vector<StreamEvent> out;
    if (!chunk.is_object())
        return out;
    if (chunk.contains("id") && chunk["id"].is_string() && state.response_id.empty())
        state.response_id = chunk["id"].get<std::string>();
    // 流中错误事件（OpenAI：data:{"error":{...}}，无 choices）→ 直接 Error 终结
    if (chunk.contains("error") && chunk["error"].is_object()) {
        auto const& err = chunk["error"];
        std::string message = "provider error";
        if (err.contains("message") && err["message"].is_string())
            message = err["message"].get<std::string>();
        state.stop_reason = StopReason::Error;
        state.error_message = message;
        out.push_back(StreamEvent{ Error{ Errc::ProviderError, message } });
        return out;
    }
    // usage chunk（include_usage 开启时流末尾出现，choices 空数组）
    if (chunk.contains("usage") && !chunk["usage"].is_null()) {
        state.usage = parse_usage(chunk["usage"]);
        out.push_back(StreamEvent{ UsageEvent{ state.usage } });
    }
    if (!chunk.contains("choices") || !chunk["choices"].is_array() || chunk["choices"].empty())
        return out;
    auto const& choice = chunk["choices"][0];
    // finish_reason → StopReason
    if (choice.contains("finish_reason") && !choice["finish_reason"].is_null()) {
        std::string fr = choice["finish_reason"].get<std::string>();
        state.stop_reason = map_stop_reason(fr);
        if (state.stop_reason == StopReason::Error)
            state.error_message = "Provider finish_reason: " + fr;
    }
    if (!choice.contains("delta") || !choice["delta"].is_object())
        return out;
    auto const& delta = choice["delta"];
    // 正文增量
    if (delta.contains("content") && delta["content"].is_string()) {
        std::string text = delta["content"].get<std::string>();
        if (!text.empty()) {
            state.text += text;
            out.push_back(StreamEvent{ TextDelta{ std::move(text) } });
        }
    }
    // 思考增量（策略层：OpenAI reasoning / DeepSeek reasoning_content）
    if (auto thinking = ThinkingPolicy::extract_delta(delta)) {
        state.thinking += *thinking;
        out.push_back(StreamEvent{ ThinkingDelta{ std::move(*thinking) } });
    }
    // 工具调用增量（按 index 归组，arguments 逐段累积）
    if (delta.contains("tool_calls") && delta["tool_calls"].is_array()) {
        for (auto const& tc : delta["tool_calls"]) {
            int idx = tc.value("index", static_cast<int>(state.tools.size()));
            auto& slot = state.tools[idx];
            if (tc.contains("id") && tc["id"].is_string() && slot.id.empty())
                slot.id = tc["id"].get<std::string>();
            nlohmann::json fn = tc.value("function", nlohmann::json::object());
            if (fn.contains("name") && fn["name"].is_string() && slot.name.empty())
                slot.name = fn["name"].get<std::string>();
            std::string args_delta;
            if (fn.contains("arguments") && fn["arguments"].is_string())
                args_delta = fn["arguments"].get<std::string>();
            slot.partial_args += args_delta;
            out.push_back(StreamEvent{ ToolCallDelta{ slot.id, slot.name, std::move(args_delta) } });
        }
    }
    // finish_reason = tool_calls → 各工具调用收尾（ToolCallEnd 带完整解析参数）
    if (choice.contains("finish_reason") && choice["finish_reason"].is_string() &&
        choice["finish_reason"].get<std::string>() == "tool_calls") {
        for (auto& [index, slot] : state.tools) {
            if (slot.finished) continue;
            slot.finished = true;
            out.push_back(StreamEvent{ ToolCallEnd{ slot.id, slot.name, parse_tool_args(slot.partial_args) } });
        }
    }
    return out;
}

template<typename ThinkingPolicy, typename Compat>
ChatResponse OpenAICompletionsEngine<ThinkingPolicy, Compat>::build_response(
    OpenAIStreamState const& state)
{
    ChatResponse resp;
    resp.response_id = state.response_id;
    resp.stop_reason = state.stop_reason;
    resp.usage = state.usage;
    if (!state.text.empty())
        resp.content.push_back(Text{ state.text });
    if (!state.thinking.empty())
        resp.content.push_back(Thinking{ state.thinking });
    for (auto const& [index, slot] : state.tools)
        resp.content.push_back(ToolCall{ slot.id, slot.name, parse_tool_args(slot.partial_args) });
    return resp;
}

// ───────────────────── 终结提取 ─────────────────────

template<typename ThinkingPolicy, typename Compat>
std::optional<ChatResponse> OpenAICompletionsEngine<ThinkingPolicy, Compat>::as_done(
    StreamEvent const& event)
{
    if (event.type() == StreamEvent::Type::Done)
        return std::get<DoneEvent>(event.data).response;
    return std::nullopt;
}

template<typename ThinkingPolicy, typename Compat>
std::optional<Error> OpenAICompletionsEngine<ThinkingPolicy, Compat>::as_error(
    StreamEvent const& event)
{
    if (event.type() == StreamEvent::Type::Error)
        return std::get<Error>(event.data);
    return std::nullopt;
}

// ───────────────────── 构造 / 流式主循环 ─────────────────────

template<typename ThinkingPolicy, typename Compat>
OpenAICompletionsEngine<ThinkingPolicy, Compat>::OpenAICompletionsEngine(EndpointConfig config)
    : config_(std::move(config))
{
}

template<typename ThinkingPolicy, typename Compat>
asio::awaitable<void> OpenAICompletionsEngine<ThinkingPolicy, Compat>::stream_async(
    ModelView const& model, Context const& ctx, StreamOptions const& opts,
    AsyncStream<StreamEvent> sink)
{
    auto ex = co_await asio::this_coro::executor;

    std::string base = opts.base_url.value_or(config_.base_url);
    std::string url = base;
    if (!url.ends_with('/')) url += '/';
    url += "chat/completions";

    // 头合并：default_headers + 认证 + opts.headers − suppress_headers
    HttpRequestOptions req;
    req.headers = config_.default_headers;
    std::string api_key = opts.api_key.value_or(config_.api_key);
    if (!api_key.empty())
        req.headers.emplace_back("Authorization", "Bearer " + api_key);
    // session affinity 头（对齐 pi）：cache_retention≠none 且给了 session_id 且 Compat 开启才发
    if (opts.cache_retention.has_value() && *opts.cache_retention != CacheRetention::None
        && opts.session_id.has_value() && Compat::send_session_affinity) {
        add_session_affinity_headers(req.headers, Compat::session_affinity_format, *opts.session_id);
    }
    for (auto const& [key, value] : opts.headers)
        req.headers.emplace_back(key, value);
    if (!opts.suppress_headers.empty()) {
        req.headers.erase(
            std::remove_if(req.headers.begin(), req.headers.end(),
                           [&](auto const& h) {
                               return std::find(opts.suppress_headers.begin(), opts.suppress_headers.end(),
                                                h.first) != opts.suppress_headers.end();
                           }),
            req.headers.end());
    }
    req.connect_timeout_ms = opts.connect_timeout_ms;
    req.idle_timeout_ms = opts.idle_timeout_ms;
    req.total_timeout_ms = opts.total_timeout_ms;
    req.max_retries = opts.max_retries;
    req.max_retry_delay_ms = opts.max_retry_delay_ms;
    if (opts.cancel)
        req.cancel = opts.cancel->slot();

    nlohmann::json body = build_params(model, ctx, opts);
    auto reader = co_await HttpStreamReader::open(ex, url, body, req);
    if (!reader) {
        co_await sink.send(StreamEvent{ Error{ reader.error().code, reader.error().message } });
        sink.close();
        co_return;
    }
    int status = reader->status();
    if (status / 100 != 2) {
        // 重试耗尽后仍有响应头 → 读错误 body（引擎负责错误映射）
        std::string err_body;
        while (true) {
            auto chunk = co_await reader->next_chunk();
            if (!chunk) break;
            if (!*chunk) break;
            err_body += **chunk;
        }
        co_await sink.send(StreamEvent{
            Error{ map_http_status(status), extract_error_message(err_body) } });
        sink.close();
        co_return;
    }

    // 流式主循环：SseParser 切事件 → parse_chunk → 推 sink
    detail::SseParser parser;
    OpenAIStreamState state;
    bool done = false;
    while (!done) {
        auto chunk = co_await reader->next_chunk();
        if (!chunk) {
            // 首字节后的传输错误（断流/idle 超时/取消）绝不重试
            co_await sink.send(StreamEvent{ Error{ chunk.error().code, chunk.error().message } });
            sink.close();
            co_return;
        }
        if (!*chunk)
            break;   // EOF
        parser.feed(**chunk);
        while (auto event = parser.next_event()) {
            if (*event == "[DONE]") { done = true; break; }
            nlohmann::json j = nlohmann::json::parse(*event, nullptr, false);
            if (j.is_discarded()) continue;
            for (auto& e : parse_chunk(state, j)) {
                bool is_error = e.type() == StreamEvent::Type::Error;
                if (!(co_await sink.send(std::move(e))))
                    co_return;   // 消费端提前关闭
                if (is_error) {   // 流中 error 事件 → 立即终结
                    sink.close();
                    co_return;
                }
            }
        }
    }

    // 终结：content_filter 等 → Error；否则 Done 聚合完整响应
    if (state.stop_reason == StopReason::Error) {
        co_await sink.send(StreamEvent{ Error{ Errc::ProviderError, state.error_message } });
    } else {
        co_await sink.send(StreamEvent{ DoneEvent{ build_response(state) } });
    }
    sink.close();
    co_return;
}

// ───────────────────── 显式实例化 ─────────────────────
// 模板成员定义在本 TU；实例化必须与定义同 TU（否则链接不到）。
// 厂商组合（ThinkingPolicy + Compat）由 providers 类型决定——引擎本身仍共享。
template class OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>;
template class OpenAICompletionsEngine<DeepSeekThinking, OpenAICompatibleCompat>;
template class OpenAICompletionsEngine<OpenAIThinking, OpenAICompatibleCompat>;   // 第三方兼容端点通用

}  // namespace agent::detail
