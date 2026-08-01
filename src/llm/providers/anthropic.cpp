// Anthropic Messages 协议引擎实现。
//
// 官方文档依据（URL 约束：实现字段对照文档，无依据不写）：
//   https://docs.anthropic.com/en/api/messages-streaming   （2026-08-01 webfetch 通读）
//   https://docs.anthropic.com/en/docs/build-with-claude/handling-stop-reasons
//   https://platform.claude.com/docs/api-reference/messages
//   官方 SDK（.venv anthropic 0.82.0）事件类型与请求 schema 对照验证：
//     - 认证 x-api-key + anthropic-version: 2023-06-01（SDK default_headers）
//     - 请求必填 max_tokens；tools 用 input_schema；system 顶层 {type:"text"}
//     - 事件按 data 行 JSON 的 type 判别：message_start / message_delta /
//       message_stop / content_block_start / content_block_delta / content_block_stop / ping
//     - content_block_delta 的 delta.type：text_delta / thinking_delta /
//       input_json_delta（partial_json 增量）/ signature_delta（thinking 签名）
//     - message_delta.usage 为累积值（官方文档明示 Warning），缺字段保留现值
//     - usage 无 total_tokens：由 input/output/cache_read/cache_creation 求和
//     - stop_reason 值表：end_turn/max_tokens/stop_sequence/tool_use/pause_turn/refusal/
//       model_context_window_exceeded（文档要求当截断处理）
//   pi 参考实现：tmp/pi/packages/ai/src/api/anthropic-messages.ts

#include <agent/core/http_client.hpp>
#include <agent/llm/providers/anthropic.hpp>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

namespace agent::detail {

namespace {

/// @brief stop_reason → StopReason（官方值表，docs.anthropic.com/.../handling-stop-reasons：
///        end_turn / max_tokens / stop_sequence / tool_use / pause_turn / refusal /
///        model_context_window_exceeded）。
///        pause_turn（服务器工具迭代上限，发回即可继续）与 stop_sequence 视为正常停止；
///        model_context_window_exceeded 文档要求当作截断处理 → Length；
///        refusal / 未知值 → 异常终止（pi 同）。
StopReason map_stop_reason(std::string_view reason, std::string& error_message)
{
    if (reason == "end_turn" || reason == "pause_turn" || reason == "stop_sequence")
        return StopReason::Stop;
    if (reason == "max_tokens" || reason == "model_context_window_exceeded")
        return StopReason::Length;
    if (reason == "tool_use")
        return StopReason::ToolUse;
    error_message = "Anthropic stop_reason: " + std::string(reason);
    return StopReason::Error;
}

/// @brief usage → 统一 Usage（Anthropic 无 total_tokens，由组件求和）。
Usage parse_usage(nlohmann::json const& u)
{
    Usage usage;
    usage.input_tokens = u.value("input_tokens", 0);
    usage.output_tokens = u.value("output_tokens", 0);
    usage.cache_read_tokens = u.value("cache_read_input_tokens", 0);
    usage.cache_write_tokens = u.value("cache_creation_input_tokens", 0);
    usage.total_tokens = usage.input_tokens + usage.output_tokens
        + usage.cache_read_tokens + usage.cache_write_tokens;
    return usage;
}

/// @brief message_delta 的 usage 是累积值（MessageDeltaUsage），缺字段保留现值——
///        输入侧 usage 来自 message_start（官方：input 可能缺），只更新出现的字段。
void apply_usage(Usage& usage, nlohmann::json const& u)
{
    if (u.contains("input_tokens") && u["input_tokens"].is_number())
        usage.input_tokens = u["input_tokens"].get<int>();
    if (u.contains("output_tokens") && u["output_tokens"].is_number())
        usage.output_tokens = u["output_tokens"].get<int>();
    if (u.contains("cache_read_input_tokens") && u["cache_read_input_tokens"].is_number())
        usage.cache_read_tokens = u["cache_read_input_tokens"].get<int>();
    if (u.contains("cache_creation_input_tokens") && u["cache_creation_input_tokens"].is_number())
        usage.cache_write_tokens = u["cache_creation_input_tokens"].get<int>();
    usage.total_tokens = usage.input_tokens + usage.output_tokens
        + usage.cache_read_tokens + usage.cache_write_tokens;
}

/// @brief cache_retention → cache_control（官方缓存机制）：
///        Short → {type:ephemeral}；Long → 加 ttl:"1h"；None → 不挂。
std::optional<nlohmann::json> make_cache_control(CacheRetention retention)
{
    if (retention == CacheRetention::None)
        return std::nullopt;
    nlohmann::json cc{{"type", "ephemeral"}};
    if (retention == CacheRetention::Long)
        cc["ttl"] = "1h";
    return cc;
}

/// @brief tool arguments 增量累积后解析为 JSON；非合法 JSON → 空对象（不崩，错误由上层处理）。
nlohmann::json parse_tool_args(std::string const& partial)
{
    if (partial.empty())
        return nlohmann::json::object();
    nlohmann::json j = nlohmann::json::parse(partial, nullptr, false);
    return j.is_discarded() ? nlohmann::json::object() : j;
}

/// @brief HTTP 状态 → 统一错误码（429→RateLimited / 401,403→AuthError / 其他→ProviderError）。
Errc map_http_status(int status)
{
    if (status == 429) return Errc::RateLimited;
    if (status == 401 || status == 403) return Errc::AuthError;
    return Errc::ProviderError;
}

/// @brief 非 2xx 响应体 → 错误消息（Anthropic：{"type":"error","error":{"message":...}}）。
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

// ───────────────────── 请求体构建 ─────────────────────

nlohmann::json AnthropicMessagesEngine::convert_tools(std::vector<ToolInfo> const& tools,
                                                      nlohmann::json const* cache_control)
{
    nlohmann::json arr = nlohmann::json::array();
    for (std::size_t i = 0; i < tools.size(); ++i) {
        auto const& tool = tools[i];
        nlohmann::json entry{{"name", tool.name}};
        if (!tool.description.empty())
            entry["description"] = tool.description;
        // Anthropic input_schema = 标准 JSON Schema（我们工具的 parameters 已是）。
        entry["input_schema"] = tool.parameters;
        // 缓存挂最后一个 tool（官方推荐位置，文档: Cache tool definitions）。
        if (cache_control && i + 1 == tools.size())
            entry["cache_control"] = *cache_control;
        arr.push_back(std::move(entry));
    }
    return arr;
}

nlohmann::json AnthropicMessagesEngine::convert_messages(Context const& ctx, nlohmann::json const* cache_control,
                                                         bool supports_image)
{
    // 模型不支持图片 → 图片替换为占位符文本（对齐 pi downgradeUnsupportedImages）
    std::vector<Message> input_messages = ctx.messages;
    downgrade_unsupported_images(input_messages, supports_image);

    nlohmann::json messages = nlohmann::json::array();
    std::size_t last_user_index = SIZE_MAX;   // 缓存挂载点：最后 user 消息（含 tool_result 合并）

    for (std::size_t i = 0; i < input_messages.size(); ++i) {
        auto const& m = input_messages[i];
        switch (m.role) {
            case Role::User: {
                // 纯文本 → content 字符串；含图片 → content blocks（text/image）
                std::string text;
                bool has_image = false;
                for (auto const& block : m.content) {
                    if (auto t = std::get_if<Text>(&block))
                        text += t->text;
                    else if (std::get_if<Image>(&block))
                        has_image = true;
                }
                nlohmann::json msg{{"role", "user"}};
                if (!has_image) {
                    msg["content"] = std::move(text);
                } else {
                    nlohmann::json blocks = nlohmann::json::array();
                    for (auto const& block : m.content) {
                        if (auto t = std::get_if<Text>(&block))
                            blocks.push_back({ { "type", "text" }, { "text", t->text } });
                        else if (auto img = std::get_if<Image>(&block))
                            blocks.push_back({ { "type", "image" },
                                               { "source", { { "type", "base64" },
                                                             { "media_type", img->mime_type },
                                                             { "data", img->data } } } });
                    }
                    msg["content"] = std::move(blocks);
                }
                messages.push_back(std::move(msg));
                last_user_index = messages.size() - 1;
                break;
            }
            case Role::Assistant: {
                nlohmann::json blocks = nlohmann::json::array();
                for (auto const& block : m.content) {
                    if (auto t = std::get_if<Text>(&block)) {
                        if (!t->text.empty())
                            blocks.push_back({ { "type", "text" }, { "text", t->text } });
                    } else if (auto th = std::get_if<Thinking>(&block)) {
                        if (th->redacted) {
                            // redacted_thinking：data 是官方 opaque 载荷（签名），原样带回
                            blocks.push_back({ { "type", "redacted_thinking" }, { "data", th->signature } });
                        } else if (!th->signature.empty()) {
                            // 正常 thinking 块：多轮必须带签名（官方要求，否则 400）
                            blocks.push_back({ { "type", "thinking" },
                                               { "thinking", th->text },
                                               { "signature", th->signature } });
                        } else if (!th->text.empty()) {
                            // 无签名（如中断流）→ 降级为文本（对齐 pi：不触发 thinking 校验）
                            blocks.push_back({ { "type", "text" }, { "text", th->text } });
                        }
                    } else if (auto tc = std::get_if<ToolCall>(&block)) {
                        blocks.push_back({ { "type", "tool_use" },
                                           { "id", tc->id }, { "name", tc->name },
                                           { "input", tc->arguments } });
                    }
                }
                if (blocks.empty())
                    continue;
                messages.push_back({ { "role", "assistant" }, { "content", std::move(blocks) } });
                break;
            }
            case Role::ToolResult: {
                // 合并连续 tool_result → 单个 user 消息（Anthropic 要求角色交替，
                // tool_result 必须包在 user 里；对齐 pi 的连续合并）
                nlohmann::json blocks = nlohmann::json::array();
                std::size_t j = i;
                while (j < input_messages.size() && input_messages[j].role == Role::ToolResult) {
                    auto const& msg = input_messages[j];
                    // 收集该消息的图片（支持图片的模型）与 tool_result 块
                    std::vector<Image const*> images;
                    std::vector<ToolResult const*> results;
                    for (auto const& block : msg.content) {
                        if (auto img = std::get_if<Image>(&block))
                            images.push_back(img);
                        else if (auto tr = std::get_if<ToolResult>(&block))
                            results.push_back(tr);
                    }
                    for (std::size_t k = 0; k < results.size(); ++k) {
                        auto const* tr = results[k];
                        nlohmann::json entry{ { "type", "tool_result" },
                                              { "tool_use_id", tr->tool_call_id } };
                        // 图片附到该消息最后一个 tool_result 的 content（官方 tool_result
                        // content 支持 text + image blocks 数组）
                        bool attach_images = (k + 1 == results.size()) && !images.empty();
                        if (attach_images) {
                            nlohmann::json content = nlohmann::json::array();
                            content.push_back({ { "type", "text" }, { "text", tr->output } });
                            for (auto const* img : images)
                                content.push_back({ { "type", "image" },
                                                    { "source", { { "type", "base64" },
                                                                  { "media_type", img->mime_type },
                                                                  { "data", img->data } } } });
                            entry["content"] = std::move(content);
                        } else {
                            entry["content"] = tr->output;
                        }
                        if (tr->is_error)
                            entry["is_error"] = true;
                        blocks.push_back(std::move(entry));
                    }
                    ++j;
                }
                messages.push_back({ { "role", "user" }, { "content", std::move(blocks) } });
                last_user_index = messages.size() - 1;
                i = j - 1;
                break;
            }
        }
    }

    // 缓存挂最后 user 消息的 content 末尾块（官方推荐缓存对话历史的位置）
    if (cache_control && last_user_index != SIZE_MAX) {
        auto& content = messages[last_user_index]["content"];
        if (content.is_string()) {
            std::string text = content.get<std::string>();
            content = nlohmann::json::array({ nlohmann::json{ { "type", "text" }, { "text", std::move(text) },
                                                               { "cache_control", *cache_control } } });
        } else if (content.is_array() && !content.empty()) {
            content.back()["cache_control"] = *cache_control;
        }
    }
    return messages;
}

nlohmann::json AnthropicMessagesEngine::build_params(
    ModelView const& model, Context const& ctx, StreamOptions const& opts)
{
    nlohmann::json params;
    params["model"] = model.id;
    auto cache_control = make_cache_control(opts.cache_retention.value_or(CacheRetention::None));
    params["messages"] = convert_messages(ctx, cache_control ? &*cache_control : nullptr,
                                          model.supports_image_input);
    params["stream"] = true;
    // Anthropic max_tokens 必填：用户不传 → 模型表 max_output_tokens；两者皆无则不写。
    int max_tokens = opts.max_tokens.value_or(model.max_output_tokens);
    if (max_tokens > 0)
        params["max_tokens"] = max_tokens;
    // system 顶层：[{type:"text", text, cache_control?}]
    if (!ctx.system_prompt.empty()) {
        nlohmann::json block{ { "type", "text" }, { "text", ctx.system_prompt } };
        if (cache_control)
            block["cache_control"] = *cache_control;
        params["system"] = nlohmann::json::array({ std::move(block) });
    }
    // thinking：统一 ThinkingLevel → 厂商原生（effort 档 adaptive / budget 档 enabled）
    std::optional<std::string_view> thinking_value;
    bool thinking_on = false;
    if (opts.reasoning.has_value() && model.reasoning) {
        ThinkingLevel level = clamp_thinking_level(model, *opts.reasoning);
        thinking_value = model.thinking_level_map[static_cast<std::size_t>(level)];
        if (thinking_value && *thinking_value != "off")
            thinking_on = true;
    }
    // temperature 与 extended thinking 不兼容（官方文档明示）→ 思考开启时不传
    if (opts.temperature.has_value() && !thinking_on)
        params["temperature"] = *opts.temperature;
    if (thinking_value) {
        std::string_view value = *thinking_value;
        if (value == "off") {
            params["thinking"] = { { "type", "disabled" } };
        } else if (value == "on") {
            // toggle 型启用档：预算思考默认 1024
            params["thinking"] = { { "type", "enabled" }, { "budget_tokens", 1024 },
                                   { "display", "summarized" } };
        } else {
            std::string text(value);
            bool numeric = !text.empty();
            for (char c : text)
                if (c < '0' || c > '9') { numeric = false; break; }
            if (numeric) {
                // budget 档：预算 token 数 → enabled
                params["thinking"] = { { "type", "enabled" },
                                       { "budget_tokens", std::atoi(text.c_str()) },
                                       { "display", "summarized" } };
            } else {
                // effort 档（low..max）→ adaptive + output_config.effort
                params["thinking"] = { { "type", "adaptive" }, { "display", "summarized" } };
                params["output_config"] = { { "effort", std::move(text) } };
            }
        }
    }
    // tools：input_schema，cache_control 挂最后一个 tool
    if (!ctx.tools.empty())
        params["tools"] = convert_tools(ctx.tools, cache_control ? &*cache_control : nullptr);
    // 非公约数透传：原样并入（引擎不解释）
    if (!opts.extra.empty()) {
        for (auto it = opts.extra.begin(); it != opts.extra.end(); ++it)
            params[it.key()] = it.value();
    }
    return params;
}

// ───────────────────── 流解析 ─────────────────────

std::vector<StreamEvent> AnthropicMessagesEngine::parse_chunk(
    AnthropicStreamState& state, nlohmann::json const& chunk)
{
    std::vector<StreamEvent> out;
    if (!chunk.is_object())
        return out;
    std::string type = chunk.value("type", "");

    if (type == "message_start") {
        // 输入侧 usage + 响应 id；最终输出 usage 由 message_delta 累积
        auto const& msg = chunk.value("message", nlohmann::json::object());
        if (msg.contains("id") && msg["id"].is_string())
            state.response_id = msg["id"].get<std::string>();
        if (msg.contains("usage") && msg["usage"].is_object())
            state.usage = parse_usage(msg["usage"]);
        return out;
    }
    if (type == "content_block_start") {
        auto const& block = chunk.value("content_block", nlohmann::json::object());
        int index = chunk.value("index", 0);
        std::string block_type = block.value("type", "");
        state.block_kinds[index] = block_type;
        if (block_type == "tool_use") {
            auto& slot = state.tools[index];
            if (block.contains("id") && block["id"].is_string())
                slot.id = block["id"].get<std::string>();
            if (block.contains("name") && block["name"].is_string())
                slot.name = block["name"].get<std::string>();
        } else if (block_type == "redacted_thinking") {
            if (block.contains("data") && block["data"].is_string())
                state.thinking_signature = block["data"].get<std::string>();
            state.redacted = true;
        }
        return out;
    }
    if (type == "content_block_delta") {
        auto const& delta = chunk.value("delta", nlohmann::json::object());
        std::string delta_type = delta.value("type", "");
        int index = chunk.value("index", 0);
        if (delta_type == "text_delta") {
            std::string text = delta.value("text", "");
            if (!text.empty()) {
                state.text += text;
                out.push_back(StreamEvent{ TextDelta{ std::move(text) } });
            }
        } else if (delta_type == "thinking_delta") {
            std::string text = delta.value("thinking", "");
            if (!text.empty()) {
                state.thinking += text;
                out.push_back(StreamEvent{ ThinkingDelta{ std::move(text) } });
            }
        } else if (delta_type == "input_json_delta") {
            auto& slot = state.tools[index];
            std::string args_delta = delta.value("partial_json", "");
            slot.partial_args += args_delta;
            out.push_back(StreamEvent{ ToolCallDelta{ slot.id, slot.name, std::move(args_delta) } });
        } else if (delta_type == "signature_delta") {
            // thinking 块签名增量：多轮回传必须原样带回
            state.thinking_signature += delta.value("signature", "");
        }
        return out;
    }
    if (type == "content_block_stop") {
        // tool_use 块收尾：完整解析参数
        int index = chunk.value("index", 0);
        auto it = state.tools.find(index);
        if (it != state.tools.end() && !it->second.finished) {
            it->second.finished = true;
            out.push_back(StreamEvent{ ToolCallEnd{ it->second.id, it->second.name,
                                                    parse_tool_args(it->second.partial_args) } });
        }
        return out;
    }
    if (type == "message_delta") {
        auto const& delta = chunk.value("delta", nlohmann::json::object());
        if (delta.contains("stop_reason") && delta["stop_reason"].is_string()) {
            std::string reason = delta["stop_reason"].get<std::string>();
            state.stop_reason = map_stop_reason(reason, state.error_message);
        }
        // message_delta.usage 是累积值（官方 MessageDeltaUsage），缺字段保留
        if (chunk.contains("usage") && chunk["usage"].is_object()) {
            apply_usage(state.usage, chunk["usage"]);
            out.push_back(StreamEvent{ UsageEvent{ state.usage } });
        }
        return out;
    }
    if (type == "error") {
        // 流中错误事件 → 直接 Error 终结
        auto const& err = chunk.value("error", nlohmann::json::object());
        std::string message = err.value("message", "provider error");
        state.stop_reason = StopReason::Error;
        state.error_message = message;
        out.push_back(StreamEvent{ Error{ Errc::ProviderError, std::move(message) } });
        return out;
    }
    // ping / message_stop / 未知类型：无事件（message_stop 由 stream_async 判定收尾）
    return out;
}

ChatResponse AnthropicMessagesEngine::build_response(AnthropicStreamState const& state)
{
    ChatResponse resp;
    resp.response_id = state.response_id;
    resp.stop_reason = state.stop_reason;
    resp.usage = state.usage;
    if (!state.text.empty())
        resp.content.push_back(Text{ state.text });
    if (!state.thinking.empty() || state.redacted) {
        Thinking thinking;
        thinking.text = state.thinking;
        thinking.redacted = state.redacted;
        thinking.signature = state.thinking_signature;
        resp.content.push_back(std::move(thinking));
    }
    for (auto const& [index, slot] : state.tools)
        resp.content.push_back(ToolCall{ slot.id, slot.name, parse_tool_args(slot.partial_args) });
    return resp;
}

// ───────────────────── 终结提取 ─────────────────────

std::optional<ChatResponse> AnthropicMessagesEngine::as_done(StreamEvent const& event)
{
    if (event.type() == StreamEvent::Type::Done)
        return std::get<DoneEvent>(event.data).response;
    return std::nullopt;
}

std::optional<Error> AnthropicMessagesEngine::as_error(StreamEvent const& event)
{
    if (event.type() == StreamEvent::Type::Error)
        return std::get<Error>(event.data);
    return std::nullopt;
}

// ───────────────────── 构造 / 流式主循环 ─────────────────────

AnthropicMessagesEngine::AnthropicMessagesEngine(EndpointConfig config)
    : config_(std::move(config))
{
}

asio::awaitable<void> AnthropicMessagesEngine::stream_async(
    ModelView const& model, Context const& ctx, StreamOptions const& opts,
    AsyncStream<StreamEvent> sink)
{
    auto ex = co_await asio::this_coro::executor;

    // 端点：POST /v1/messages（base_url 不带 /v1，引擎自动拼）
    std::string base = opts.base_url.value_or(config_.base_url);
    std::string url = base;
    if (!url.ends_with('/')) url += '/';
    url += "v1/messages";

    HttpRequestOptions req;
    req.headers = config_.default_headers;
    std::string api_key = opts.api_key.value_or(config_.api_key);
    if (!api_key.empty())
        req.headers.emplace_back("x-api-key", api_key);
    req.headers.emplace_back("anthropic-version", "2023-06-01");   // 官方 API 版本头
    // session affinity 头（对齐 pi：sessionId && sendSessionAffinityHeaders）
    if (opts.cache_retention.has_value() && *opts.cache_retention != CacheRetention::None
        && opts.session_id.has_value() && send_session_affinity) {
        req.headers.emplace_back("x-session-affinity", *opts.session_id);
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

    // 流式主循环：SseParser 切事件 → parse_chunk → 推 sink。
    // Anthropic 无 [DONE] 哨兵：message_stop 事件标记完成；EOF 兜底收尾。
    detail::SseParser parser;
    AnthropicStreamState state;
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
            nlohmann::json j = nlohmann::json::parse(*event, nullptr, false);
            if (j.is_discarded())
                continue;
            if (j.value("type", "") == "message_stop") {   // 官方终止事件
                done = true;
                break;
            }
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

    // 终结：refusal/sensitive 等 → Error；否则 Done 聚合完整响应
    if (state.stop_reason == StopReason::Error) {
        co_await sink.send(StreamEvent{ Error{ Errc::ProviderError, state.error_message } });
    } else {
        co_await sink.send(StreamEvent{ DoneEvent{ build_response(state) } });
    }
    sink.close();
    co_return;
}

}  // namespace agent::detail
