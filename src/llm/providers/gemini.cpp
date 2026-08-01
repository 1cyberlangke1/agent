// Gemini GenerateContent 引擎实现。
//
// 官方文档依据（URL 约束：实现字段对照文档，无依据不写）：
//   https://ai.google.dev/api/generate-content   （curl 拉取已验证可访问，2026-08-01）
//   https://ai.google.dev/gemini-api/docs/text-generation
//   真实 API 探测（gemma-4-26b-a4b-it，2026-08-01）：
//     - 流式端点 `:streamGenerateContent?alt=sse`（官方文档 Method: models.streamGenerateContent）
//     - 认证 `x-goog-api-key` 头（非 Bearer）
//     - 流式事件 data: {"candidates":[{"content":{"parts":[{"text":"...","thought":true}],
//       "role":"model"},"index":0}],"usageMetadata":{...},"responseId":"..."}
//     - 思考输出在 thought:true 的 text part（官方 Part.thought）；thinking 预算走
//       generationConfig.thinkingConfig.thinkingBudget（官方 ThinkingConfig.thinkingBudget）
//     - 工具调用一次到达：part.functionCall = {"name","args","id"}（官方 FunctionCall.Name/Args）
//     - UsageMetadata：promptTokenCount / candidatesTokenCount / totalTokenCount /
//       cachedContentTokenCount / thoughtsTokenCount（官方 UsageMetadata 字段）

#include <agent/core/http_client.hpp>
#include <agent/llm/providers/gemini.hpp>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace agent::detail {

namespace {

/// @brief finishReason → StopReason（Gemini 官方值表）。
StopReason map_finish_reason(std::string_view reason)
{
    if (reason == "STOP") return StopReason::Stop;
    if (reason == "MAX_TOKENS") return StopReason::Length;
    if (reason == "SAFETY" || reason == "RECITATION" || reason == "LANGUAGE" ||
        reason == "IMAGE_SAFETY" || reason == "PROHIBITED_CONTENT" ||
        reason == "BLOCKLIST" || reason == "MALFORMED_FUNCTION_CALL" || reason == "OTHER")
        return StopReason::Error;
    return StopReason::Error;
}

/// @brief usageMetadata → 统一 Usage（prompt/candidates/total + 缓存命中读取）。
Usage parse_usage(nlohmann::json const& u)
{
    Usage usage;
    usage.input_tokens = u.value("promptTokenCount", 0);
    usage.output_tokens = u.value("candidatesTokenCount", 0);
    usage.total_tokens = u.value("totalTokenCount", 0);
    usage.cache_read_tokens = u.value("cachedContentTokenCount", 0);
    if (usage.total_tokens > 0 && usage.output_tokens == 0)
        usage.output_tokens = usage.total_tokens - usage.input_tokens;
    return usage;
}

/// @brief HTTP 状态 → 统一错误码（Gemini 429/403/其他）。
Errc map_http_status(int status)
{
    if (status == 429) return Errc::RateLimited;
    if (status == 401 || status == 403) return Errc::AuthError;
    return Errc::ProviderError;
}

/// @brief 错误 body → 消息（Gemini：{"error":{"message":...}}）。
std::string extract_error_message(std::string const& body)
{
    nlohmann::json j = nlohmann::json::parse(body, nullptr, false);
    if (!j.is_discarded() && j.contains("error") && j["error"].is_object()) {
        auto const& e = j["error"];
        if (e.contains("message") && e["message"].is_string())
            return e["message"].get<std::string>();
    }
    return body.empty() ? "HTTP error" : (body.size() <= 512 ? body : "HTTP error");
}

}  // namespace

// ───────────────────── 请求体构建 ─────────────────────

nlohmann::json GeminiGenerateContentEngine::convert_tools(std::vector<ToolInfo> const& tools)
{
    nlohmann::json decls = nlohmann::json::array();
    for (auto const& t : tools) {
        nlohmann::json fn{{"name", t.name}};
        if (!t.description.empty()) fn["description"] = t.description;
        if (!t.parameters.empty()) fn["parameters"] = t.parameters;
        decls.push_back(std::move(fn));
    }
    // 官方 tools 结构：tools: [{functionDeclarations: [...]}]。
    // 注意 nlohmann initializer `json::array({ {"functionDeclarations", decls} })` 会歧义成
    // `[["functionDeclarations",[...]]]`（数组而非对象）——须显式构造 object 再 push。
    nlohmann::json tools_arr = nlohmann::json::array();
    nlohmann::json entry;
    entry["functionDeclarations"] = std::move(decls);
    tools_arr.push_back(std::move(entry));
    return tools_arr;
}

nlohmann::json GeminiGenerateContentEngine::convert_contents(Context const& ctx)
{
    // Gemini 的 system 走顶层 systemInstruction（build_params），contents 不含 system。
    nlohmann::json contents = nlohmann::json::array();
    // functionResponse part 需要工具名：从 assistant 的 tool_calls 建立 id → name 映射
    std::unordered_map<std::string, std::string> call_names;
    for (auto const& m : ctx.messages) {
        if (m.role != Role::Assistant) continue;
        for (auto const& b : m.content)
            if (auto tc = std::get_if<ToolCall>(&b)) call_names[tc->id] = tc->name;
    }

    for (auto const& m : ctx.messages) {
        switch (m.role) {
            case Role::User: {
                nlohmann::json parts = nlohmann::json::array();
                for (auto const& b : m.content)
                    if (auto t = std::get_if<Text>(&b)) parts.push_back({ { "text", t->text } });
                if (!parts.empty())
                    contents.push_back({ { "role", "user" }, { "parts", std::move(parts) } });
                break;
            }
            case Role::Assistant: {
                nlohmann::json parts = nlohmann::json::array();
                for (auto const& b : m.content) {
                    if (auto t = std::get_if<Text>(&b)) {
                        parts.push_back({ { "text", t->text } });
                    } else if (auto tc = std::get_if<ToolCall>(&b)) {
                        parts.push_back({ { "functionCall",
                                            { { "name", tc->name }, { "args", tc->arguments } } } });
                    }
                }
                if (!parts.empty())
                    contents.push_back({ { "role", "model" }, { "parts", std::move(parts) } });
                break;
            }
            case Role::ToolResult: {
                for (auto const& b : m.content) {
                    if (auto tr = std::get_if<ToolResult>(&b)) {
                        auto it = call_names.find(tr->tool_call_id);
                        std::string name = it != call_names.end() ? it->second : "unknown";
                        // 显式构造避免 nlohmann initializer 把 parts 歧义成数组
                        nlohmann::json part;
                        part["functionResponse"] = nlohmann::json{
                            { "name", name },
                            { "response", nlohmann::json{ { "output", tr->output } } },
                        };
                        nlohmann::json parts = nlohmann::json::array();
                        parts.push_back(std::move(part));
                        contents.push_back({ { "role", "function" }, { "parts", std::move(parts) } });
                    }
                }
                break;
            }
        }
    }
    return contents;
}

nlohmann::json GeminiGenerateContentEngine::build_params(
    ModelView const& model, Context const& ctx, StreamOptions const& opts)
{
    nlohmann::json params;
    if (!ctx.system_prompt.empty())
        params["systemInstruction"] = { { "parts", { { { "text", ctx.system_prompt } } } } };
    params["contents"] = convert_contents(ctx);
    if (!ctx.tools.empty())
        params["tools"] = convert_tools(ctx.tools);

    nlohmann::json gc;
    if (opts.temperature.has_value())
        gc["temperature"] = *opts.temperature;
    if (opts.max_tokens.has_value())
        gc["maxOutputTokens"] = *opts.max_tokens;
    // thinking：map 档位值（budget 数字字符串）→ thinkingConfig.thinkingBudget（token 预算）
    if (opts.reasoning.has_value() && model.reasoning) {
        ThinkingLevel level = clamp_thinking_level(model, *opts.reasoning);
        auto const& value = model.thinking_level_map[static_cast<std::size_t>(level)];
        if (value.has_value() && *value != "off" && *value != "on") {
            std::string num(*value);
            char* end = nullptr;
            long budget = std::strtol(num.c_str(), &end, 10);
            if (end && *end == '\0' && budget > 0)
                gc["thinkingConfig"] = { { "thinkingBudget", budget } };
        }
    }
    if (!gc.empty())
        params["generationConfig"] = std::move(gc);
    if (!opts.extra.empty()) {
        for (auto it = opts.extra.begin(); it != opts.extra.end(); ++it)
            params[it.key()] = it.value();
    }
    return params;
}

// ───────────────────── 流解析 ─────────────────────

std::vector<StreamEvent> GeminiGenerateContentEngine::parse_chunk(
    GeminiStreamState& state, nlohmann::json const& chunk)
{
    std::vector<StreamEvent> out;
    if (!chunk.is_object())
        return out;
    if (chunk.contains("responseId") && chunk["responseId"].is_string())
        state.response_id = chunk["responseId"].get<std::string>();
    if (chunk.contains("usageMetadata") && chunk["usageMetadata"].is_object())
        state.usage = parse_usage(chunk["usageMetadata"]);
    if (!chunk.contains("candidates") || !chunk["candidates"].is_array() || chunk["candidates"].empty())
        return out;

    for (auto const& cand : chunk["candidates"]) {
        int index = cand.value("index", 0);
        if (cand.contains("finishReason") && cand["finishReason"].is_string()) {
            std::string fr = cand["finishReason"].get<std::string>();
            state.stop_reason = map_finish_reason(fr);
            if (state.stop_reason == StopReason::Error)
                state.error_message = "Gemini finishReason: " + fr;
        }
        if (!cand.contains("content") || !cand["content"].is_object())
            continue;
        auto const& parts = cand["content"].value("parts", nlohmann::json::array());
        if (!parts.is_array())
            continue;
        for (auto const& part : parts) {
            if (part.contains("text") && part["text"].is_string()) {
                std::string text = part["text"].get<std::string>();
                if (text.empty()) continue;
                bool thought = part.value("thought", false);
                if (thought) {
                    state.thinking += text;
                    out.push_back(StreamEvent{ ThinkingDelta{ std::move(text) } });
                } else {
                    state.text += text;
                    out.push_back(StreamEvent{ TextDelta{ std::move(text) } });
                }
            } else if (part.contains("functionCall") && part["functionCall"].is_object()) {
                auto const& fc = part["functionCall"];
                auto& slot = state.tools[index];
                if (fc.contains("name") && fc["name"].is_string())
                    slot.name = fc["name"].get<std::string>();
                if (fc.contains("id") && fc["id"].is_string())
                    slot.id = fc["id"].get<std::string>();
                slot.args = fc.value("args", nlohmann::json::object());
                if (!slot.finished) {   // functionCall 一次到达，无增量
                    slot.finished = true;
                    out.push_back(StreamEvent{ ToolCallEnd{ slot.id, slot.name, slot.args } });
                }
            }
        }
    }
    return out;
}

ChatResponse GeminiGenerateContentEngine::build_response(GeminiStreamState const& state)
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
        resp.content.push_back(ToolCall{ slot.id, slot.name, slot.args });
    return resp;
}

// ───────────────────── 终结提取 ─────────────────────

std::optional<ChatResponse> GeminiGenerateContentEngine::as_done(StreamEvent const& event)
{
    if (event.type() == StreamEvent::Type::Done)
        return std::get<DoneEvent>(event.data).response;
    return std::nullopt;
}

std::optional<Error> GeminiGenerateContentEngine::as_error(StreamEvent const& event)
{
    if (event.type() == StreamEvent::Type::Error)
        return std::get<Error>(event.data);
    return std::nullopt;
}

// ───────────────────── 构造 / 流式主循环 ─────────────────────

GeminiGenerateContentEngine::GeminiGenerateContentEngine(EndpointConfig config)
    : config_(std::move(config))
{
}

asio::awaitable<void> GeminiGenerateContentEngine::stream_async(
    ModelView const& model, Context const& ctx, StreamOptions const& opts,
    AsyncStream<StreamEvent> sink)
{
    auto ex = co_await asio::this_coro::executor;

    // 端点：/v1beta/models/{model}:streamGenerateContent?alt=sse（alt=sse 必须显式加）
    std::string base = opts.base_url.value_or(config_.base_url);
    std::string url = base;
    if (!url.ends_with('/')) url += '/';
    url += "v1beta/models/" + std::string(model.id) + ":streamGenerateContent?alt=sse";

    HttpRequestOptions req;
    req.headers = config_.default_headers;
    std::string api_key = opts.api_key.value_or(config_.api_key);
    if (!api_key.empty())
        req.headers.emplace_back("x-goog-api-key", api_key);   // Gemini 认证非 Bearer
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
        std::string err_body;
        while (true) {
            auto chunk = co_await reader->next_chunk();
            if (!chunk) break;
            if (!*chunk) break;
            err_body += **chunk;
        }
        co_await sink.send(StreamEvent{ Error{ map_http_status(status), extract_error_message(err_body) } });
        sink.close();
        co_return;
    }

    detail::SseParser parser;
    GeminiStreamState state;
    while (true) {
        auto chunk = co_await reader->next_chunk();
        if (!chunk) {
            co_await sink.send(StreamEvent{ Error{ chunk.error().code, chunk.error().message } });
            sink.close();
            co_return;
        }
        if (!*chunk)
            break;   // EOF
        parser.feed(**chunk);
        while (auto event = parser.next_event()) {
            if (*event == "[DONE]") break;
            nlohmann::json j = nlohmann::json::parse(*event, nullptr, false);
            if (j.is_discarded()) continue;
            for (auto& e : parse_chunk(state, j)) {
                bool is_error = e.type() == StreamEvent::Type::Error;
                if (!(co_await sink.send(std::move(e))))
                    co_return;
                if (is_error) {
                    sink.close();
                    co_return;
                }
            }
        }
    }

    if (state.stop_reason == StopReason::Error) {
        co_await sink.send(StreamEvent{ Error{ Errc::ProviderError, state.error_message } });
    } else {
        co_await sink.send(StreamEvent{ DoneEvent{ build_response(state) } });
    }
    sink.close();
    co_return;
}

}  // namespace agent::detail
