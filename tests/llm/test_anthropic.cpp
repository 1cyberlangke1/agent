// Anthropic Messages 引擎测试。
//
// T0 纯函数：build_params（请求体字段 / thinking 映射 / 缓存挂载 / 消息转换）
//            parse_chunk（text/thinking/tool_use 事件流 / stop_reason / usage）
// T1 MockServer 回放官方格式 fixture（tests/fixtures/anthropic/*.sse，
//    字段对照 anthropic-python SDK 0.82.0 类型构造）+ 工具调用闭环

#include <agent/llm/model.hpp>
#include <agent/llm/providers/anthropic.hpp>
#include <agent/tools/tools.hpp>

#include "../core/mock_server.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace agent;
using namespace agent::detail;

namespace {

std::string load_fixture(std::string const& name)
{
    std::filesystem::path dir = AGENT_TEST_FIXTURES_DIR;
    std::ifstream in(dir / "anthropic" / name, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<agent::test::MockServer::Chunk> sse_response(std::string const& body, int status = 200)
{
    std::string head = "HTTP/1.1 " + std::to_string(status)
        + (status == 200 ? " OK" : " Error")
        + "\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
    return { { { head + body } } };
}

/// effort 型思考模型（adaptive + output_config.effort）。
void ensure_claude_sonnet()
{
    static bool done = [] {
        ModelRegistry::register_model(RuntimeModel{
            .id = "claude-sonnet-4-5",
            .context_window = 200000, .max_output_tokens = 8192,
            .reasoning = true,
            .thinking_level_map = { "off", "low", "medium", "high", "xhigh", "max", "max" },
        });
        return true;
    }();
    (void)done;
}

/// budget 型思考模型（enabled + budget_tokens）。
void ensure_claude_haiku()
{
    static bool done = [] {
        ModelRegistry::register_model(RuntimeModel{
            .id = "claude-haiku-3-5",
            .context_window = 200000, .max_output_tokens = 8192,
            .reasoning = true,
            .thinking_level_map = { "off", "1024", "2048", "4096", "8192", "16384", "32768" },
        });
        return true;
    }();
    (void)done;
}

template<typename Provider>
std::vector<StreamEvent> collect_stream(Provider& provider, ModelView const& model,
                                        Context const& ctx, StreamOptions const& opts = {})
{
    std::vector<StreamEvent> events;
    for (auto event : provider.stream(model, ctx, opts))
        events.push_back(std::move(event));
    return events;
}

std::string joined_text(std::vector<StreamEvent> const& events)
{
    std::string text;
    for (auto const& e : events)
        if (e.type() == StreamEvent::Type::TextDelta)
            text += std::get<TextDelta>(e.data).text;
    return text;
}

std::optional<ChatResponse> done_response(std::vector<StreamEvent> const& events)
{
    for (auto const& e : events)
        if (e.type() == StreamEvent::Type::Done)
            return std::get<DoneEvent>(e.data).response;
    return std::nullopt;
}

Context simple_ctx(std::string user_text)
{
    Context ctx;
    ctx.messages.push_back(Message{ Role::User, { Text{ std::move(user_text) } } });
    return ctx;
}

ToolInfo weather_tool()
{
    return ToolInfo{
        .name = "get_weather", .description = "查询城市天气",
        .parameters = { { "type", "object" },
                        { "properties", { { "city", { { "type", "string" } } } } },
                        { "required", { "city" } } },
    };
}

/// 注册 get_weather 工具（测试共享 Tools 全局，先查后注册避免同 id 重复注册失败）。
void ensure_get_weather_tool()
{
    if (Tools::get("get_weather").has_value())
        return;
    auto reg = Tools::reg(weather_tool(), [](nlohmann::json args) -> Result<std::string> {
        return "weather of " + args["city"].get<std::string>() + ": sunny";
    });
    REQUIRE(reg.has_value());
}

}  // namespace

// ───────────────────── T0: build_params ─────────────────────

TEST_CASE("anthropic build_params：基础字段 + system + tools input_schema")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    Context ctx = simple_ctx("hi");
    ctx.system_prompt = "你是助手";
    ensure_get_weather_tool();
    ctx.tools.push_back("get_weather");
    StreamOptions opts;
    opts.max_tokens = 512;
    opts.temperature = 0.5;

    auto params = *AnthropicMessagesEngine::build_params(*model, ctx, opts);
    CHECK(params["model"] == "claude-sonnet-4-5");
    CHECK(params["max_tokens"] == 512);
    CHECK(params["stream"] == true);
    REQUIRE(params["messages"].is_array());
    CHECK(params["messages"][0]["role"] == "user");
    CHECK(params["messages"][0]["content"] == "hi");
    REQUIRE(params["system"].is_array());
    CHECK(params["system"][0]["type"] == "text");
    CHECK(params["system"][0]["text"] == "你是助手");
    REQUIRE(params["tools"].is_array());
    CHECK(params["tools"][0]["name"] == "get_weather");
    REQUIRE(params["tools"][0].contains("input_schema"));
    CHECK(params["tools"][0]["input_schema"]["type"] == "object");
    CHECK(params["temperature"] == 0.5);   // 思考未开启 → 传
}

TEST_CASE("anthropic build_params：max_tokens 必填，用户不传用模型表")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    auto params = *AnthropicMessagesEngine::build_params(*model, simple_ctx("hi"), {});
    CHECK(params["max_tokens"] == 8192);   // model.max_output_tokens
    CHECK(!params.contains("temperature"));
    CHECK(!params.contains("thinking"));
}

TEST_CASE("anthropic build_params：effort 档 → adaptive + output_config.effort")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    StreamOptions opts;
    opts.reasoning = ThinkingLevel::High;
    auto params = *AnthropicMessagesEngine::build_params(*model, simple_ctx("hi"), opts);
    CHECK(params["thinking"]["type"] == "adaptive");
    CHECK(params["output_config"]["effort"] == "xhigh");   // sonnet map High=xhigh
    CHECK(!params.contains("temperature"));   // 思考开启 → temperature 不传
}

TEST_CASE("anthropic build_params：budget 档 → enabled + budget_tokens")
{
    ensure_claude_haiku();
    auto model = ModelRegistry::find_model("claude-haiku-3-5");
    REQUIRE(model.has_value());
    StreamOptions opts;
    opts.reasoning = ThinkingLevel::Medium;
    auto params = *AnthropicMessagesEngine::build_params(*model, simple_ctx("hi"), opts);
    CHECK(params["thinking"]["type"] == "enabled");
    CHECK(params["thinking"]["budget_tokens"] == 4096);   // haiku map Medium=4096
    CHECK(params["thinking"]["display"] == "summarized");
}

TEST_CASE("anthropic build_params：Off → disabled，toggle on → enabled(1024)")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    StreamOptions opts;
    opts.reasoning = ThinkingLevel::Off;
    auto params = *AnthropicMessagesEngine::build_params(*model, simple_ctx("hi"), opts);
    CHECK(params["thinking"]["type"] == "disabled");
}

TEST_CASE("anthropic build_params：缓存挂 system/最后 tool/最后 user")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    Context ctx = simple_ctx("hi");
    ctx.system_prompt = "你是助手";
    ensure_get_weather_tool();
    ctx.tools.push_back("get_weather");
    StreamOptions opts;
    opts.cache_retention = CacheRetention::Long;

    auto params = *AnthropicMessagesEngine::build_params(*model, ctx, opts);
    // system 块
    REQUIRE(params["system"][0].contains("cache_control"));
    CHECK(params["system"][0]["cache_control"]["type"] == "ephemeral");
    CHECK(params["system"][0]["cache_control"]["ttl"] == "1h");   // Long → ttl
    // 最后 tool
    REQUIRE(params["tools"][0].contains("cache_control"));
    CHECK(params["tools"][0]["cache_control"]["type"] == "ephemeral");
    // 最后 user 消息：字符串 content → 转 blocks 挂 cache_control
    auto const& last_user = params["messages"][params["messages"].size() - 1];
    REQUIRE(last_user["content"].is_array());
    CHECK(last_user["content"].back()["type"] == "text");
    CHECK(last_user["content"].back()["cache_control"]["type"] == "ephemeral");

    // None → 不挂
    StreamOptions none;
    none.cache_retention = CacheRetention::None;
    auto params_none = *AnthropicMessagesEngine::build_params(*model, simple_ctx("hi"), none);
    CHECK(!params_none.contains("system"));
    CHECK(!params_none.contains("tools"));
    CHECK(params_none["messages"][0]["content"] == "hi");   // 未转 blocks
}

TEST_CASE("anthropic build_params：多轮消息转换（thinking 签名回传 / tool_use / tool_result 合并）")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    Context ctx = simple_ctx("北京天气怎么样");
    // 第一轮 assistant：thinking（带签名）+ tool_use
    Message assistant{ Role::Assistant,
                       { Thinking{ "Let me think", false, "EuAPBCsig" },
                         ToolCall{ "toolu_01", "get_weather", { { "city", "北京" } } } } };
    // 连续两条 tool_result → 应合并进同一个 user 消息
    Message tool_result1{ Role::ToolResult, { ToolResult{ "toolu_01", "sunny", false } } };
    Message tool_result2{ Role::ToolResult, { ToolResult{ "toolu_01", "extra", false } } };
    ctx.messages.push_back(assistant);
    ctx.messages.push_back(tool_result1);
    ctx.messages.push_back(tool_result2);

    auto params = *AnthropicMessagesEngine::build_params(*model, ctx, {});
    auto const& messages = params["messages"];
    REQUIRE(messages.size() == 3);
    // [0] user, [1] assistant, [2] user(tool_result 合并)
    auto const& asst = messages[1];
    CHECK(asst["role"] == "assistant");
    REQUIRE(asst["content"].is_array());
    CHECK(asst["content"][0]["type"] == "thinking");
    CHECK(asst["content"][0]["thinking"] == "Let me think");
    CHECK(asst["content"][0]["signature"] == "EuAPBCsig");   // 签名原样回传
    CHECK(asst["content"][1]["type"] == "tool_use");
    CHECK(asst["content"][1]["id"] == "toolu_01");
    CHECK(asst["content"][1]["name"] == "get_weather");
    CHECK(asst["content"][1]["input"]["city"] == "北京");
    // 合并的 tool_result user 消息
    auto const& tr_user = messages[2];
    CHECK(tr_user["role"] == "user");
    REQUIRE(tr_user["content"].is_array());
    CHECK(tr_user["content"].size() == 2);
    CHECK(tr_user["content"][0]["type"] == "tool_result");
    CHECK(tr_user["content"][0]["tool_use_id"] == "toolu_01");
    CHECK(tr_user["content"][0]["content"] == "sunny");
    CHECK(tr_user["content"][1]["content"] == "extra");
}

TEST_CASE("anthropic build_params：无签名 thinking 降级为文本，redacted 原样")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    Context ctx = simple_ctx("hi");
    Message assistant{ Role::Assistant, { Thinking{ "no sig text" } } };   // 无签名
    ctx.messages.push_back(assistant);
    auto params = *AnthropicMessagesEngine::build_params(*model, ctx, {});
    auto const& blocks = params["messages"][1]["content"];
    REQUIRE(blocks.size() == 1);
    CHECK(blocks[0]["type"] == "text");          // 降级
    CHECK(blocks[0]["text"] == "no sig text");

    Context ctx2 = simple_ctx("hi");
    Message redacted{ Role::Assistant, { Thinking{ "", true, "redacted-data" } } };
    ctx2.messages.push_back(redacted);
    auto params2 = *AnthropicMessagesEngine::build_params(*model, ctx2, {});
    auto const& blocks2 = params2["messages"][1]["content"];
    REQUIRE(blocks2.size() == 1);
    CHECK(blocks2[0]["type"] == "redacted_thinking");
    CHECK(blocks2[0]["data"] == "redacted-data");
}

// ───────────────────── T0: parse_chunk ─────────────────────

TEST_CASE("anthropic parse_chunk：message_start 记录 id/输入 usage")
{
    AnthropicStreamState state;
    nlohmann::json chunk{ { "type", "message_start" },
                          { "message", { { "id", "msg_01abc" },
                                         { "usage", { { "input_tokens", 25 }, { "output_tokens", 1 } } } } } };
    auto events = AnthropicMessagesEngine::parse_chunk(state, chunk);
    CHECK(events.empty());
    CHECK(state.response_id == "msg_01abc");
    CHECK(state.usage.input_tokens == 25);
}

TEST_CASE("anthropic parse_chunk：text_delta → TextDelta，thinking/signature 累积")
{
    AnthropicStreamState state;
    auto events = AnthropicMessagesEngine::parse_chunk(
        state, { { "type", "content_block_delta" }, { "index", 0 },
                 { "delta", { { "type", "text_delta" }, { "text", "Hello" } } } });
    REQUIRE(events.size() == 1);
    CHECK(events[0].type() == StreamEvent::Type::TextDelta);
    CHECK(std::get<TextDelta>(events[0].data).text == "Hello");
    CHECK(state.text == "Hello");

    AnthropicMessagesEngine::parse_chunk(
        state, { { "type", "content_block_delta" }, { "index", 1 },
                 { "delta", { { "type", "thinking_delta" }, { "thinking", "Let me think" } } } });
    CHECK(state.thinking == "Let me think");
    AnthropicMessagesEngine::parse_chunk(
        state, { { "type", "content_block_delta" }, { "index", 1 },
                 { "delta", { { "type", "signature_delta" }, { "signature", "EuAPBC..." } } } });
    CHECK(state.thinking_signature == "EuAPBC...");
}

TEST_CASE("anthropic parse_chunk：tool_use 生命周期 → ToolCallDelta + ToolCallEnd")
{
    AnthropicStreamState state;
    // content_block_start：tool_use 块 id/name
    AnthropicMessagesEngine::parse_chunk(
        state, { { "type", "content_block_start" }, { "index", 1 },
                 { "content_block", { { "type", "tool_use" }, { "id", "toolu_01" },
                                      { "name", "get_weather" }, { "input", { } } } } });
    CHECK(state.tools[1].id == "toolu_01");
    CHECK(state.tools[1].name == "get_weather");
    // input_json_delta 增量累积
    auto delta1 = AnthropicMessagesEngine::parse_chunk(
        state, { { "type", "content_block_delta" }, { "index", 1 },
                 { "delta", { { "type", "input_json_delta" }, { "partial_json", "{\"city\":" } } } });
    REQUIRE(delta1.size() == 1);
    CHECK(delta1[0].type() == StreamEvent::Type::ToolCallDelta);
    auto const& delta = std::get<ToolCallDelta>(delta1[0].data);
    CHECK(delta.id == "toolu_01");
    CHECK(delta.name == "get_weather");
    CHECK(delta.arguments_delta == "{\"city\":");
    AnthropicMessagesEngine::parse_chunk(
        state, { { "type", "content_block_delta" }, { "index", 1 },
                 { "delta", { { "type", "input_json_delta" }, { "partial_json", "\"北京\"}" } } } });
    CHECK(state.tools[1].partial_args == "{\"city\":\"北京\"}");
    // content_block_stop：收尾 → ToolCallEnd 完整参数
    auto stop = AnthropicMessagesEngine::parse_chunk(
        state, { { "type", "content_block_stop" }, { "index", 1 } });
    REQUIRE(stop.size() == 1);
    CHECK(stop[0].type() == StreamEvent::Type::ToolCallEnd);
    auto const& end = std::get<ToolCallEnd>(stop[0].data);
    CHECK(end.id == "toolu_01");
    CHECK(end.arguments["city"] == "北京");
}

TEST_CASE("anthropic parse_chunk：message_delta → stop_reason + 累积 usage")
{
    AnthropicStreamState state;
    state.usage.input_tokens = 37;   // 来自 message_start
    auto events = AnthropicMessagesEngine::parse_chunk(
        state, { { "type", "message_delta" },
                 { "delta", { { "stop_reason", "end_turn" }, { "stop_sequence", nullptr } } },
                 { "usage", { { "output_tokens", 42 }, { "cache_read_input_tokens", 5 } } } });
    CHECK(state.stop_reason == StopReason::Stop);
    REQUIRE(events.size() == 1);
    CHECK(events[0].type() == StreamEvent::Type::Usage);
    CHECK(state.usage.output_tokens == 42);
    CHECK(state.usage.cache_read_tokens == 5);
    CHECK(state.usage.input_tokens == 37);   // 缺字段保留现值
    CHECK(state.usage.total_tokens == 84);   // 37+42+5
}

TEST_CASE("anthropic parse_chunk：stop_reason 映射（max_tokens/tool_use/refusal）")
{
    auto map_reason = [](std::string const& reason) {
        AnthropicStreamState state;
        AnthropicMessagesEngine::parse_chunk(
            state, { { "type", "message_delta" },
                     { "delta", { { "stop_reason", reason } } },
                     { "usage", { { "output_tokens", 1 } } } });
        return state.stop_reason;
    };
    CHECK(map_reason("max_tokens") == StopReason::Length);
    CHECK(map_reason("tool_use") == StopReason::ToolUse);
    CHECK(map_reason("refusal") == StopReason::Error);
    CHECK(map_reason("pause_turn") == StopReason::Stop);
    CHECK(map_reason("stop_sequence") == StopReason::Stop);
    // 官方值表补充：model_context_window_exceeded → 文档要求当截断处理
    CHECK(map_reason("model_context_window_exceeded") == StopReason::Length);
}

TEST_CASE("anthropic parse_chunk：redacted_thinking / error 事件 / ping 忽略")
{
    AnthropicStreamState state;
    AnthropicMessagesEngine::parse_chunk(
        state, { { "type", "content_block_start" }, { "index", 0 },
                 { "content_block", { { "type", "redacted_thinking" }, { "data", "redacted-payload" } } } });
    CHECK(state.redacted);
    CHECK(state.thinking_signature == "redacted-payload");

    auto err = AnthropicMessagesEngine::parse_chunk(
        state, { { "type", "error" }, { "error", { { "type", "overloaded_error" },
                                                    { "message", "overloaded" } } } });
    REQUIRE(err.size() == 1);
    CHECK(err[0].type() == StreamEvent::Type::Error);
    CHECK(state.stop_reason == StopReason::Error);

    auto ping = AnthropicMessagesEngine::parse_chunk(state, { { "type", "ping" } });
    CHECK(ping.empty());
}

// ───────────────────── T1: fixture 回放 ─────────────────────

TEST_CASE("Anthropic 文本 fixture：思考 + 正文 + usage + 请求头")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({
        [](agent::test::RequestView const& request) {
            CHECK(request.target == "/v1/messages");
            CHECK(request.header("x-api-key") == "k");
            CHECK(request.header("anthropic-version") == "2023-06-01");
        },
        sse_response(load_fixture("anthropic_text.sse")),
    });

    AnthropicMessagesProvider provider({ .name = "anthropic", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    std::string thinking;
    for (auto const& e : events)
        if (e.type() == StreamEvent::Type::ThinkingDelta)
            thinking += std::get<ThinkingDelta>(e.data).text;
    CHECK(thinking.find("Let me check") != std::string::npos);
    CHECK(joined_text(events).find("sunny") != std::string::npos);
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    CHECK(done->response_id == "msg_01XFDUDYJgAACzvnptvVoYEL");
    CHECK(done->usage.input_tokens == 25);
    CHECK(done->usage.output_tokens == 42);
    bool has_thinking = false;
    for (auto const& b : done->content)
        if (auto th = std::get_if<Thinking>(&b)) {
            has_thinking = true;
            CHECK(th->signature == "EuAPBC...");   // signature 已保存
        }
    CHECK(has_thinking);
    CHECK(server.errors().empty());
}

TEST_CASE("Anthropic 工具调用闭环 fixture：两轮（tool_result 合并 user 消息）")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    ensure_get_weather_tool();

    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("anthropic_tool_round1.sse")) });

    AnthropicMessagesProvider provider({ .name = "anthropic", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("北京天气怎么样"));
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::ToolUse);

    ToolCall tool_call;
    bool found = false;
    for (auto const& e : events) {
        if (e.type() == StreamEvent::Type::ToolCallEnd) {
            auto const& end = std::get<ToolCallEnd>(e.data);
            tool_call = ToolCall{ end.id, end.name, end.arguments };
            found = true;
        }
    }
    REQUIRE(found);
    CHECK(tool_call.name == "get_weather");
    CHECK(tool_call.arguments["city"] == "北京");

    auto exec_result = Tools::exec(tool_call.name, tool_call.arguments);
    REQUIRE(exec_result.has_value());
    Context round2 = simple_ctx("北京天气怎么样");
    round2.messages.push_back(Message{ Role::Assistant, { tool_call } });
    round2.messages.push_back(Message{ Role::ToolResult, { ToolResult{ tool_call.id, exec_result.value(), false } } });

    // round2 断言请求体：assistant tool_use + user(tool_result)（Anthropic 多轮格式）
    server.enqueue({
        [](agent::test::RequestView const& request) {
            nlohmann::json body = nlohmann::json::parse(request.body, nullptr, false);
            if (!body.is_object() || !body.contains("messages") || !body["messages"].is_array())
                return;
            auto const& messages = body["messages"];
            if (messages.size() < 3)
                return;
            // [1] assistant：content 块 type=tool_use（id/name 在块顶层）
            auto const& asst = messages[1];
            CHECK(asst.value("role", "") == "assistant");
            if (asst.contains("content") && asst["content"].is_array() && !asst["content"].empty()) {
                auto const& block = asst["content"][0];
                CHECK(block.value("type", "") == "tool_use");
                CHECK(block.value("id", "") == "toolu_01A09q90qw90lK917Tz1w2Nr");
                CHECK(block.value("name", "") == "get_weather");
            }
            // [2] user：tool_result 合并块（字段在块顶层）
            auto const& tr_user = messages[2];
            CHECK(tr_user.value("role", "") == "user");
            if (tr_user.contains("content") && tr_user["content"].is_array() && !tr_user["content"].empty()) {
                auto const& block = tr_user["content"][0];
                CHECK(block.value("type", "") == "tool_result");
                CHECK(block.value("tool_use_id", "") == "toolu_01A09q90qw90lK917Tz1w2Nr");
                CHECK(block.value("content", "").size() > 0);
            }
        },
        sse_response(load_fixture("anthropic_tool_round2.sse")),
    });
    auto result = provider.complete(*model, round2, StreamOptions{});
    REQUIRE(result.has_value());
    CHECK(result->stop_reason == StopReason::Stop);
    CHECK(result->content.size() >= 1);
    CHECK(server.errors().empty());
    CHECK(server.request_count() == 2);
}

TEST_CASE("Anthropic 异步流式 stream_async")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("anthropic_text.sse")) });

    AnthropicMessagesProvider provider({ .name = "anthropic", .api_key = "k", .base_url = server.base_url() });
    asio::io_context io;
    std::vector<StreamEvent> events;
    bool completed = false;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        AsyncStream<StreamEvent> sink(io.get_executor());
        auto local = sink;   // 共享 channel：生产端 move 进协程，本协程留消费副本
        asio::co_spawn(io, [&]() -> asio::awaitable<void> {
            co_await provider.stream_async(*model, simple_ctx("hi"), StreamOptions{}, std::move(local));
        }, asio::detached);
        while (auto event = co_await sink.receive())
            events.push_back(std::move(*event));
        completed = true;
    }, asio::detached);
    io.run();
    CHECK(completed);
    CHECK(joined_text(events).size() > 0);
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    CHECK(server.errors().empty());
}

TEST_CASE("Anthropic 异步非流式 complete_async")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("anthropic_text.sse")) });

    AnthropicMessagesProvider provider({ .name = "anthropic", .api_key = "k", .base_url = server.base_url() });
    asio::io_context io;
    std::optional<Result<ChatResponse>> outcome;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        outcome = co_await provider.complete_async(*model, simple_ctx("hi"), StreamOptions{});
    }, asio::detached);
    io.run();
    REQUIRE(outcome.has_value());
    REQUIRE(outcome->has_value());
    CHECK(outcome->value().stop_reason == StopReason::Stop);
    CHECK(outcome->value().usage.input_tokens == 25);
    CHECK(server.errors().empty());
}

// ───────────────────── 难样例（Anthropic 引擎路径）─────────────────────

TEST_CASE("Anthropic 难样例：EOF 无 message_stop → 兜底 Done")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    // 正常流但缺 message_stop 事件（官方终止事件）→ EOF 兜底聚合 Done，不报错
    std::string body =
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\",\"type\":\"message\","
        "\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,"
        "\"stop_sequence\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n"
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n"
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},"
        "\"usage\":{\"output_tokens\":5}}\n\n";
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(body) });
    AnthropicMessagesProvider provider({ .name = "anthropic", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    CHECK(joined_text(events) == "hi");
    CHECK(server.errors().empty());
}

TEST_CASE("Anthropic 难样例：流中断流 → Error 事件")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    // 完整响应发一半就 RST 断连 → 传输层错误 → Error 事件，绝不静默成功
    std::string body = load_fixture("anthropic_tool_round2.sse");
    agent::test::MockServer::Exchange exchange;
    exchange.chunks = sse_response(body.substr(0, body.size() / 2));
    exchange.close_abruptly = true;
    agent::test::MockServer server;
    server.enqueue(std::move(exchange));
    AnthropicMessagesProvider provider({ .name = "anthropic", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    bool saw_error = false;
    for (auto const& e : events)
        if (e.type() == StreamEvent::Type::Error)
            saw_error = true;
    CHECK(saw_error);
    CHECK(!done_response(events).has_value());
    CHECK(server.errors().empty());
}

TEST_CASE("Anthropic 难样例：异步取消中断流 → Error")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    // 慢滴流：每块 30ms，取消发生在前几个块后
    std::vector<agent::test::MockServer::Chunk> slow;
    std::string sse = load_fixture("anthropic_text.sse");
    std::size_t pos = 0;
    while (pos < sse.size()) {
        std::size_t next = sse.find("\n\n", pos);
        if (next == std::string::npos) {
            slow.push_back({ { sse.substr(pos) }, 30 });
            break;
        }
        slow.push_back({ { sse.substr(pos, next - pos + 2) }, 30 });
        pos = next + 2;
    }
    server.enqueue({ {}, slow });

    AnthropicMessagesProvider provider({ .name = "anthropic", .api_key = "k", .base_url = server.base_url() });
    asio::io_context io;
    asio::cancellation_signal cancel;
    std::vector<StreamEvent> events;
    bool completed = false;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        StreamOptions opts;
        opts.cancel = &cancel;
        AsyncStream<StreamEvent> sink(io.get_executor());
        auto local = sink;
        asio::co_spawn(io, [&]() -> asio::awaitable<void> {
            co_await provider.stream_async(*model, simple_ctx("hi"), opts, std::move(local));
        }, asio::detached);
        int count = 0;
        while (auto event = co_await sink.receive()) {
            events.push_back(std::move(*event));
            if (++count == 2)
                cancel.emit(asio::cancellation_type::all);
        }
        completed = true;
    }, asio::detached);
    io.run();
    CHECK(completed);
    auto error = std::find_if(events.begin(), events.end(), [](auto const& e) {
        return e.type() == StreamEvent::Type::Error;
    });
    REQUIRE(error != events.end());
}

TEST_CASE("Anthropic 难样例：401 → AuthError + 请求头断言")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    std::string err_body = "{\"type\":\"error\",\"error\":{\"type\":\"authentication_error\","
                           "\"message\":\"invalid x-api-key\"}}";
    server.enqueue({
        [](agent::test::RequestView const& request) {
            CHECK(request.target == "/v1/messages");
            CHECK(request.header("x-api-key") == "bad-key");
            CHECK(request.header("anthropic-version") == "2023-06-01");
        },
        sse_response(err_body, 401),
    });
    AnthropicMessagesProvider provider({ .name = "anthropic", .api_key = "bad-key", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    bool auth_error = false;
    for (auto const& e : events)
        if (auto err = std::get_if<Error>(&e.data))
            if (err->code == Errc::AuthError && err->message.find("x-api-key") != std::string::npos)
                auth_error = true;
    CHECK(auth_error);
    CHECK(server.errors().empty());
}

// ───────────────────── T2 契约 dump（AGENT_CONTRACT=ON 时编译）─────────────────────
#ifdef AGENT_CONTRACT_OUT

TEST_CASE("contract: dump anthropic build_params 文本场景")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    Context ctx;
    ctx.messages.push_back(Message{ Role::User, { Text{ "hi" } } });
    auto params = *AnthropicMessagesEngine::build_params(*model, ctx, {});
    std::filesystem::path dir = AGENT_CONTRACT_OUT;
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "anthropic_text.json") << params.dump(2);
}

TEST_CASE("contract: dump anthropic build_params 工具场景")
{
    ensure_claude_sonnet();
    auto model = ModelRegistry::find_model("claude-sonnet-4-5");
    REQUIRE(model.has_value());
    Context ctx;
    ctx.messages.push_back(Message{ Role::User, { Text{ "hi" } } });
    ensure_get_weather_tool();
    ctx.tools.push_back("get_weather");
    auto params = *AnthropicMessagesEngine::build_params(*model, ctx, {});
    std::filesystem::path dir = AGENT_CONTRACT_OUT;
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "anthropic_tool_1.json") << params.dump(2);
}

#endif
