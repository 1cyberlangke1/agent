// OpenAI / DeepSeek 引擎测试。
//
// T0 纯函数（无网络）：build_params / parse_chunk——归一化正确性
// T1 MockServer 回放官方 SDK 真实快照（tests/fixtures/openai/*.sse）：
//    四接口全覆盖（同步/异步 × 流式/非流式）+ 工具调用闭环（真实 Tools::exec）

#include <agent/llm/model.hpp>
#include <agent/llm/providers/compatible.hpp>
#include <agent/llm/providers/deepseek.hpp>
#include <agent/llm/providers/openai.hpp>
#include <agent/tools/tools.hpp>

#include "../core/mock_server.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace agent;
using namespace agent::detail;

namespace {

std::string load_fixture(std::string const& name, std::string const& provider = "openai")
{
    std::filesystem::path dir = AGENT_TEST_FIXTURES_DIR;
    std::ifstream in(dir / provider / name, std::ios::binary);
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

// 注册独立测试模型（不依赖生成表，thinking map 可控）
void ensure_test_models()
{
    static bool done = [] {
        ModelRegistry::register_model(RuntimeModel{
            .id = "t-effort", .context_window = 32000, .max_output_tokens = 8192,
            .reasoning = true,
            .thinking_level_map = { "off", "low", "medium", "high", "high", "high", "max" },
        });
        ModelRegistry::register_model(RuntimeModel{
            .id = "t-plain", .context_window = 32000, .max_output_tokens = 8192,
            .reasoning = false,
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
    for (auto const& event : events)
        if (event.type() == StreamEvent::Type::TextDelta)
            text += std::get<TextDelta>(event.data).text;
    return text;
}

std::string text_of_blocks(std::vector<ContentBlock> const& blocks)
{
    std::string text;
    for (auto const& b : blocks)
        if (auto t = std::get_if<Text>(&b)) text += t->text;
    return text;
}

std::optional<ChatResponse> done_response(std::vector<StreamEvent> const& events)
{
    for (auto const& event : events)
        if (event.type() == StreamEvent::Type::Done)
            return std::get<DoneEvent>(event.data).response;
    return std::nullopt;
}

Context simple_ctx(std::string user_text)
{
    Context ctx;
    ctx.system_prompt = "你是助手";
    ctx.messages.push_back(Message{ Role::User, { Text{ std::move(user_text) } } });
    return ctx;
}

/// 构造单个 delta tool_calls 的 chunk（显式逐层构建，避免嵌套 initializer 歧义）。
nlohmann::json tool_delta_chunk(int tc_index, std::string args, std::string id = "", std::string name = "")
{
    nlohmann::json tc;
    tc["index"] = tc_index;
    if (!id.empty()) tc["id"] = id;
    tc["type"] = "function";
    if (!name.empty()) tc["function"]["name"] = name;
    tc["function"]["arguments"] = std::move(args);

    nlohmann::json delta;
    delta["tool_calls"] = nlohmann::json::array();
    delta["tool_calls"].push_back(std::move(tc));

    nlohmann::json chunk;
    chunk["choices"] = nlohmann::json::array();
    chunk["choices"].push_back({ { "index", 0 }, { "delta", delta } });
    return chunk;
}

/// 构造只带 finish_reason 的 chunk。
nlohmann::json finish_chunk(std::string reason)
{
    nlohmann::json chunk;
    chunk["choices"] = nlohmann::json::array();
    chunk["choices"].push_back({ { "index", 0 }, { "delta", nlohmann::json::object() },
                                 { "finish_reason", std::move(reason) } });
    return chunk;
}

/// 注册 get_weather 工具（测试共享 Tools 全局，先查后注册避免同 id 重复注册失败）。
void ensure_get_weather_tool()
{
    if (Tools::get("get_weather").has_value())
        return;
    auto reg = Tools::reg(ToolInfo{
        .name = "get_weather", .description = "查询城市天气",
        .parameters = { { "type", "object" },
                        { "properties", { { "city", { { "type", "string" } } } } },
                        { "required", { "city" } } },
    }, [](nlohmann::json args) -> Result<std::string> {
        return "weather of " + args["city"].get<std::string>() + ": sunny";
    });
    REQUIRE(reg.has_value());
}

}  // namespace

// ───────────────────── T0: build_params ─────────────────────

TEST_CASE("build_params：未注册工具名 → Result 错误")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    Context ctx = simple_ctx("hi");
    ctx.tools.push_back("no_such_tool");
    auto result = OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(*model, ctx, {});
    CHECK(!result.has_value());
    CHECK(result.error().code == Errc::NotFound);
}

TEST_CASE("build_params 基本结构 + system role")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-effort");
    REQUIRE(model.has_value());
    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = "http://x" });

    auto params = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(*model, simple_ctx("hi"), {});
    CHECK(params["model"] == "t-effort");
    CHECK(params["stream"] == true);
    CHECK(params["stream_options"]["include_usage"] == true);
    REQUIRE(params["messages"].is_array());
    CHECK(params["messages"][0]["role"] == "system");   // Compat::system_role
    CHECK(params["messages"][0]["content"] == "你是助手");
    CHECK(params["messages"][1]["role"] == "user");
    // 未传 max_tokens → 不上传任何 max 字段
    CHECK(!params.contains("max_completion_tokens"));
    CHECK(!params.contains("max_tokens"));
}

TEST_CASE("build_params max_tokens 字段名随 Compat")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-effort");
    REQUIRE(model.has_value());
    StreamOptions opts;
    opts.max_tokens = 2048;

    auto openai_params = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(
        *model, simple_ctx("hi"), opts);
    CHECK(openai_params["max_completion_tokens"] == 2048);   // OpenAI 官方 o 系列字段
    CHECK(!openai_params.contains("max_tokens"));

    auto deepseek_params = *OpenAICompletionsEngine<DeepSeekThinking, OpenAICompatibleCompat>::build_params(
        *model, simple_ctx("hi"), opts);
    CHECK(deepseek_params["max_tokens"] == 2048);            // 第三方兼容端点通用字段
    CHECK(!deepseek_params.contains("max_completion_tokens"));
}

TEST_CASE("build_params temperature + tools 转换")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-effort");
    REQUIRE(model.has_value());
    StreamOptions opts;
    opts.temperature = 0.7;

    Context ctx = simple_ctx("hi");
    ensure_get_weather_tool();
    ctx.tools.push_back("get_weather");
    auto params = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(*model, ctx, opts);
    CHECK(params["temperature"] == 0.7);
    REQUIRE(params["tools"].is_array());
    CHECK(params["tools"][0]["type"] == "function");
    CHECK(params["tools"][0]["function"]["name"] == "get_weather");
    CHECK(params["tools"][0]["function"]["description"] == "查询城市天气");
    CHECK(params["tools"][0]["function"]["parameters"]["type"] == "object");
}

TEST_CASE("build_params thinking：OpenAI effort 映射 + 未指定不发")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-effort");
    REQUIRE(model.has_value());

    // 未指定 reasoning → 不发 reasoning_effort（模型默认）
    auto plain = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(*model, simple_ctx("hi"), {});
    CHECK(!plain.contains("reasoning_effort"));

    // High → clamp 到 t-effort map[High]=="high"
    StreamOptions opts;
    opts.reasoning = ThinkingLevel::High;
    auto params = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(*model, simple_ctx("hi"), opts);
    CHECK(params["reasoning_effort"] == "high");

    // 非推理模型 → 不发
    auto plain_model = ModelRegistry::find_model("t-plain");
    REQUIRE(plain_model.has_value());
    auto noop = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(*plain_model, simple_ctx("hi"), opts);
    CHECK(!noop.contains("reasoning_effort"));
}

TEST_CASE("build_params thinking：DeepSeek thinking + 移除采样参数")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-effort");
    REQUIRE(model.has_value());
    StreamOptions opts;
    opts.reasoning = ThinkingLevel::Max;
    opts.temperature = 0.9;

    auto params = *OpenAICompletionsEngine<DeepSeekThinking, OpenAICompatibleCompat>::build_params(
        *model, simple_ctx("hi"), opts);
    CHECK(params["thinking"]["type"] == "enabled");
    CHECK(params["reasoning_effort"] == "max");
    // 官方文档：思考模式禁 temperature/top_p/presence_penalty/frequency_penalty
    CHECK(!params.contains("temperature"));
    CHECK(!params.contains("top_p"));
    CHECK(!params.contains("presence_penalty"));
    CHECK(!params.contains("frequency_penalty"));

    // Off → 明确关闭
    StreamOptions off_opts;
    off_opts.reasoning = ThinkingLevel::Off;
    auto off = *OpenAICompletionsEngine<DeepSeekThinking, OpenAICompatibleCompat>::build_params(
        *model, simple_ctx("hi"), off_opts);
    CHECK(off["thinking"]["type"] == "disabled");
}

TEST_CASE("build_params 缓存：prompt_cache_key + retention")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-effort");
    REQUIRE(model.has_value());
    StreamOptions opts;
    opts.cache_retention = CacheRetention::Long;
    opts.session_id = "session-abc";

    auto params = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(*model, simple_ctx("hi"), opts);
    CHECK(params["prompt_cache_key"] == "session-abc");
    CHECK(params["prompt_cache_retention"] == "24h");

    // None → 无缓存字段
    StreamOptions none_opts;
    none_opts.cache_retention = CacheRetention::None;
    none_opts.session_id = "session-abc";
    auto no_cache = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(*model, simple_ctx("hi"), none_opts);
    CHECK(!no_cache.contains("prompt_cache_key"));
    CHECK(!no_cache.contains("prompt_cache_retention"));
}

TEST_CASE("build_params extra 透传 + 覆盖")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-effort");
    REQUIRE(model.has_value());
    StreamOptions opts;
    opts.extra = { { "store", false }, { "metadata", { { "k", "v" } } } };

    auto params = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(*model, simple_ctx("hi"), opts);
    CHECK(params["store"] == false);
    CHECK(params["metadata"]["k"] == "v");
}

// ───────────────────── T0: parse_chunk ─────────────────────

TEST_CASE("parse_chunk 文本增量 + usage + finish_reason")
{
    OpenAIStreamState state;
    auto events = OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(state, {
        { "id", "chatcmpl-1" },
        { "choices", { { { "index", 0 }, { "delta", { { "role", "assistant" }, { "content", "你" } } } } } },
    });
    REQUIRE(events.size() == 1);
    CHECK(events[0].type() == StreamEvent::Type::TextDelta);
    CHECK(std::get<TextDelta>(events[0].data).text == "你");
    CHECK(state.response_id == "chatcmpl-1");

    // 第二个 chunk 累积 + finish_reason=stop + usage
    auto events2 = OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(state, {
        { "id", "chatcmpl-1" },
        { "choices", { { { "index", 0 }, { "delta", { { "content", "好" } } },
                         { "finish_reason", "stop" } } } },
        { "usage", { { "prompt_tokens", 5 }, { "completion_tokens", 3 }, { "total_tokens", 8 } } },
    });
    CHECK(state.text == "你好");
    CHECK(state.stop_reason == StopReason::Stop);
    CHECK(state.usage.input_tokens == 5);
    bool has_usage = std::any_of(events2.begin(), events2.end(),
                                 [](auto const& e) { return e.type() == StreamEvent::Type::Usage; });
    CHECK(has_usage);
}

TEST_CASE("parse_chunk thinking：OpenAI reasoning 字段")
{
    OpenAIStreamState state;
    auto events = OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(state, {
        { "choices", { { { "index", 0 }, { "delta", { { "reasoning", "想一下" } } } } } },
    });
    REQUIRE(events.size() == 1);
    CHECK(events[0].type() == StreamEvent::Type::ThinkingDelta);
    CHECK(std::get<ThinkingDelta>(events[0].data).text == "想一下");
    CHECK(state.thinking == "想一下");
}

TEST_CASE("parse_chunk thinking：DeepSeek reasoning_content 字段")
{
    OpenAIStreamState state;
    auto events = OpenAICompletionsEngine<DeepSeekThinking, OpenAICompatibleCompat>::parse_chunk(state, {
        { "choices", { { { "index", 0 }, { "delta", { { "reasoning_content", "思考" } } } } } },
    });
    REQUIRE(events.size() == 1);
    CHECK(events[0].type() == StreamEvent::Type::ThinkingDelta);
    CHECK(std::get<ThinkingDelta>(events[0].data).text == "思考");
}

TEST_CASE("parse_chunk 工具调用：参数跨 chunk 累积 + ToolCallEnd")
{
    OpenAIStreamState state;
    // chunk 1：声明工具（id + name + 空参数）
    auto e1 = OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(
        state, tool_delta_chunk(0, "", "call_1", "get_weather"));
    CHECK(std::get<ToolCallDelta>(e1[0].data).id == "call_1");
    CHECK(std::get<ToolCallDelta>(e1[0].data).name == "get_weather");
    // chunk 2/3：参数逐段
    OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(state, tool_delta_chunk(0, "{\"city\":"));
    OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(state, tool_delta_chunk(0, "\"NYC\"}"));
    // chunk 4：finish_reason=tool_calls → ToolCallEnd
    auto e4 = OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(state, finish_chunk("tool_calls"));
    auto end = std::find_if(e4.begin(), e4.end(), [](auto const& e) {
        return e.type() == StreamEvent::Type::ToolCallEnd;
    });
    REQUIRE(end != e4.end());
    auto const& tool = std::get<ToolCallEnd>(end->data);
    CHECK(tool.id == "call_1");
    CHECK(tool.name == "get_weather");
    CHECK(tool.arguments["city"] == "NYC");
    CHECK(state.stop_reason == StopReason::ToolUse);
}

// usage 缓存字段三种格式（DeepSeek 可能发 OpenAI 格式也可能发自己格式，都需正确处理）
TEST_CASE("parse_usage：DeepSeek 实际格式 prompt_tokens_details.cached_tokens")
{
    OpenAIStreamState state;
    // 真实录制（deepseek fixture）格式：缓存命中走 prompt_tokens_details.cached_tokens
    nlohmann::json chunk;
    chunk["choices"] = nlohmann::json::array();   // usage chunk 的 choices 恒空
    chunk["usage"] = { { "prompt_tokens", 96 }, { "completion_tokens", 114 },
                       { "total_tokens", 210 },
                       { "prompt_tokens_details", { { "cached_tokens", 50 } } } };
    auto events = OpenAICompletionsEngine<DeepSeekThinking, OpenAICompatibleCompat>::parse_chunk(state, chunk);
    CHECK(state.usage.input_tokens == 96);
    CHECK(state.usage.output_tokens == 114);
    CHECK(state.usage.total_tokens == 210);
    CHECK(state.usage.cache_read_tokens == 50);
    bool has_usage = std::any_of(events.begin(), events.end(),
                                 [](auto const& e) { return e.type() == StreamEvent::Type::Usage; });
    CHECK(has_usage);
}

TEST_CASE("parse_usage：DeepSeek 文档格式 prompt_cache_hit/miss 顶层")
{
    OpenAIStreamState state;
    // 官方文档列出的顶层格式（兼容路径）
    nlohmann::json chunk;
    chunk["choices"] = nlohmann::json::array();
    chunk["usage"] = { { "prompt_tokens", 96 }, { "completion_tokens", 114 },
                       { "total_tokens", 210 },
                       { "prompt_cache_hit_tokens", 30 }, { "prompt_cache_miss_tokens", 66 } };
    OpenAICompletionsEngine<DeepSeekThinking, OpenAICompatibleCompat>::parse_chunk(state, chunk);
    CHECK(state.usage.cache_read_tokens == 30);
    CHECK(state.usage.cache_write_tokens == 66);
}

TEST_CASE("parse_usage：OpenAI 格式 prompt_tokens_details.cached_tokens")
{
    OpenAIStreamState state;
    nlohmann::json chunk;
    chunk["choices"] = nlohmann::json::array();
    chunk["usage"] = { { "prompt_tokens", 79 }, { "completion_tokens", 11 }, { "total_tokens", 90 },
                       { "prompt_tokens_details", { { "cached_tokens", 20 } } } };
    OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(state, chunk);
    CHECK(state.usage.cache_read_tokens == 20);
}

TEST_CASE("parse_chunk content_filter → Error 终止")
{
    OpenAIStreamState state;
    OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(state, finish_chunk("content_filter"));
    CHECK(state.stop_reason == StopReason::Error);
    CHECK(state.error_message.find("content_filter") != std::string::npos);
}

// ───────────────────── T1: MockServer 回放 ─────────────────────

TEST_CASE("OpenAI 同步流式：文本 fixture 全链路")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("openai_text_4.sse")) });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    CHECK(joined_text(events).size() > 0);           // content 增量已累积
    CHECK(!done->content.empty());
    CHECK(server.errors().empty());
    CHECK(server.request_count() == 1);
}

TEST_CASE("OpenAI 同步非流式 complete：文本 fixture")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("openai_text_4.sse")) });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    auto result = provider.complete(*model, simple_ctx("hi"), StreamOptions{});
    REQUIRE(result.has_value());
    CHECK(result->stop_reason == StopReason::Stop);
    CHECK(!result->content.empty());
    CHECK(server.errors().empty());
}

TEST_CASE("OpenAI 异步流式 stream_async")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("openai_text_4.sse")) });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    asio::io_context io;
    std::vector<StreamEvent> events;
    bool completed = false;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        AsyncStream<StreamEvent> sink(io.get_executor());
        auto local = sink;   // 共享 channel：生产端 move 进协程，本协程留消费副本
        asio::co_spawn(io, [&]() -> asio::awaitable<void> {
            co_await provider.stream_async(*model, simple_ctx("hi"), StreamOptions{}, std::move(local));
        }, asio::detached);
        while (auto event = co_await sink.receive()) {
            events.push_back(std::move(*event));
        }
        completed = true;
    }, asio::detached);
    io.run();
    CHECK(completed);
    CHECK(done_response(events).has_value());
    CHECK(server.errors().empty());
}

TEST_CASE("OpenAI 异步非流式 complete_async")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("openai_text_4.sse")) });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    asio::io_context io;
    std::optional<Result<ChatResponse>> outcome;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        outcome = co_await provider.complete_async(*model, simple_ctx("hi"), StreamOptions{});
    }, asio::detached);
    io.run();
    REQUIRE(outcome.has_value());
    REQUIRE(outcome->has_value());
    CHECK(outcome->value().stop_reason == StopReason::Stop);
    CHECK(server.errors().empty());
}

TEST_CASE("OpenAI 工具调用闭环：真实 Tools::exec + 多轮回传")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());

    // 注册真实工具（测试共享，先查后注册）
    ensure_get_weather_tool();

    // 第一轮：模型决定调工具（回放厂商真实 tool_call 快照，参数逐字符分包）
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("openai_tool_1.sse")) });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("北京天气"));
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::ToolUse);

    // 取 ToolCall
    ToolCall tool_call;
    bool found_tool = false;
    for (auto const& event : events) {
        if (event.type() == StreamEvent::Type::ToolCallEnd) {
            auto const& end = std::get<ToolCallEnd>(event.data);
            tool_call = ToolCall{ end.id, end.name, end.arguments };
            found_tool = true;
        }
    }
    REQUIRE(found_tool);
    CHECK(tool_call.name == "get_weather");
    CHECK(tool_call.arguments["city"] == "New York City");

    // 真实执行工具
    auto exec_result = Tools::exec(tool_call.name, tool_call.arguments);
    REQUIRE(exec_result.has_value());
    CHECK(exec_result.value().find("sunny") != std::string::npos);

    // 回传 assistant tool_calls + tool 结果 → 第二轮 mock 回放最终回答
    Context round2 = simple_ctx("北京天气");
    round2.messages.push_back(Message{ Role::Assistant, { tool_call } });
    round2.messages.push_back(Message{ Role::ToolResult, { ToolResult{ tool_call.id, exec_result.value(), false } } });

    // 第二轮断言请求体：assistant 带 tool_calls 回传 + tool 角色消息
    server.enqueue({
        [](agent::test::RequestView const& request) {
            nlohmann::json body = nlohmann::json::parse(request.body, nullptr, false);
            REQUIRE(body.is_object());
            REQUIRE(body.contains("messages"));
            REQUIRE(body["messages"].is_array());
            // system / user / assistant(tool_calls) / tool
            REQUIRE(body["messages"].size() >= 4);
            auto const& assistant_msg = body["messages"][2];
            CHECK(assistant_msg["role"] == "assistant");
            REQUIRE(assistant_msg.contains("tool_calls"));
            CHECK(assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather");
            CHECK(assistant_msg["tool_calls"][0]["function"]["arguments"].is_string());
            auto const& tool_msg = body["messages"][3];
            CHECK(tool_msg["role"] == "tool");
            CHECK(tool_msg["tool_call_id"].is_string());
            CHECK(tool_msg["content"].get<std::string>().find("sunny") != std::string::npos);
        },
        sse_response(load_fixture("openai_text_4.sse")),
    });

    auto result = provider.complete(*model, round2, StreamOptions{});
    REQUIRE(result.has_value());
    CHECK(result->stop_reason == StopReason::Stop);
    CHECK(server.errors().empty());
    CHECK(server.request_count() == 2);
}

TEST_CASE("OpenAI 非 2xx → Error 事件（429 → RateLimited）")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    // 429 + 错误 body（首字节前重试默认 2 次，脚本只给 1 个 → 其余 unexpected）
    server.enqueue({ {}, sse_response(R"({"error":{"message":"rate limited"}})", 429) });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url(),
                              .default_headers = {} });
    // 关闭重试，避免请求数不匹配
    StreamOptions opts;
    opts.max_retries = 0;
    auto events = collect_stream(provider, *model, simple_ctx("hi"), opts);
    auto error = std::find_if(events.begin(), events.end(), [](auto const& e) {
        return e.type() == StreamEvent::Type::Error;
    });
    REQUIRE(error != events.end());
    CHECK(std::get<Error>(error->data).code == Errc::RateLimited);
    CHECK(server.request_count() == 1);
}

TEST_CASE("DeepSeek 流式：文本 fixture（复用共享引擎 + 兼容字段名）")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("openai_text_4.sse")) });

    DeepSeekProvider provider({ .name = "deepseek", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    CHECK(joined_text(events).size() > 0);
    CHECK(server.errors().empty());
}

TEST_CASE("parse_chunk 并行工具调用：index 交错累积")
{
    OpenAIStreamState state;
    // 声明两个工具（index 0/1）
    OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(state, tool_delta_chunk(0, "", "call_a", "get_weather"));
    OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(state, tool_delta_chunk(1, "", "call_b", "get_time"));
    // 参数交错到达
    OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(state, tool_delta_chunk(0, "{\"city\":\"NYC\""));
    OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(state, tool_delta_chunk(1, "{\"city\":\"LA\""));
    OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(state, tool_delta_chunk(0, "}"));
    OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(state, tool_delta_chunk(1, "}"));
    // finish → 两个 ToolCallEnd
    auto ends = OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(state, finish_chunk("tool_calls"));
    int end_count = 0;
    std::map<std::string, std::string> args_by_name;
    for (auto const& e : ends) {
        if (e.type() != StreamEvent::Type::ToolCallEnd) continue;
        ++end_count;
        auto const& end = std::get<ToolCallEnd>(e.data);
        args_by_name[end.name] = end.arguments.dump();
    }
    CHECK(end_count == 2);
    CHECK(args_by_name["get_weather"].find("NYC") != std::string::npos);
    CHECK(args_by_name["get_time"].find("LA") != std::string::npos);
    CHECK(state.stop_reason == StopReason::ToolUse);
}

TEST_CASE("OpenAI thinking 流式：reasoning delta → ThinkingDelta + 正文")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-effort");
    REQUIRE(model.has_value());
    std::string sse =
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"reasoning\":\"\xe6\x80\x9d\xe8\x80\x83\xe4\xb8\xad\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\"\xe5\x9b\x9e\xe7\xad\x94\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
        "data: [DONE]\n\n";
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(sse) });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    std::string thinking;
    for (auto const& e : events)
        if (e.type() == StreamEvent::Type::ThinkingDelta)
            thinking += std::get<ThinkingDelta>(e.data).text;
    CHECK(thinking == "思考中");
    CHECK(joined_text(events) == "回答");
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    CHECK(server.errors().empty());
}

TEST_CASE("DeepSeek thinking 流式：reasoning_content delta")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-effort");
    REQUIRE(model.has_value());
    std::string sse =
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"\xe8\x80\x83\xe8\x99\x91\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\"\xe5\xa5\xbd\xe4\xba\x86\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
        "data: [DONE]\n\n";
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(sse) });

    DeepSeekProvider provider({ .name = "deepseek", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    std::string thinking;
    for (auto const& e : events)
        if (e.type() == StreamEvent::Type::ThinkingDelta)
            thinking += std::get<ThinkingDelta>(e.data).text;
    CHECK(thinking == "考虑");
    CHECK(joined_text(events) == "好了");
    CHECK(server.errors().empty());
}

TEST_CASE("OpenAI 401 → AuthError")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(R"({"error":{"message":"bad key"}})", 401) });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    StreamOptions opts;
    opts.max_retries = 0;
    auto events = collect_stream(provider, *model, simple_ctx("hi"), opts);
    auto error = std::find_if(events.begin(), events.end(), [](auto const& e) {
        return e.type() == StreamEvent::Type::Error;
    });
    REQUIRE(error != events.end());
    CHECK(std::get<Error>(error->data).code == Errc::AuthError);
    CHECK(server.request_count() == 1);
}

TEST_CASE("OpenAI 非流式 complete 工具调用：聚合 ToolCall")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("openai_tool_1.sse")) });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    auto result = provider.complete(*model, simple_ctx("北京天气"), StreamOptions{});
    REQUIRE(result.has_value());
    CHECK(result->stop_reason == StopReason::ToolUse);
    // 聚合的 content 含 ToolCall（非流式路径也正确收尾工具调用）
    bool found_tool = false;
    for (auto const& block : result->content) {
        if (auto tc = std::get_if<ToolCall>(&block)) {
            found_tool = true;
            CHECK(tc->name == "get_weather");
            CHECK(tc->arguments["city"] == "New York City");
        }
    }
    CHECK(found_tool);
    CHECK(server.errors().empty());
}

TEST_CASE("OpenAI 流中 error 事件 → Error 终结")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    // 先一个正常 chunk，随后 SSE 内嵌 error（OpenAI data:{"error":{...}}）
    std::string sse =
        "data: {\"choices\":[{\"delta\":{\"content\":\"\xe5\x89\x8d\xe5\x8d\x8a\"}}]}\n\n"
        "data: {\"error\":{\"message\":\"stream aborted\"}}\n\n";
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(sse) });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    auto error = std::find_if(events.begin(), events.end(), [](auto const& e) {
        return e.type() == StreamEvent::Type::Error;
    });
    REQUIRE(error != events.end());
    CHECK(std::get<Error>(error->data).code == Errc::ProviderError);
    CHECK(std::get<Error>(error->data).message.find("stream aborted") != std::string::npos);
    CHECK(joined_text(events) == "前半");
    CHECK(server.errors().empty());
}

TEST_CASE("OpenAI 流式 idle 超时：流中停顿超限 → NetworkError")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    // 只发一个事件（无 [DONE]），随后服务器停顿 > idle 才关连接——
    // 引擎必然在等待剩余 body 时触发 idle 超时（确定性，不依赖竞态）
    std::string sse = "data: {\"choices\":[{\"delta\":{\"content\":\"start\"}}]}\n\n";
    std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
    std::vector<agent::test::MockServer::Chunk> chunks;
    chunks.push_back({ { head + sse } });
    chunks.push_back({ { "" }, 1000 });   // 停顿 1s 超 idle 200ms
    server.enqueue({ {}, std::move(chunks) });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    StreamOptions opts;
    opts.idle_timeout_ms = 200;
    opts.max_retries = 0;
    auto events = collect_stream(provider, *model, simple_ctx("hi"), opts);
    auto error = std::find_if(events.begin(), events.end(), [](auto const& e) {
        return e.type() == StreamEvent::Type::Error;
    });
    REQUIRE(error != events.end());
    CHECK(std::get<Error>(error->data).code == Errc::NetworkError);
}

TEST_CASE("DeepSeek 同步非流式 complete")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("openai_text_4.sse")) });

    DeepSeekProvider provider({ .name = "deepseek", .api_key = "k", .base_url = server.base_url() });
    auto result = provider.complete(*model, simple_ctx("hi"), StreamOptions{});
    REQUIRE(result.has_value());
    CHECK(result->stop_reason == StopReason::Stop);
    CHECK(!result->content.empty());
    CHECK(server.errors().empty());
}

TEST_CASE("DeepSeek 工具调用闭环：thinking 回传 reasoning_content")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-effort");
    REQUIRE(model.has_value());

    // 注册真实工具（测试共享，先查后注册）
    ensure_get_weather_tool();

    // DeepSeek 思考模式工具调用流：reasoning_content + tool_calls + finish
    std::string sse =
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"\xe8\x80\x83\xe8\x99\x91\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_ds\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"NYC\\\"}\"}}]}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"
        "data: [DONE]\n\n";
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(sse) });

    DeepSeekProvider provider({ .name = "deepseek", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("北京天气"));
    ToolCall tool_call;
    bool found = false;
    for (auto const& event : events) {
        if (event.type() == StreamEvent::Type::ToolCallEnd) {
            auto const& end = std::get<ToolCallEnd>(event.data);
            tool_call = ToolCall{ end.id, end.name, end.arguments };
            found = true;
        }
    }
    REQUIRE(found);
    CHECK(tool_call.name == "get_weather");
    CHECK(tool_call.arguments["city"] == "NYC");

    // 真实执行 + 回传（assistant 带 thinking "考虑" + tool_calls + ToolResult）
    auto exec_result = Tools::exec(tool_call.name, tool_call.arguments);
    REQUIRE(exec_result.has_value());
    Context round2 = simple_ctx("北京天气");
    Message assistant{ Role::Assistant, { Thinking{ "考虑" }, tool_call } };
    round2.messages.push_back(std::move(assistant));
    round2.messages.push_back(Message{ Role::ToolResult, { ToolResult{ tool_call.id, exec_result.value(), false } } });

    // 第二轮断言：assistant 消息带 reasoning_content（DeepSeek 官方要求，否则 400）
    server.enqueue({
        [](agent::test::RequestView const& request) {
            nlohmann::json body = nlohmann::json::parse(request.body, nullptr, false);
            REQUIRE(body.is_object());
            REQUIRE(body.contains("messages"));
            REQUIRE(body["messages"].is_array());
            REQUIRE(body["messages"].size() >= 4);
            auto const& assistant_msg = body["messages"][2];
            CHECK(assistant_msg["role"] == "assistant");
            CHECK(assistant_msg.contains("reasoning_content"));
            CHECK(assistant_msg["reasoning_content"] == "考虑");
            REQUIRE(assistant_msg.contains("tool_calls"));
            CHECK(assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather");
            auto const& tool_msg = body["messages"][3];
            CHECK(tool_msg["role"] == "tool");
            CHECK(tool_msg["tool_call_id"] == "call_ds");
        },
        sse_response(load_fixture("openai_text_4.sse")),
    });
    auto result = provider.complete(*model, round2, StreamOptions{});
    REQUIRE(result.has_value());
    CHECK(result->stop_reason == StopReason::Stop);
    CHECK(server.errors().empty());
    CHECK(server.request_count() == 2);
}

TEST_CASE("DeepSeek 429 → RateLimited")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(R"({"error":{"message":"rate limited"}})", 429) });

    DeepSeekProvider provider({ .name = "deepseek", .api_key = "k", .base_url = server.base_url() });
    StreamOptions opts;
    opts.max_retries = 0;
    auto events = collect_stream(provider, *model, simple_ctx("hi"), opts);
    auto error = std::find_if(events.begin(), events.end(), [](auto const& e) {
        return e.type() == StreamEvent::Type::Error;
    });
    REQUIRE(error != events.end());
    CHECK(std::get<Error>(error->data).code == Errc::RateLimited);
    CHECK(server.request_count() == 1);
}

// ───────────────────── 真实 DeepSeek fixture（真实 API 录制）─────────────────────

TEST_CASE("DeepSeek 真实 thinking 流 fixture：reasoning_content + 文本 + usage")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-effort");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("deepseek_thinking.sse", "deepseek")) });

    DeepSeekProvider provider({ .name = "deepseek", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    // reasoning_content delta → ThinkingDelta
    std::string thinking;
    for (auto const& e : events)
        if (e.type() == StreamEvent::Type::ThinkingDelta)
            thinking += std::get<ThinkingDelta>(e.data).text;
    CHECK(thinking.size() > 0);
    CHECK(joined_text(events).size() > 0);
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    // 聚合内容含 Thinking 块
    bool has_thinking = false;
    for (auto const& block : done->content)
        if (std::get_if<Thinking>(&block)) has_thinking = true;
    CHECK(has_thinking);
    CHECK(done->usage.input_tokens > 0);
    CHECK(server.errors().empty());
}

TEST_CASE("DeepSeek 真实工具调用闭环 fixture：两轮 + reasoning_content 回传")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-effort");
    REQUIRE(model.has_value());

    // 注册真实工具（测试共享，先查后注册）
    ensure_get_weather_tool();

    // round1：模型真实决定调用 get_weather（{city:北京}）
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("deepseek_tool_round1.sse", "deepseek")) });

    DeepSeekProvider provider({ .name = "deepseek", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("北京天气怎么样"));
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::ToolUse);

    // 取 ToolCall + thinking（round1 有 reasoning_content）
    ToolCall tool_call;
    bool found = false;
    std::string thinking;
    for (auto const& e : events) {
        if (e.type() == StreamEvent::Type::ToolCallEnd) {
            auto const& end = std::get<ToolCallEnd>(e.data);
            tool_call = ToolCall{ end.id, end.name, end.arguments };
            found = true;
        } else if (e.type() == StreamEvent::Type::ThinkingDelta) {
            thinking += std::get<ThinkingDelta>(e.data).text;
        }
    }
    REQUIRE(found);
    CHECK(tool_call.name == "get_weather");
    CHECK(tool_call.arguments["city"] == "北京");
    CHECK(thinking.size() > 0);

    // 真实执行 + 回传（assistant 带 thinking → reasoning_content）
    auto exec_result = Tools::exec(tool_call.name, tool_call.arguments);
    REQUIRE(exec_result.has_value());
    Context round2 = simple_ctx("北京天气怎么样");
    Message assistant{ Role::Assistant, { Thinking{ thinking }, tool_call } };
    round2.messages.push_back(std::move(assistant));
    round2.messages.push_back(Message{ Role::ToolResult, { ToolResult{ tool_call.id, exec_result.value(), false } } });

    // round2：模型基于工具结果最终回答；断言请求体带 reasoning_content + tool_calls 回传
    server.enqueue({
        [](agent::test::RequestView const& request) {
            nlohmann::json body = nlohmann::json::parse(request.body, nullptr, false);
            REQUIRE(body.is_object());
            REQUIRE(body.contains("messages"));
            REQUIRE(body["messages"].is_array());
            REQUIRE(body["messages"].size() >= 4);
            auto const& assistant_msg = body["messages"][2];
            CHECK(assistant_msg["role"] == "assistant");
            CHECK(assistant_msg.contains("reasoning_content"));
            CHECK(assistant_msg["reasoning_content"].get<std::string>().size() > 0);
            REQUIRE(assistant_msg.contains("tool_calls"));
            CHECK(assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather");
            auto const& tool_msg = body["messages"][3];
            CHECK(tool_msg["role"] == "tool");
            CHECK(tool_msg["tool_call_id"].is_string());
        },
        sse_response(load_fixture("deepseek_tool_round2.sse", "deepseek")),
    });
    auto result = provider.complete(*model, round2, StreamOptions{});
    REQUIRE(result.has_value());
    CHECK(result->stop_reason == StopReason::Stop);
    CHECK(text_of_blocks(result->content).size() > 0);
    CHECK(server.errors().empty());
    CHECK(server.request_count() == 2);
}

// ───────────────────── 真实第三方端点 fixture（真实 API 录制）─────────────────────

TEST_CASE("OpenAICompatibleProvider 真实 NVIDIA fixture：文本流")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("nvidia_text.sse", "nvidia")) });

    OpenAICompatibleProvider provider({ .name = "nvidia", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    CHECK(joined_text(events).size() > 0);
    CHECK(done->usage.total_tokens > 0);
    CHECK(server.errors().empty());
}

TEST_CASE("OpenAICompatibleProvider 真实 NVIDIA fixture：工具调用闭环")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());

    // 注册真实工具（测试共享，先查后注册）
    ensure_get_weather_tool();

    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("nvidia_tool_round1.sse", "nvidia")) });

    OpenAICompatibleProvider provider({ .name = "nvidia", .api_key = "k", .base_url = server.base_url() });
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

    server.enqueue({
        [](agent::test::RequestView const& request) {
            nlohmann::json body = nlohmann::json::parse(request.body, nullptr, false);
            REQUIRE(body.is_object());
            REQUIRE(body.contains("messages"));
            REQUIRE(body["messages"].is_array());
            REQUIRE(body["messages"].size() >= 4);
            auto const& assistant_msg = body["messages"][2];
            CHECK(assistant_msg["role"] == "assistant");
            REQUIRE(assistant_msg.contains("tool_calls"));
            CHECK(assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather");
            CHECK(body["messages"][3]["role"] == "tool");
        },
        sse_response(load_fixture("nvidia_tool_round2.sse", "nvidia")),
    });
    auto result = provider.complete(*model, round2, StreamOptions{});
    REQUIRE(result.has_value());
    CHECK(result->stop_reason == StopReason::Stop);
    CHECK(text_of_blocks(result->content).size() > 0);
    CHECK(server.errors().empty());
    CHECK(server.request_count() == 2);
}

// ───────────────────── 难样例 ─────────────────────
namespace {

/// 每个 SSE 事件从中间切一刀成两个 Chunk（半包形态）。
std::vector<agent::test::MockServer::Chunk> half_split(std::string const& sse)
{
    std::vector<agent::test::MockServer::Chunk> chunks;
    std::size_t pos = 0;
    while (true) {
        std::size_t sep = sse.find("\n\n", pos);
        if (sep == std::string::npos) {
            if (pos < sse.size())
                chunks.push_back({ { sse.substr(pos) } });
            break;
        }
        std::string event = sse.substr(pos, sep - pos);
        std::size_t mid = event.size() / 2;
        chunks.push_back({ { event.substr(0, mid) } });
        chunks.push_back({ { event.substr(mid) + "\n\n" } });
        pos = sep + 2;
    }
    return chunks;
}

}  // namespace

TEST_CASE("难样例：SSE 半包（每事件中间切一刀）完整消费")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    auto chunks = half_split(load_fixture("openai_text_4.sse"));
    std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
    chunks.front().bytes = head + chunks.front().bytes;
    server.enqueue({ {}, chunks });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    CHECK(joined_text(events).size() > 0);
    CHECK(server.errors().empty());
}

TEST_CASE("难样例：UTF-8 中文字节跨 TCP 包拼接正确")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    // 一个完整 data 事件含「你好」；TCP 包在「好」的 UTF-8 首字节 \xe5 前切开
    // （SseParser 按 \n\n 切事件，事件内部字节跨包不影响 JSON 完整性）
    std::string event1 = "data: {\"choices\":[{\"delta\":{\"content\":\"\xe4\xbd\xa0\xe5\xa5\xbd\"}}]}\n\n";
    std::string event2 = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";
    std::string event3 = "data: [DONE]\n\n";
    std::string full = event1 + event2 + event3;
    std::size_t hao_pos = full.find("\xe5\xa5\xbd");
    REQUIRE(hao_pos != std::string::npos);
    std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
    agent::test::MockServer server;
    server.enqueue({ {}, { { { head + full.substr(0, hao_pos) } }, { { full.substr(hao_pos) } } } });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    CHECK(joined_text(events) == "你好");
    CHECK(server.errors().empty());
}

TEST_CASE("难样例：流中断连（RST）→ Error 事件")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    // 只发一半就 RST 断连
    std::string half = load_fixture("openai_text_4.sse").substr(0, 256);
    server.enqueue({ {}, { { "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n" + half } },
                          true /* close_abruptly */ });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    auto error = std::find_if(events.begin(), events.end(), [](auto const& e) {
        return e.type() == StreamEvent::Type::Error;
    });
    REQUIRE(error != events.end());
    CHECK(std::get<Error>(error->data).code == Errc::NetworkError);
}

TEST_CASE("难样例：异步取消中断流 → Error")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    // 慢滴流：每块 30ms，取消发生在前几个块后
    std::vector<agent::test::MockServer::Chunk> slow;
    std::string sse = load_fixture("openai_text_4.sse");
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

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    asio::io_context io;
    asio::cancellation_signal cancel;
    std::vector<StreamEvent> events;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        StreamOptions opts;
        opts.cancel = &cancel;
        AsyncStream<StreamEvent> sink(io.get_executor());
        auto local = sink;
        asio::co_spawn(io, [&]() -> asio::awaitable<void> {
            co_await provider.stream_async(*model, simple_ctx("hi"), opts, std::move(local));
        }, asio::detached);
        int received = 0;
        while (received < 2) {
            auto event = co_await sink.receive();
            if (!event) break;
            events.push_back(std::move(*event));
            ++received;
        }
        cancel.emit(asio::cancellation_type::all);
        while (auto event = co_await sink.receive())
            events.push_back(std::move(*event));
    }, asio::detached);
    io.run();
    auto error = std::find_if(events.begin(), events.end(), [](auto const& e) {
        return e.type() == StreamEvent::Type::Error;
    });
    REQUIRE(error != events.end());
    CHECK(std::get<Error>(error->data).code == Errc::NetworkError);
}

// ───────────────────── session affinity 头─────────────────────

TEST_CASE("session affinity 头：openai 三头格式")
{
    std::vector<std::pair<std::string, std::string>> headers;
    add_session_affinity_headers(headers, "openai", "sess-1");
    CHECK(headers.size() == 3);
    auto find = [&](std::string_view name) {
        for (auto const& [k, v] : headers)
            if (k == name) return v;
        return std::string{};
    };
    CHECK(find("session_id") == "sess-1");
    CHECK(find("x-client-request-id") == "sess-1");
    CHECK(find("x-session-affinity") == "sess-1");
}

TEST_CASE("session affinity 头：openai-nosession 两头格式（无 session_id）")
{
    std::vector<std::pair<std::string, std::string>> headers;
    add_session_affinity_headers(headers, "openai-nosession", "sess-1");
    CHECK(headers.size() == 2);
    auto find = [&](std::string_view name) {
        for (auto const& [k, v] : headers)
            if (k == name) return v;
        return std::string{};
    };
    CHECK(find("session_id").empty());
    CHECK(find("x-client-request-id") == "sess-1");
    CHECK(find("x-session-affinity") == "sess-1");
}

TEST_CASE("session affinity 头：openrouter 单头格式")
{
    std::vector<std::pair<std::string, std::string>> headers;
    add_session_affinity_headers(headers, "openrouter", "sess-1");
    CHECK(headers.size() == 1);
    CHECK(headers[0].first == "x-session-id");
    CHECK(headers[0].second == "sess-1");
}

TEST_CASE("默认 Provider：cache_retention + session_id 只发 prompt_cache_key，不发 session 头")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({
        [](agent::test::RequestView const& request) {
            // Compat 默认 send_session_affinity=false → 不发 session 头
            CHECK(request.header("session_id").empty());
            CHECK(request.header("x-session-affinity").empty());
            CHECK(request.header("x-client-request-id").empty());
            // 但 prompt_cache_key 照发（缓存 key 与 session 头独立）
            nlohmann::json body = nlohmann::json::parse(request.body, nullptr, false);
            REQUIRE(body.is_object());
            CHECK(body["prompt_cache_key"] == "sess-1");
            CHECK(body["prompt_cache_retention"] == "24h");
        },
        sse_response(load_fixture("openai_text_4.sse")),
    });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    StreamOptions opts;
    opts.cache_retention = CacheRetention::Long;
    opts.session_id = "sess-1";
    auto events = collect_stream(provider, *model, simple_ctx("hi"), opts);
    REQUIRE(done_response(events).has_value());
    CHECK(server.errors().empty());
}

// ───────────────────── 真实环境鲁棒性（确定性，不依赖竞态）─────────────────────

TEST_CASE("鲁棒性：畸形 chunk（非法 JSON）跳过，后续正常")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    std::string sse =
        "data: {not valid json\n\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
        "data: [DONE]\n\n";
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(sse) });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    CHECK(joined_text(events) == "ok");   // 畸形 chunk 被跳过，后续正常
    CHECK(server.errors().empty());
}

TEST_CASE("鲁棒性：空流（仅 [DONE]）→ Done 空内容")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response("data: [DONE]\n\n") });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    CHECK(done->content.empty());
    CHECK(server.errors().empty());
}

TEST_CASE("鲁棒性：服务器 EOF 未发 [DONE] → Done（EOF 兜底）")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    // 无 [DONE] 无 finish_reason，服务器优雅关闭 → 引擎 EOF 兜底 Done
    std::string sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(sse) });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(joined_text(events) == "hi");
    CHECK(server.errors().empty());
}

TEST_CASE("鲁棒性：429 首字节前重试 → 200 成功")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(R"({"error":{"message":"slow"}})", 429) });
    server.enqueue({ {}, sse_response(load_fixture("openai_text_4.sse")) });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    StreamOptions opts;
    opts.max_retries = 2;
    auto events = collect_stream(provider, *model, simple_ctx("hi"), opts);
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    CHECK(server.request_count() == 2);   // 恰好重试一次
    CHECK(server.errors().empty());
}

// ───────────────────── 随机猴子测试（确定性种子，可复现稳定）─────────────────────

namespace {

/// 在任意字节位置随机切分 SSE（含事件内部 / \n\n 中间），第一个 chunk 前缀 HTTP 头。
std::vector<agent::test::MockServer::Chunk> random_split(std::string const& sse, std::mt19937& rng)
{
    std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
    std::uniform_int_distribution<std::size_t> cut_count(1, 5);
    std::uniform_int_distribution<std::size_t> pos(0, sse.size());
    std::uniform_int_distribution<int> delay(0, 5);
    std::vector<std::size_t> cuts;
    int n = static_cast<int>(cut_count(rng));
    for (int i = 0; i < n; ++i)
        cuts.push_back(pos(rng));
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());

    std::vector<agent::test::MockServer::Chunk> chunks;
    std::size_t prev = 0;
    for (std::size_t c : cuts) {
        if (c > prev)
            chunks.push_back({ { sse.substr(prev, c - prev) }, delay(rng) });
        prev = c;
    }
    if (prev < sse.size())
        chunks.push_back({ { sse.substr(prev) }, delay(rng) });
    chunks.front().bytes = head + chunks.front().bytes;
    return chunks;
}

}  // namespace

TEST_CASE("猴子测试：文本 fixture 随机切分（多种子）→ 与整包解析一致")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    std::string sse = load_fixture("openai_text_4.sse");

    // 整包基准
    agent::test::MockServer base;
    base.enqueue({ {}, sse_response(sse) });
    OpenAIProvider base_provider({ .name = "openai", .api_key = "k", .base_url = base.base_url() });
    std::string base_text = joined_text(collect_stream(base_provider, *model, simple_ctx("hi")));
    REQUIRE(base_text.size() > 0);

    for (unsigned seed : { 1u, 2u, 3u, 42u, 99u, 123u }) {
        std::mt19937 rng(seed);
        agent::test::MockServer fuzz;
        fuzz.enqueue({ {}, random_split(sse, rng) });
        OpenAIProvider fuzz_provider({ .name = "openai", .api_key = "k", .base_url = fuzz.base_url() });
        auto events = collect_stream(fuzz_provider, *model, simple_ctx("hi"));
        CHECK(joined_text(events) == base_text);   // 任意切分文本一致
        auto done = done_response(events);
        CHECK(done.has_value());
        if (done) CHECK(done->stop_reason == StopReason::Stop);
        CHECK(fuzz.errors().empty());
    }
}

TEST_CASE("猴子测试：工具 fixture 随机切分 → ToolCallEnd 参数与整包一致")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    std::string sse = load_fixture("openai_tool_1.sse");

    auto extract_tool = [](std::vector<StreamEvent> const& events) -> ToolCall {
        ToolCall tool;
        for (auto const& e : events)
            if (e.type() == StreamEvent::Type::ToolCallEnd) {
                auto const& end = std::get<ToolCallEnd>(e.data);
                tool = ToolCall{ end.id, end.name, end.arguments };
            }
        return tool;
    };

    agent::test::MockServer base;
    base.enqueue({ {}, sse_response(sse) });
    OpenAIProvider base_provider({ .name = "openai", .api_key = "k", .base_url = base.base_url() });
    ToolCall base_tool = extract_tool(collect_stream(base_provider, *model, simple_ctx("hi")));
    REQUIRE(base_tool.name == "get_weather");

    for (unsigned seed : { 7u, 13u, 21u }) {
        std::mt19937 rng(seed);
        agent::test::MockServer fuzz;
        fuzz.enqueue({ {}, random_split(sse, rng) });
        OpenAIProvider fuzz_provider({ .name = "openai", .api_key = "k", .base_url = fuzz.base_url() });
        ToolCall tool = extract_tool(collect_stream(fuzz_provider, *model, simple_ctx("hi")));
        CHECK(tool.name == base_tool.name);
        CHECK(tool.arguments == base_tool.arguments);   // 跨任意切分参数完整
        CHECK(fuzz.errors().empty());
    }
}

TEST_CASE("猴子测试：随机注入畸形 chunk → 不崩，正常终结")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    std::string sse = load_fixture("openai_text_4.sse");
    std::mt19937 rng(2026);

    for (int round = 0; round < 10; ++round) {
        // 在随机位置插入畸形 data 行（非法 JSON / 截断 JSON）
        std::string mutated = sse;
        std::uniform_int_distribution<std::size_t> pos(0, mutated.size());
        std::uniform_int_distribution<int> kind(0, 2);
        std::size_t at = pos(rng);
        std::string junk;
        switch (kind(rng)) {
            case 0: junk = "data: {oops\n\n"; break;
            case 1: junk = "data: {\"choices\":[{\"delta\":{\"content\":\""; break;   // 截断
            default: junk = "data: [DONE\n\n"; break;                                  // 残缺哨兵
        }
        mutated.insert(at, junk);

        agent::test::MockServer fuzz;
        fuzz.enqueue({ {}, sse_response(mutated) });
        OpenAIProvider fuzz_provider({ .name = "openai", .api_key = "k", .base_url = fuzz.base_url() });
        // 必须不抛、正常终结（Done 或 Error 皆可），不得崩溃/死锁
        auto events = collect_stream(fuzz_provider, *model, simple_ctx("hi"));
        bool terminal = done_response(events).has_value()
            || std::any_of(events.begin(), events.end(), [](auto const& e) {
                return e.type() == StreamEvent::Type::Error;
            });
        CHECK(terminal);
        CHECK(fuzz.errors().empty());
    }
}

TEST_CASE("猴子测试：build_params 随机参数组合 → 不抛 + 结构合法")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-effort");
    REQUIRE(model.has_value());
    std::mt19937 rng(777);
    std::uniform_int_distribution<int> flag(0, 1);

    for (int i = 0; i < 50; ++i) {
        StreamOptions opts;
        std::uniform_real_distribution<double> temp(0.0, 2.0);
        if (flag(rng)) opts.temperature = temp(rng);
        std::uniform_int_distribution<int> tokens(1, 16384);
        if (flag(rng)) opts.max_tokens = tokens(rng);
        if (flag(rng)) opts.reasoning = static_cast<ThinkingLevel>(std::uniform_int_distribution<int>(0, 6)(rng));
        if (flag(rng)) opts.cache_retention = static_cast<CacheRetention>(std::uniform_int_distribution<int>(0, 2)(rng));
        if (flag(rng)) opts.session_id = "session-" + std::to_string(i);
        if (flag(rng)) opts.extra = { { "custom", i }, { "nested", { { "k", true } } } };

        // 不抛异常即通过；结构必须合法
        auto params = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(*model, simple_ctx("hi"), opts);
        CHECK(params["model"] == "t-effort");
        CHECK(params["messages"].is_array());
        CHECK(params["stream"] == true);
        if (opts.max_tokens.has_value())
            CHECK(params["max_completion_tokens"] == *opts.max_tokens);
        else
            CHECK(!params.contains("max_completion_tokens"));
    }
}

TEST_CASE("OpenAICompatibleProvider：第三方兼容端点流式 + max_tokens 字段")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    // expect 断言请求体用 max_tokens（第三方端点通用字段，非 max_completion_tokens）
    server.enqueue({
        [](agent::test::RequestView const& request) {
            nlohmann::json body = nlohmann::json::parse(request.body, nullptr, false);
            REQUIRE(body.is_object());
            CHECK(body.contains("max_tokens"));
            CHECK(!body.contains("max_completion_tokens"));
        },
        sse_response(load_fixture("openai_text_4.sse")),
    });

    OpenAICompatibleProvider provider({ .name = "vllm", .api_key = "k", .base_url = server.base_url() });
    StreamOptions opts;
    opts.max_tokens = 2048;
    auto events = collect_stream(provider, *model, simple_ctx("hi"), opts);
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    CHECK(joined_text(events).size() > 0);
    CHECK(server.errors().empty());
}

// ───────────────────── T2 契约 dump（AGENT_CONTRACT=ON 时编译）─────────────────────
#ifdef AGENT_CONTRACT_OUT

TEST_CASE("contract: dump build_params 工具场景")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    Context ctx;
    ctx.messages.push_back(Message{ Role::User, { Text{ "hi" } } });
    ensure_get_weather_tool();
    ctx.tools.push_back("get_weather");
    auto params = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(*model, ctx, {});
    std::filesystem::path dir = AGENT_CONTRACT_OUT;
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "tool_1.json") << params.dump(2);
}

TEST_CASE("contract: dump build_params 文本场景")
{
    ensure_test_models();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());
    Context ctx;
    ctx.messages.push_back(Message{ Role::User, { Text{ "hi" } } });
    auto params = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(*model, ctx, {});
    std::filesystem::path dir = AGENT_CONTRACT_OUT;
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "text_4.json") << params.dump(2);
}

#endif
