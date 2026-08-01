// Gemini（gemma）引擎测试。
//
// T0 纯函数：build_params / parse_chunk（thought part / functionCall / finishReason）
// T1 MockServer 回放真实录制 fixture（tests/fixtures/gemini/*.sse，真实 API 录制）

#include <agent/llm/model.hpp>
#include <agent/llm/providers/gemini.hpp>
#include <agent/tools/tools.hpp>

#include "../core/mock_server.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace agent;
using namespace agent::detail;

namespace {

std::string load_fixture(std::string const& name)
{
    std::filesystem::path dir = AGENT_TEST_FIXTURES_DIR;
    std::ifstream in(dir / "gemini" / name, std::ios::binary);
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

void ensure_gemma_model()
{
    static bool done = [] {
        ModelRegistry::register_model(RuntimeModel{
            .id = "gemma-4-26b-a4b-it",
            .context_window = 128000, .max_output_tokens = 8192,
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

TEST_CASE("gemini build_params：contents/systemInstruction/tools/generationConfig")
{
    ensure_gemma_model();
    auto model = ModelRegistry::find_model("gemma-4-26b-a4b-it");
    REQUIRE(model.has_value());
    Context ctx = simple_ctx("hi");
    ctx.system_prompt = "你是助手";
    ctx.tools.push_back(ToolInfo{
        .name = "get_weather", .description = "查天气",
        .parameters = { { "type", "object" },
                        { "properties", { { "city", { { "type", "string" } } } } },
                        { "required", { "city" } } },
    });
    StreamOptions opts;
    opts.max_tokens = 512;
    opts.temperature = 0.5;

    auto params = GeminiGenerateContentEngine::build_params(*model, ctx, opts);
    CHECK(params["systemInstruction"]["parts"][0]["text"] == "你是助手");
    REQUIRE(params["contents"].is_array());
    CHECK(params["contents"][0]["role"] == "user");
    CHECK(params["contents"][0]["parts"][0]["text"] == "hi");
    REQUIRE(params["tools"].is_array());
    CHECK(params["tools"][0]["functionDeclarations"][0]["name"] == "get_weather");
    CHECK(params["generationConfig"]["maxOutputTokens"] == 512);
    CHECK(params["generationConfig"]["temperature"] == 0.5);
}

TEST_CASE("gemini build_params：thinking → thinkingConfig.thinkingBudget")
{
    ensure_gemma_model();
    auto model = ModelRegistry::find_model("gemma-4-26b-a4b-it");
    REQUIRE(model.has_value());
    StreamOptions opts;
    opts.reasoning = ThinkingLevel::High;
    auto params = GeminiGenerateContentEngine::build_params(*model, simple_ctx("hi"), opts);
    // gemma map High=8192 → thinkingBudget
    CHECK(params["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 8192);
}

TEST_CASE("gemini build_params：不传采样参数不上传")
{
    ensure_gemma_model();
    auto model = ModelRegistry::find_model("gemma-4-26b-a4b-it");
    REQUIRE(model.has_value());
    auto params = GeminiGenerateContentEngine::build_params(*model, simple_ctx("hi"), {});
    CHECK(!params.contains("maxOutputTokens"));
    CHECK(!params.contains("temperature"));
}

// ───────────────────── T0: parse_chunk ─────────────────────

TEST_CASE("gemini parse_chunk：thought part → ThinkingDelta，text → TextDelta")
{
    GeminiStreamState state;
    nlohmann::json chunk;
    chunk["responseId"] = "resp_1";
    chunk["candidates"] = nlohmann::json::array();
    chunk["candidates"].push_back({ { "index", 0 },
                                    { "content", { { "parts", { { { "text", "思考中" }, { "thought", true } },
                                                                 { { "text", "正式回答" } } } } } } });
    auto events = GeminiGenerateContentEngine::parse_chunk(state, chunk);
    CHECK(state.response_id == "resp_1");
    REQUIRE(events.size() == 2);
    CHECK(events[0].type() == StreamEvent::Type::ThinkingDelta);
    CHECK(std::get<ThinkingDelta>(events[0].data).text == "思考中");
    CHECK(events[1].type() == StreamEvent::Type::TextDelta);
    CHECK(std::get<TextDelta>(events[1].data).text == "正式回答");
    CHECK(state.thinking == "思考中");
    CHECK(state.text == "正式回答");
}

TEST_CASE("gemini parse_chunk：functionCall 一次到达 → ToolCallEnd")
{
    GeminiStreamState state;
    nlohmann::json chunk;
    chunk["candidates"] = nlohmann::json::array();
    chunk["candidates"].push_back({ { "index", 0 },
                                    { "content", { { "parts", { { { "functionCall",
                                        { { "name", "get_weather" }, { "id", "call_1" },
                                          { "args", { { "city", "北京" } } } } } } } } } },
                                    { "finishReason", "STOP" } });
    auto events = GeminiGenerateContentEngine::parse_chunk(state, chunk);
    REQUIRE(events.size() == 1);
    CHECK(events[0].type() == StreamEvent::Type::ToolCallEnd);
    auto const& end = std::get<ToolCallEnd>(events[0].data);
    CHECK(end.name == "get_weather");
    CHECK(end.id == "call_1");
    CHECK(end.arguments["city"] == "北京");
    CHECK(state.stop_reason == StopReason::Stop);
}

TEST_CASE("gemini parse_chunk：finishReason 映射 + usageMetadata")
{
    GeminiStreamState state;
    nlohmann::json chunk;
    chunk["candidates"] = nlohmann::json::array();
    chunk["candidates"].push_back({ { "index", 0 }, { "content", { { "parts", nlohmann::json::array() } } },
                                    { "finishReason", "MAX_TOKENS" } });
    chunk["usageMetadata"] = { { "promptTokenCount", 12 }, { "candidatesTokenCount", 8 },
                               { "totalTokenCount", 20 }, { "cachedContentTokenCount", 6 } };
    auto events = GeminiGenerateContentEngine::parse_chunk(state, chunk);
    CHECK(state.stop_reason == StopReason::Length);
    CHECK(state.usage.input_tokens == 12);
    CHECK(state.usage.output_tokens == 8);
    CHECK(state.usage.cache_read_tokens == 6);
}

TEST_CASE("gemini parse_chunk：SAFETY → Error 终止")
{
    GeminiStreamState state;
    nlohmann::json chunk;
    chunk["candidates"] = nlohmann::json::array();
    chunk["candidates"].push_back({ { "index", 0 }, { "content", { { "parts", nlohmann::json::array() } } },
                                    { "finishReason", "SAFETY" } });
    GeminiGenerateContentEngine::parse_chunk(state, chunk);
    CHECK(state.stop_reason == StopReason::Error);
    CHECK(state.error_message.find("SAFETY") != std::string::npos);
}

// ───────────────────── T1: 真实 fixture 回放 ─────────────────────

TEST_CASE("Gemini 真实 gemma 文本 fixture：思考 + 正文 + usage")
{
    ensure_gemma_model();
    auto model = ModelRegistry::find_model("gemma-4-26b-a4b-it");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("gemma_text.sse")) });

    GeminiGenerateContentProvider provider({ .name = "gemini", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("hi"));
    std::string thinking;
    for (auto const& e : events)
        if (e.type() == StreamEvent::Type::ThinkingDelta)
            thinking += std::get<ThinkingDelta>(e.data).text;
    CHECK(thinking.size() > 0);                    // gemma 思考模型：thought part 已解析
    CHECK(joined_text(events).size() > 0);
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    CHECK(done->usage.total_tokens > 0);
    bool has_thinking_block = false;
    for (auto const& b : done->content)
        if (std::get_if<Thinking>(&b)) has_thinking_block = true;
    CHECK(has_thinking_block);
    CHECK(server.errors().empty());
}

TEST_CASE("Gemini 真实 gemma 工具调用闭环 fixture：两轮")
{
    ensure_gemma_model();
    auto model = ModelRegistry::find_model("gemma-4-26b-a4b-it");
    REQUIRE(model.has_value());

    // 注册真实工具（测试共享，先查后注册）
    ensure_get_weather_tool();

    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("gemma_tool_round1.sse")) });

    GeminiGenerateContentProvider provider({ .name = "gemini", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, simple_ctx("北京天气怎么样"));
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);

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

    // round2 断言请求体：model functionCall + function functionResponse（Gemini 多轮格式）
    server.enqueue({
        [](agent::test::RequestView const& request) {
            nlohmann::json body = nlohmann::json::parse(request.body, nullptr, false);
            REQUIRE(body.is_object());
            REQUIRE(body.contains("contents"));
            REQUIRE(body["contents"].is_array());
            REQUIRE(body["contents"].size() >= 3);
            auto const& model_msg = body["contents"][1];
            CHECK(model_msg["role"] == "model");
            REQUIRE(model_msg["parts"][0].contains("functionCall"));
            CHECK(model_msg["parts"][0]["functionCall"]["name"] == "get_weather");
            auto const& fn_msg = body["contents"][2];
            CHECK(fn_msg["role"] == "function");
            CHECK(fn_msg["parts"][0]["functionResponse"]["name"] == "get_weather");
            CHECK(fn_msg["parts"][0]["functionResponse"]["response"]["output"].is_string());
        },
        sse_response(load_fixture("gemma_tool_round2.sse")),
    });
    auto result = provider.complete(*model, round2, StreamOptions{});
    REQUIRE(result.has_value());
    CHECK(result->stop_reason == StopReason::Stop);
    CHECK(server.errors().empty());
    CHECK(server.request_count() == 2);
}
