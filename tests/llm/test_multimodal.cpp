// 多模态（图片输入）测试。
//
// T0：三家引擎 build_params 的图片块格式（OpenAI image_url / Gemini inlineData /
//      Anthropic image block）+ tool_result 图片 + 模型不支持时的降级（占位符文本）。
// 模型能力差异统一由 ModelView.supports_image_input（模型表）承载，引擎无模型特判。

#include <agent/llm/engine/image_support.hpp>
#include <agent/llm/model.hpp>
#include <agent/llm/providers/anthropic.hpp>
#include <agent/llm/providers/gemini.hpp>
#include <agent/llm/providers/openai.hpp>

#include "../core/mock_server.hpp"

#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace agent;
using namespace agent::detail;

namespace {

// 1x1 透明 PNG（base64，测试图）
std::string const kPngBase64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

/// 视觉模型（OpenAI 引擎场景，supports_image_input=true）。
void ensure_vision_model()
{
    static bool done = [] {
        ModelRegistry::register_model(RuntimeModel{
            .id = "gpt-4o-vision", .context_window = 128000, .max_output_tokens = 4096,
            .reasoning = false, .supports_image_input = true,
        });
        return true;
    }();
    (void)done;
}

/// 非视觉模型（supports_image_input=false）。
void ensure_text_model()
{
    static bool done = [] {
        ModelRegistry::register_model(RuntimeModel{
            .id = "text-only", .context_window = 128000, .max_output_tokens = 4096,
            .reasoning = false, .supports_image_input = false,
        });
        return true;
    }();
    (void)done;
}

Context image_ctx(std::string text)
{
    Context ctx;
    ctx.messages.push_back(
        Message{ Role::User, { Text{ std::move(text) }, Image{ kPngBase64, "image/png" } } });
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

std::string load_fixture(std::string const& provider, std::string const& name)
{
    std::filesystem::path dir = AGENT_TEST_FIXTURES_DIR;
    std::ifstream in(dir / provider / name, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<agent::test::MockServer::Chunk> sse_response(std::string const& body)
{
    std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
    return { { { head + body } } };
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

}  // namespace

// ───────────────────── OpenAI 引擎 ─────────────────────

TEST_CASE("multimodal openai：user 图片 → content 数组 image_url data URL")
{
    ensure_vision_model();
    auto model = ModelRegistry::find_model("gpt-4o-vision");
    REQUIRE(model.has_value());
    Context ctx = image_ctx("图里是什么");
    auto params = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(*model, ctx, {});
    auto const& content = params["messages"][0]["content"];
    REQUIRE(content.is_array());
    REQUIRE(content.size() == 2);
    CHECK(content[0]["type"] == "text");
    CHECK(content[0]["text"] == "图里是什么");
    CHECK(content[1]["type"] == "image_url");
    CHECK(content[1]["image_url"]["url"]
          == "data:image/png;base64," + kPngBase64);
}

TEST_CASE("multimodal openai：tool_result 图片 → 独立 user 消息 Attached image(s)")
{
    ensure_vision_model();
    auto model = ModelRegistry::find_model("gpt-4o-vision");
    REQUIRE(model.has_value());
    Context ctx;
    ctx.messages.push_back(Message{ Role::User, { Text{ "看下截图" } } });
    ctx.messages.push_back(Message{ Role::ToolResult,
                                    { ToolResult{ "call_1", "ok", false }, Image{ kPngBase64, "image/png" } } });
    auto params = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(*model, ctx, {});
    auto const& messages = params["messages"];
    REQUIRE(messages.size() == 3);
    CHECK(messages[1]["role"] == "tool");
    CHECK(messages[1]["tool_call_id"] == "call_1");
    // 图片 → 独立 user 消息（文本 + image_url 块）
    CHECK(messages[2]["role"] == "user");
    REQUIRE(messages[2]["content"].is_array());
    CHECK(messages[2]["content"][0]["text"] == "Attached image(s) from tool result:");
    CHECK(messages[2]["content"][1]["type"] == "image_url");
    CHECK(messages[2]["content"][1]["image_url"]["url"].is_string());
}

TEST_CASE("multimodal 降级：非视觉模型 user 图片 → 占位符文本")
{
    ensure_text_model();
    auto model = ModelRegistry::find_model("text-only");
    REQUIRE(model.has_value());
    auto params = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(*model, image_ctx("hi"), {});
    auto const& content = params["messages"][0]["content"];
    CHECK(content.is_string());
    // 图片 → 占位符（文本在前 → 拼接；对齐 pi replaceImagesWithPlaceholder）
    CHECK(content == "hi" + std::string(kNonVisionUserImagePlaceholder));
}

TEST_CASE("multimodal 降级：非视觉模型 tool_result 图片 → 占位符 + 连续合并")
{
    ensure_text_model();
    auto model = ModelRegistry::find_model("text-only");
    REQUIRE(model.has_value());
    Context ctx;
    ctx.messages.push_back(Message{ Role::User, { Text{ "hi" } } });
    // 同一 tool_result 消息：两张图 + 文本 → 合并成一个占位符
    ctx.messages.push_back(Message{ Role::ToolResult,
                                    { ToolResult{ "call_1", "ok", false },
                                      Image{ kPngBase64, "image/png" }, Image{ kPngBase64, "image/png" } } });
    auto params = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(*model, ctx, {});
    auto const& messages = params["messages"];
    REQUIRE(messages.size() == 2);
    CHECK(messages[1]["role"] == "tool");
    // 图片全部被降级（tool 消息 content 是纯文本，无 Attached user 消息）
    CHECK(messages[1]["content"] == "ok");
    CHECK(messages.size() == 2);   // 无图片 user 消息产生
}

// ───────────────────── Gemini 引擎 ─────────────────────

TEST_CASE("multimodal gemini：user 图片 → parts inlineData")
{
    ensure_vision_model();
    auto model = ModelRegistry::find_model("gpt-4o-vision");
    REQUIRE(model.has_value());
    auto params = *GeminiGenerateContentEngine::build_params(*model, image_ctx("图里是什么"), {});
    auto const& parts = params["contents"][0]["parts"];
    REQUIRE(parts.is_array());
    REQUIRE(parts.size() == 2);
    CHECK(parts[0]["text"] == "图里是什么");
    REQUIRE(parts[1].contains("inlineData"));
    CHECK(parts[1]["inlineData"]["mimeType"] == "image/png");
    CHECK(parts[1]["inlineData"]["data"] == kPngBase64);
}

TEST_CASE("multimodal gemini：tool_result 图片 → functionResponse.parts 内嵌")
{
    ensure_vision_model();
    auto model = ModelRegistry::find_model("gpt-4o-vision");
    REQUIRE(model.has_value());
    Context ctx;
    ctx.messages.push_back(Message{ Role::User, { Text{ "看下" } } });
    ctx.messages.push_back(Message{ Role::Assistant, { ToolCall{ "call_1", "screenshot", { } } } });
    ctx.messages.push_back(Message{ Role::ToolResult,
                                    { ToolResult{ "call_1", "done", false }, Image{ kPngBase64, "image/png" } } });
    auto params = *GeminiGenerateContentEngine::build_params(*model, ctx, {});
    auto const& fn_part = params["contents"][2]["parts"][0]["functionResponse"];
    CHECK(fn_part["name"] == "screenshot");
    REQUIRE(fn_part.contains("parts"));
    CHECK(fn_part["parts"][0]["inlineData"]["mimeType"] == "image/png");
}

TEST_CASE("multimodal gemini：非视觉模型降级 → 占位符 + 无 parts 内嵌")
{
    ensure_text_model();
    auto model = ModelRegistry::find_model("text-only");
    REQUIRE(model.has_value());
    Context ctx;
    ctx.messages.push_back(Message{ Role::User, { Image{ kPngBase64, "image/png" } } });
    ctx.messages.push_back(Message{ Role::Assistant, { ToolCall{ "call_1", "screenshot", { } } } });
    ctx.messages.push_back(Message{ Role::ToolResult,
                                    { ToolResult{ "call_1", "done", false }, Image{ kPngBase64, "image/png" } } });
    auto params = *GeminiGenerateContentEngine::build_params(*model, ctx, {});
    // user 图片 → 占位符文本 part
    CHECK(params["contents"][0]["parts"][0]["text"] == kNonVisionUserImagePlaceholder);
    // tool_result 图片降级 → 无 parts 内嵌
    auto const& fn_part = params["contents"][2]["parts"][0]["functionResponse"];
    CHECK(!fn_part.contains("parts"));
}

// ───────────────────── Anthropic 引擎 ─────────────────────

TEST_CASE("multimodal anthropic：user 图片 → image block")
{
    ensure_vision_model();
    auto model = ModelRegistry::find_model("gpt-4o-vision");
    REQUIRE(model.has_value());
    auto params = *AnthropicMessagesEngine::build_params(*model, image_ctx("图里是什么"), {});
    auto const& content = params["messages"][0]["content"];
    REQUIRE(content.is_array());
    REQUIRE(content.size() == 2);
    CHECK(content[1]["type"] == "image");
    CHECK(content[1]["source"]["type"] == "base64");
    CHECK(content[1]["source"]["media_type"] == "image/png");
    CHECK(content[1]["source"]["data"] == kPngBase64);
}

TEST_CASE("multimodal anthropic：tool_result 图片 → content blocks 含 image")
{
    ensure_vision_model();
    auto model = ModelRegistry::find_model("gpt-4o-vision");
    REQUIRE(model.has_value());
    Context ctx;
    ctx.messages.push_back(Message{ Role::User, { Text{ "看下" } } });
    ctx.messages.push_back(Message{ Role::Assistant, { ToolCall{ "toolu_1", "screenshot", { } } } });
    ctx.messages.push_back(Message{ Role::ToolResult,
                                    { ToolResult{ "toolu_1", "done", false }, Image{ kPngBase64, "image/png" } } });
    auto params = *AnthropicMessagesEngine::build_params(*model, ctx, {});
    auto const& tr_block = params["messages"][2]["content"][0];
    CHECK(tr_block["type"] == "tool_result");
    REQUIRE(tr_block["content"].is_array());
    CHECK(tr_block["content"][0]["type"] == "text");
    CHECK(tr_block["content"][1]["type"] == "image");
    CHECK(tr_block["content"][1]["source"]["media_type"] == "image/png");
}

TEST_CASE("multimodal anthropic：非视觉模型降级 → 占位符")
{
    ensure_text_model();
    auto model = ModelRegistry::find_model("text-only");
    REQUIRE(model.has_value());
    auto params = *AnthropicMessagesEngine::build_params(*model, image_ctx("hi"), {});
    // 图片 → 占位符（文本在前 → 拼接成字符串 content）
    CHECK(params["messages"][0]["content"] == "hi" + std::string(kNonVisionUserImagePlaceholder));
}

// ───────────────────── 降级 helper 单测 ─────────────────────

TEST_CASE("multimodal 降级 helper：连续图片合并 + 占位符文本不重复")
{
    std::vector<Message> messages;
    messages.push_back(Message{ Role::User,
                                { Text{ "a" }, Image{ kPngBase64, "image/png" },
                                  Image{ kPngBase64, "image/png" }, Text{ "b" } } });
    downgrade_unsupported_images(messages, false);
    REQUIRE(messages[0].content.size() == 3);
    CHECK(std::get<Text>(messages[0].content[0]).text == "a");
    CHECK(std::get<Text>(messages[0].content[1]).text == kNonVisionUserImagePlaceholder);
    CHECK(std::get<Text>(messages[0].content[2]).text == "b");
}

TEST_CASE("multimodal 降级 helper：支持图片时原样保留")
{
    std::vector<Message> messages;
    messages.push_back(Message{ Role::User, { Image{ kPngBase64, "image/png" } } });
    downgrade_unsupported_images(messages, true);
    REQUIRE(messages[0].content.size() == 1);
    CHECK(std::get_if<Image>(&messages[0].content[0]) != nullptr);
}

TEST_CASE("multimodal 降级 helper：assistant 消息不受影响")
{
    std::vector<Message> messages;
    messages.push_back(Message{ Role::Assistant, { Text{ "ok" } } });
    messages.push_back(Message{ Role::User, { Image{ kPngBase64, "image/png" } } });
    downgrade_unsupported_images(messages, false);
    CHECK(std::get_if<Text>(&messages[0].content[0]) != nullptr);
    CHECK(std::get_if<Text>(&messages[1].content[0]) != nullptr);
    CHECK(std::get<Text>(messages[1].content[0]).text == kNonVisionUserImagePlaceholder);
}

// ───────────────────── T1: 图片请求 → 视觉响应回放 ─────────────────────
// fixture 为协议构造（与真实视觉响应同构的文本流，图片输入请求的响应）。

TEST_CASE("multimodal T1：OpenAI 图片请求 → 视觉响应回放")
{
    ensure_vision_model();
    auto model = ModelRegistry::find_model("gpt-4o-vision");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("openai", "openai_vision_response.sse")) });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, image_ctx("图里是什么"));
    CHECK(joined_text(events) == "The image shows the number 5.");
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    CHECK(done->usage.input_tokens == 85);
    CHECK(server.errors().empty());
}

TEST_CASE("multimodal T1：Anthropic 图片请求 → 视觉响应回放")
{
    ensure_vision_model();
    auto model = ModelRegistry::find_model("gpt-4o-vision");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("anthropic", "anthropic_vision_response.sse")) });

    AnthropicMessagesProvider provider({ .name = "anthropic", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, image_ctx("图里是什么"));
    CHECK(joined_text(events) == "The image shows the number 5.");
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    CHECK(done->usage.input_tokens == 88);
    CHECK(server.errors().empty());
}

TEST_CASE("multimodal T1：Gemini 图片请求 → 视觉响应回放")
{
    ensure_vision_model();
    auto model = ModelRegistry::find_model("gpt-4o-vision");
    REQUIRE(model.has_value());
    agent::test::MockServer server;
    server.enqueue({ {}, sse_response(load_fixture("gemini", "gemma_vision_response.sse")) });

    GeminiGenerateContentProvider provider({ .name = "gemini", .api_key = "k", .base_url = server.base_url() });
    auto events = collect_stream(provider, *model, image_ctx("图里是什么"));
    CHECK(joined_text(events) == "The image shows the number 5.");
    auto done = done_response(events);
    REQUIRE(done.has_value());
    CHECK(done->stop_reason == StopReason::Stop);
    CHECK(server.errors().empty());
}

// ───────────────────── T2 契约 dump（AGENT_CONTRACT=ON 时编译）─────────────────────
// 与官方 openai SDK 含图片请求对比（run_mirror.py 的 openai_image case → golden）

#ifdef AGENT_CONTRACT_OUT

TEST_CASE("contract: dump openai 图片 build_params")
{
    ensure_vision_model();
    auto model = ModelRegistry::find_model("gpt-4o-vision");
    REQUIRE(model.has_value());
    auto params = *OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::build_params(
        *model, image_ctx("what's in this image?"), {});
    std::filesystem::path dir = AGENT_CONTRACT_OUT;
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "openai_image_1.json") << params.dump(2);
}

TEST_CASE("contract: dump anthropic 图片 build_params")
{
    ensure_vision_model();
    auto model = ModelRegistry::find_model("gpt-4o-vision");
    REQUIRE(model.has_value());
    auto params = *AnthropicMessagesEngine::build_params(
        *model, image_ctx("what's in this image?"), {});
    std::filesystem::path dir = AGENT_CONTRACT_OUT;
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "anthropic_image_1.json") << params.dump(2);
}

#endif
