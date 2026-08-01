// T2 契约层：响应侧镜像对比的 C++ 侧（我们的解析结果 dump）。
//
// 同一份 fixture 字节，官方 SDK（run_mirror.py → sdk_response/）和我们的库
// （本文件 → out/<case>_response.json）各自解析，compare_responses.py 对比两者
// 是否一致——验证「收到相同的包」：两边对同一响应的解析语义对齐。
//
// 用纯函数（SseParser + parse_chunk）直接喂 fixture，无网络、无 mock。

#include <agent/core/http_client.hpp>
#include <agent/llm/engine/openai_completions.hpp>
#include <agent/llm/providers/anthropic.hpp>
#include <agent/llm/providers/openai.hpp>

#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace agent;
using namespace agent::detail;

#ifdef AGENT_CONTRACT_OUT

namespace {

std::string read_fixture(std::string const& provider, std::string const& name)
{
    std::filesystem::path dir = AGENT_TEST_FIXTURES_DIR;
    std::ifstream in(dir / provider / name, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

/// StopReason → 统一字符串（与 SDK 侧 run_mirror.py 的映射一致）。
std::string stop_str(StopReason reason)
{
    switch (reason) {
        case StopReason::Stop: return "stop";
        case StopReason::Length: return "length";
        case StopReason::ToolUse: return "tool_use";
        case StopReason::Error: return "error";
        case StopReason::Aborted: return "aborted";
    }
    return "unknown";
}

nlohmann::json usage_json(Usage const& u)
{
    return { { "input", u.input_tokens }, { "output", u.output_tokens },
             { "cache_read", u.cache_read_tokens }, { "cache_write", u.cache_write_tokens } };
}

/// tool 参数增量累积 → 完整对象（非法 JSON → 空对象，与引擎行为一致）。
nlohmann::json parse_args(std::string const& partial)
{
    if (partial.empty())
        return nlohmann::json::object();
    nlohmann::json j = nlohmann::json::parse(partial, nullptr, false);
    return j.is_discarded() ? nlohmann::json::object() : j;
}

void write_dump(std::string const& case_name, nlohmann::json const& result)
{
    std::filesystem::path dir = AGENT_CONTRACT_OUT;
    std::filesystem::create_directories(dir);
    std::ofstream(dir / (case_name + "_response.json")) << result.dump(2);
}

}  // namespace

TEST_CASE("contract: openai 响应解析镜像 dump")
{
    for (auto const& [file, case_name] : std::vector<std::pair<std::string, std::string>>{
             { "openai_text_4.sse", "text_4" },
             { "openai_tool_1.sse", "tool_1" } }) {
        detail::SseParser parser;
        parser.feed(read_fixture("openai", file));
        OpenAIStreamState state;
        while (auto event = parser.next_event()) {
            if (*event == "[DONE]")
                break;
            nlohmann::json j = nlohmann::json::parse(*event, nullptr, false);
            if (!j.is_discarded())
                OpenAICompletionsEngine<OpenAIThinking, OpenAICompat>::parse_chunk(state, j);
        }
        nlohmann::json tools = nlohmann::json::array();
        for (auto const& [index, slot] : state.tools)
            tools.push_back({ { "id", slot.id }, { "name", slot.name },
                              { "arguments", parse_args(slot.partial_args) } });
        write_dump(case_name, { { "text", state.text }, { "thinking", state.thinking },
                                { "tools", tools }, { "stop_reason", stop_str(state.stop_reason) },
                                { "usage", usage_json(state.usage) } });
    }
}

TEST_CASE("contract: anthropic 响应解析镜像 dump")
{
    for (auto const& [file, case_name] : std::vector<std::pair<std::string, std::string>>{
             { "anthropic_text.sse", "anthropic_text" },
             { "anthropic_tool_round1.sse", "anthropic_tool_1" } }) {
        detail::SseParser parser;
        parser.feed(read_fixture("anthropic", file));
        AnthropicStreamState state;
        while (auto event = parser.next_event()) {
            nlohmann::json j = nlohmann::json::parse(*event, nullptr, false);
            if (!j.is_discarded())
                AnthropicMessagesEngine::parse_chunk(state, j);
        }
        nlohmann::json tools = nlohmann::json::array();
        for (auto const& [index, slot] : state.tools)
            tools.push_back({ { "id", slot.id }, { "name", slot.name },
                              { "arguments", parse_args(slot.partial_args) } });
        write_dump(case_name, { { "text", state.text }, { "thinking", state.thinking },
                                { "tools", tools }, { "stop_reason", stop_str(state.stop_reason) },
                                { "usage", usage_json(state.usage) } });
    }
}

#endif
