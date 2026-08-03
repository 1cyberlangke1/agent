#pragma once

// T1 Agent 测试用的脚本化 mock Provider（测试代码豁免 std::function 禁令，AGENTS.md）。
//
// Agent 从 EndpointConfig 构造自己的 Provider——测试没法直接拿到 provider_，
// 所以状态经全局 registry 按 config.name 存取：测试先 reset_mock(name) 拿共享句柄，
// 设剧本、读断言；Agent 的 MockProvider 构造时按 name 自动取同一份状态。
//
// 剧本：
//   state->stream_script(ctx, call_index) → 该次请求返回的事件序列（不拉网络）
//   state->complete_script(ctx)           → 压缩摘要的 complete 响应
// 断言：
//   state->seen_contexts / seen_opts / seen_models  每次 stream_async 请求
//   state->complete_contexts / complete_opts        每次 complete（压缩摘要）请求

#include <agent/agent.hpp>

#include <asio.hpp>

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace agent::test {

struct MockState {
    using StreamScript = std::function<std::vector<agent::StreamEvent>(agent::Context const&, int)>;
    using CompleteScript = std::function<agent::Result<agent::ChatResponse>(agent::Context const&)>;

    StreamScript stream_script;                        ///< stream 剧本（空 = 无事件直接关流）
    CompleteScript complete_script;                    ///< 压缩摘要剧本（空 = 空响应）
    int stream_delay_ms = 0;                           ///< 每次 stream_async 先等这么久（abort 测试用）
    std::vector<agent::Context> seen_contexts;         ///< stream_async 收到的请求
    std::vector<agent::StreamOptions> seen_opts;
    std::vector<std::string> seen_models;
    std::vector<agent::Context> complete_contexts;     ///< complete（压缩摘要）收到的请求
    std::vector<agent::StreamOptions> complete_opts;
    int call_count = 0;
    std::string script_error;                          ///< 剧本抛异常时的消息（测试应断言为空）
};

inline std::unordered_map<std::string, std::shared_ptr<MockState>>& mock_states()
{
    static std::unordered_map<std::string, std::shared_ptr<MockState>> states;
    return states;
}

/// 重置指定名字的 mock 状态并返回共享句柄（每个 TEST_CASE 各用唯一名字）。
inline std::shared_ptr<MockState> reset_mock(std::string name)
{
    std::shared_ptr<MockState> state = std::make_shared<MockState>();
    mock_states()[std::move(name)] = state;
    return state;
}

struct MockProvider {
    std::shared_ptr<MockState> state;

    explicit MockProvider(agent::EndpointConfig const& config)
        : state(mock_states()[config.name])
    {
    }

    asio::awaitable<void> stream_async(agent::ModelView const& model, agent::Context const& ctx,
                                       agent::StreamOptions const& opts,
                                       agent::AsyncStream<agent::StreamEvent> sink)
    {
        state->seen_contexts.push_back(ctx);
        state->seen_opts.push_back(opts);
        state->seen_models.push_back(std::string(model.id));
        if (state->stream_delay_ms > 0) {
            // 先阻塞（取消路径测试）：as_tuple 保证取消不抛，仅返回错误码
            auto ex = co_await asio::this_coro::executor;
            asio::steady_timer timer(ex, std::chrono::milliseconds(state->stream_delay_ms));
            co_await timer.async_wait(asio::as_tuple(asio::use_awaitable));
        }
        int index = state->call_count++;
        std::vector<agent::StreamEvent> events;
        bool script_failed = false;
        if (state->stream_script) {
            // 测试允许 try/catch：剧本抛异常 → 记录 + 推 Error 事件，测试因事件不匹配直接失败。
            // ⚠️ co_await 不能写在 catch handler 里（GCC 协程限制），发送移到 catch 外。
            try {
                events = state->stream_script(ctx, index);
            } catch (std::exception const& e) {
                state->script_error = e.what();
                script_failed = true;
            } catch (...) {
                state->script_error = "unknown";
                script_failed = true;
            }
        }
        if (script_failed) {
            co_await sink.send(agent::StreamEvent{ agent::Error{ agent::Errc::ExecutionFailed,
                "mock script threw: " + state->script_error } });
            sink.close();
            co_return;
        }
        for (agent::StreamEvent& e : events) {
            if (!(co_await sink.send(std::move(e))))
                co_return;   // 消费端提前关闭（取消/break）
        }
        sink.close();
    }

    agent::Result<agent::ChatResponse> complete(agent::ModelView const& model, agent::Context const& ctx,
                                                agent::StreamOptions const& opts) const
    {
        (void)model;
        state->complete_contexts.push_back(ctx);
        state->complete_opts.push_back(opts);
        if (state->complete_script) {
            // 测试允许 try/catch：摘要剧本抛异常 → 返回 CompactionFailed，测试断言错误即失败
            try {
                return state->complete_script(ctx);
            } catch (std::exception const& e) {
                state->script_error = e.what();
                return std::unexpected(agent::Error{ agent::Errc::CompactionFailed, "mock complete threw: " + state->script_error });
            } catch (...) {
                state->script_error = "unknown";
                return std::unexpected(agent::Error{ agent::Errc::CompactionFailed, "mock complete threw: unknown" });
            }
        }
        return agent::ChatResponse{};
    }
};

// ───────────────────── 事件构造小工具 ─────────────────────

inline agent::StreamEvent text_delta(std::string text)
{
    return agent::StreamEvent{ agent::TextDelta{ std::move(text) } };
}
inline agent::StreamEvent thinking_delta(std::string text)
{
    return agent::StreamEvent{ agent::ThinkingDelta{ std::move(text) } };
}
inline agent::StreamEvent tool_call_delta(std::string id, std::string name, std::string args)
{
    return agent::StreamEvent{ agent::ToolCallDelta{ std::move(id), std::move(name), std::move(args) } };
}
inline agent::StreamEvent tool_call_end(std::string id, std::string name, nlohmann::json args)
{
    return agent::StreamEvent{ agent::ToolCallEnd{ std::move(id), std::move(name), std::move(args) } };
}
inline agent::StreamEvent done_text(std::string text, agent::Usage usage = {})
{
    agent::ChatResponse response;
    response.content.push_back(agent::Text{ std::move(text) });
    response.stop_reason = agent::StopReason::Stop;
    response.usage = usage;
    return agent::StreamEvent{ agent::DoneEvent{ std::move(response) } };
}
inline agent::StreamEvent done_with_tool(std::string id, std::string name, nlohmann::json args,
                                         agent::Usage usage = {})
{
    agent::ChatResponse response;
    response.content.push_back(agent::ToolCall{ std::move(id), std::move(name), std::move(args) });
    response.stop_reason = agent::StopReason::ToolUse;
    response.usage = usage;
    return agent::StreamEvent{ agent::DoneEvent{ std::move(response) } };
}

/// 测试模型（context_window / 单价可控，成本断言用）。
inline agent::ModelView test_model(int context_window = 100000, double price_input = 1.0,
                                   double price_output = 2.0)
{
    static const std::string id = "agent-test-model";
    agent::ModelView model;
    model.id = id;
    model.context_window = context_window;
    model.max_output_tokens = 4096;
    model.reasoning = true;
    model.price_input = price_input;
    model.price_output = price_output;
    return model;
}

// ───────────────────── 常用测试工具 ─────────────────────

/// 注册 agent_weather 测试工具（全局 Tools 注册表共享，只注册一次）。
inline void ensure_weather_tool()
{
    static bool registered = false;
    if (registered)
        return;
    registered = true;
    (void)agent::Tools::reg(
        agent::ToolInfo{ "agent_weather", "查询天气",
                         {{"type", "object"},
                          {"properties", {{"location", {{"type", "string"}}}}},
                          {"required", {"location"}}} },
        [](nlohmann::json args) -> agent::Result<std::string> {
            std::string location = args.value("location", "?");
            return location + " 30°C 晴";
        });
}

inline agent::Message user(std::string text)
{
    return agent::Message{ agent::Role::User, { agent::Text{ std::move(text) } } };
}
inline agent::Message assistant(std::string text)
{
    return agent::Message{ agent::Role::Assistant, { agent::Text{ std::move(text) } } };
}
inline std::string message_text(agent::Message const& message)
{
    for (auto const& block : message.content)
        if (auto text = std::get_if<agent::Text>(&block))
            return text->text;
    return {};
}

/// 事件流中是否存在某类事件。
inline int count_type(std::vector<agent::AgentEvent> const& events, agent::AgentEvent::Type type)
{
    int count = 0;
    for (agent::AgentEvent const& e : events)
        if (e.type() == type)
            ++count;
    return count;
}

/// 事件流中第一个 T 载荷；无则 nullptr。
template<typename T>
T const* first_of(std::vector<agent::AgentEvent> const& events)
{
    for (auto const& e : events)
        if (auto* p = std::get_if<T>(&e.data))
            return p;
    return nullptr;
}

/// 消息列表里是否存在某个 Role。
inline bool has_role(std::vector<agent::Message> const& messages, agent::Role role)
{
    for (agent::Message const& m : messages)
        if (m.role == role)
            return true;
    return false;
}

/// 工具往返剧本：第一次请求回工具调用，后续（工具结果在 ctx）回最终答案。
inline std::vector<agent::StreamEvent> weather_tool_roundtrip(agent::Context const& ctx, int index)
{
    if (index == 0) {
        return std::vector<agent::StreamEvent>{
            tool_call_delta("call_1", "agent_weather", "{\"loc"),
            tool_call_delta("", "", "ation\":\"杭州\"}"),
            tool_call_end("call_1", "agent_weather", {{"location", "杭州"}}),
            done_with_tool("call_1", "agent_weather", {{"location", "杭州"}}, agent::Usage{ 10, 5, 0, 0, 15 }),
        };
    }
    (void)ctx;
    return std::vector<agent::StreamEvent>{
        text_delta("杭州 30°C 晴"),
        done_text("杭州 30°C 晴", agent::Usage{ 15, 8, 0, 0, 23 }),
    };
}

}  // namespace agent::test
