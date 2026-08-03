// Agent 层容错 + 功能补全测试（T1）：钩子错误路径 / 工具失败 / 流中 Error /
// 异步接口 / 剩余扩展点 / 工具模式 / 队列模式。
//
// 覆盖审计补全（PLAN_3.md 五.5.3 之外的缺口）：
//   容错：钩子返回 Error → run 终止、工具 exec 失败、mock 流中 Error 事件、
//         complete 网络 Error → CompactionFailed、generator break 提前取消可复用
//   功能：run_async / continue_run / continue_async / compact_async / wait_for_idle、
//         system_prompt / set_model / set_reasoning、after_tool_call / run_start / before_request、
//         set_steering_mode / set_follow_up_mode(All)、set_tool_execution_mode(Parallel/Sequential)、
//         多工具一轮、context_usage、pending_tools

#include <agent/agent/agent.hpp>
#include <agent/agent/compaction.hpp>
#include <agent/tools/tools.hpp>

#include "mock_provider.hpp"

#include <doctest/doctest.h>

#include <chrono>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using namespace agent;
using namespace agent::test;

namespace {

template<typename AgentT>
std::vector<AgentEvent> run_all(AgentT& agent, std::vector<Message> msgs = {},
                                StreamOptions const& opts = {})
{
    std::vector<AgentEvent> out;
    for (AgentEvent const& ev : agent.run(std::move(msgs), opts))
        out.push_back(ev);
    return out;
}

template<typename AgentT>
std::vector<AgentEvent> run_continue_all(AgentT& agent, StreamOptions const& opts = {})
{
    std::vector<AgentEvent> out;
    for (AgentEvent const& ev : agent.continue_run(opts))
        out.push_back(ev);
    return out;
}

/// 从 run_async/continue_async 的 sink 里排空事件。
std::vector<AgentEvent> drain(AsyncStream<AgentEvent>& sink)
{
    std::vector<AgentEvent> out;
    while (auto item = sink.try_receive()) {
        if (*item)
            out.push_back(std::move(**item));
        else
            break;   // channel closed
    }
    return out;
}

/// 无思考、快速回文字的标准剧本。
std::vector<StreamEvent> plain_text_script(Context const&, int)
{
    return std::vector<StreamEvent>{ text_delta("ok"), done_text("ok") };
}

/// 注册一个执行必失败的工具（容错测试用，唯一名避开其他测试）。
void ensure_boom_tool()
{
    static bool registered = false;
    if (registered)
        return;
    registered = true;
    (void)Tools::reg(
        ToolInfo{ "boom_tool", "必失败工具",
                  {{"type", "object"}, {"properties", nlohmann::json::object()}} },
        [](nlohmann::json) -> Result<std::string> {
            return std::unexpected(Error{ Errc::ExecutionFailed, "工具内部爆炸" });
        });
}

// ── 容错 Behaviors：各钩子返回 Error ──

class TcErrorBehaviors : public DefaultBehaviors {
public:
    template<typename Provider>
    Result<std::vector<Message>> transform_context(Provider const&, ContextSnapshot const&) const
    {
        return std::unexpected(Error{ Errc::ExecutionFailed, "tc boom" });
    }
};
class BrErrorBehaviors : public DefaultBehaviors {
public:
    template<typename Provider>
    Result<std::optional<StreamOptions>> before_request(Provider const&, StreamOptions const&) const
    {
        return std::unexpected(Error{ Errc::ExecutionFailed, "br boom" });
    }
};
class KeyErrorBehaviors : public DefaultBehaviors {
public:
    Result<std::optional<std::string>> get_api_key(std::string_view) const
    {
        return std::unexpected(Error{ Errc::ExecutionFailed, "key boom" });
    }
};
class StopErrorBehaviors : public DefaultBehaviors {
public:
    template<typename Provider>
    Result<bool> should_stop(Provider const&, Message const&) const
    {
        return std::unexpected(Error{ Errc::ExecutionFailed, "stop boom" });
    }
};
class PntErrorBehaviors : public DefaultBehaviors {
public:
    template<typename Provider>
    Result<std::optional<NextTurnUpdate>> prepare_next_turn(
        Provider const&, Message const&, std::vector<ToolResult> const&) const
    {
        return std::unexpected(Error{ Errc::ExecutionFailed, "pnt boom" });
    }
};
class BeforeToolErrorBehaviors : public DefaultBehaviors {
public:
    template<typename Provider>
    Result<bool> before_tool_call(Provider const&, ToolCall const&, ContextSnapshot const&) const
    {
        return std::unexpected(Error{ Errc::ExecutionFailed, "before boom" });
    }
};

// ── 功能 Behaviors ──

class RewriteAfterBehaviors : public DefaultBehaviors {
public:
    template<typename Provider>
    Result<ToolResult> after_tool_call(Provider const&, ToolCall const&, ToolResult result) const
    {
        result.output = "改写后的结果";
        return result;
    }
};
class PatchStartBehaviors : public DefaultBehaviors {
public:
    template<typename Provider>
    Result<std::optional<RunStartPatch>> run_start(
        Provider const&, std::vector<Message> const&, std::string const&) const
    {
        RunStartPatch patch;
        patch.messages = { user("补丁消息") };
        patch.system_prompt = "补丁系统提示";
        return patch;
    }
};
class MaxTokensBehaviors : public DefaultBehaviors {
public:
    template<typename Provider>
    Result<std::optional<StreamOptions>> before_request(Provider const&, StreamOptions const& options) const
    {
        StreamOptions modified = options;
        modified.max_tokens = 123;
        return modified;
    }
};

class PayloadRewriteBehaviors : public DefaultBehaviors {
public:
    template<typename Provider>
    Result<std::optional<nlohmann::json>> before_payload(Provider const&, nlohmann::json const& body) const
    {
        nlohmann::json modified = body;   // 改写：在构建结果上加字段，其余保留
        modified["temperature"] = 0.7;
        modified["extra_flag"] = "改写的";
        return modified;
    }
};

class PayloadErrorBehaviors : public DefaultBehaviors {
public:
    template<typename Provider>
    Result<std::optional<nlohmann::json>> before_payload(Provider const&, nlohmann::json const&) const
    {
        return std::unexpected(Error{ Errc::ExecutionFailed, "payload boom" });
    }
};

/// 单轮两个工具调用 + 完整 Done（含两个 ToolCall 块）。
std::vector<StreamEvent> two_tools_script(Context const&, int index)
{
    if (index != 0)
        return std::vector<StreamEvent>{ text_delta("搞定"), done_text("搞定") };
    ChatResponse response;
    response.content.push_back(ToolCall{ "c1", "agent_weather", {{"location", "杭州"}} });
    response.content.push_back(ToolCall{ "c2", "agent_weather", {{"location", "上海"}} });
    response.stop_reason = StopReason::ToolUse;
    return std::vector<StreamEvent>{
        tool_call_end("c1", "agent_weather", {{"location", "杭州"}}),
        tool_call_end("c2", "agent_weather", {{"location", "上海"}}),
        StreamEvent{ DoneEvent{ std::move(response) } },
    };
}

}  // namespace

// ───────────────────── 容错：钩子返回 Error → run 以该错误终止 ─────────────────────

TEST_CASE("容错：transform_context 返回 Error → run 终止")
{
    auto st = reset_mock("rt-tc");
    st->stream_script = plain_text_script;
    Agent<MockProvider, TcErrorBehaviors> agent(EndpointConfig{ .name = "rt-tc" }, test_model(),
                                                TcErrorBehaviors{});
    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    REQUIRE(agent.last_error().has_value());
    CHECK(agent.last_error()->code == Errc::ExecutionFailed);
    CHECK(agent.last_error()->message == "tc boom");
    CHECK(count_type(events, AgentEvent::Type::AgentError) == 1);
    CHECK(events.back().type() == AgentEvent::Type::AgentEnd);
}

TEST_CASE("容错：before_request 返回 Error → run 终止")
{
    auto st = reset_mock("rt-br");
    st->stream_script = plain_text_script;
    Agent<MockProvider, BrErrorBehaviors> agent(EndpointConfig{ .name = "rt-br" }, test_model(),
                                                BrErrorBehaviors{});
    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    REQUIRE(agent.last_error().has_value());
    CHECK(agent.last_error()->code == Errc::ExecutionFailed);
    CHECK(agent.last_error()->message == "br boom");
    CHECK(events.back().type() == AgentEvent::Type::AgentEnd);
}

TEST_CASE("容错：get_api_key 返回 Error → run 终止")
{
    auto st = reset_mock("rt-key");
    st->stream_script = plain_text_script;
    Agent<MockProvider, KeyErrorBehaviors> agent(EndpointConfig{ .name = "rt-key" }, test_model(),
                                                 KeyErrorBehaviors{});
    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    REQUIRE(agent.last_error().has_value());
    CHECK(agent.last_error()->code == Errc::ExecutionFailed);
    CHECK(agent.last_error()->message == "key boom");
    CHECK(events.back().type() == AgentEvent::Type::AgentEnd);
}

TEST_CASE("容错：should_stop 返回 Error → run 终止")
{
    auto st = reset_mock("rt-stop");
    st->stream_script = plain_text_script;
    Agent<MockProvider, StopErrorBehaviors> agent(EndpointConfig{ .name = "rt-stop" }, test_model(),
                                                  StopErrorBehaviors{});
    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    REQUIRE(agent.last_error().has_value());
    CHECK(agent.last_error()->code == Errc::ExecutionFailed);
    CHECK(agent.last_error()->message == "stop boom");
    CHECK(events.back().type() == AgentEvent::Type::AgentEnd);
}

TEST_CASE("容错：prepare_next_turn 返回 Error → run 终止")
{
    auto st = reset_mock("rt-pnt");
    st->stream_script = weather_tool_roundtrip;
    Agent<MockProvider, PntErrorBehaviors> agent(EndpointConfig{ .name = "rt-pnt" }, test_model(),
                                                 PntErrorBehaviors{});
    agent.set_tools({ "agent_weather" });
    std::vector<AgentEvent> events = run_all(agent, { user("查天气") });

    REQUIRE(agent.last_error().has_value());
    CHECK(agent.last_error()->code == Errc::ExecutionFailed);
    CHECK(agent.last_error()->message == "pnt boom");
    CHECK(events.back().type() == AgentEvent::Type::AgentEnd);
}

TEST_CASE("容错：before_tool_call 返回 Error → 工具结果 is_error")
{
    ensure_weather_tool();
    auto st = reset_mock("rt-btool");
    st->stream_script = weather_tool_roundtrip;
    Agent<MockProvider, BeforeToolErrorBehaviors> agent(EndpointConfig{ .name = "rt-btool" },
                                                        test_model(), BeforeToolErrorBehaviors{});
    agent.set_tools({ "agent_weather" });
    std::vector<AgentEvent> events = run_all(agent, { user("查天气") });

    ToolExecEnd const* exec_end = first_of<ToolExecEnd>(events);
    REQUIRE(exec_end != nullptr);
    CHECK(exec_end->result.is_error);
    CHECK(exec_end->result.output.find("before boom") != std::string::npos);
    // 模型收到被拒/错误原因
    REQUIRE(st->seen_contexts.size() == 2);
    CHECK(has_tool_result_with(st->seen_contexts[1].messages, "before boom"));
}

TEST_CASE("容错：工具 exec 失败 → ToolResult 错误回传")
{
    ensure_weather_tool();
    ensure_boom_tool();
    auto st = reset_mock("rt-boom");
    st->stream_script = [](Context const&, int index) {
        if (index != 0)
            return std::vector<StreamEvent>{ text_delta("收到错误"), done_text("收到错误") };
        ChatResponse response;
        response.content.push_back(ToolCall{ "c1", "boom_tool", nlohmann::json::object() });
        response.stop_reason = StopReason::ToolUse;
        return std::vector<StreamEvent>{
            tool_call_end("c1", "boom_tool", nlohmann::json::object()),
            StreamEvent{ DoneEvent{ std::move(response) } },
        };
    };
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-boom" }, test_model());
    agent.set_tools({ "boom_tool" });
    std::vector<AgentEvent> events = run_all(agent, { user("触发爆炸") });

    ToolExecEnd const* exec_end = first_of<ToolExecEnd>(events);
    REQUIRE(exec_end != nullptr);
    CHECK(exec_end->name == "boom_tool");
    CHECK(exec_end->result.is_error);
    CHECK(exec_end->result.output.find("工具内部爆炸") != std::string::npos);
    // 模型第二次请求收到错误结果
    REQUIRE(st->seen_contexts.size() == 2);
    CHECK(has_tool_result_with(st->seen_contexts[1].messages, "工具内部爆炸"));
}

TEST_CASE("容错：mock 流中 Error 事件 → AgentError")
{
    auto st = reset_mock("rt-err");
    st->stream_script = [](Context const&, int) {
        return std::vector<StreamEvent>{ StreamEvent{ Error{ Errc::RateLimited, "上游限流" } } };
    };
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-err" }, test_model());
    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    REQUIRE(agent.last_error().has_value());
    CHECK(agent.last_error()->code == Errc::RateLimited);
    CHECK(agent.last_error()->message == "上游限流");
    CHECK(count_type(events, AgentEvent::Type::AgentError) == 1);
    CHECK(events.back().type() == AgentEvent::Type::AgentEnd);
}

TEST_CASE("容错：complete 网络 Error → CompactionFailed")
{
    auto st = reset_mock("rt-cnet");
    DefaultBehaviors behaviors;
    behaviors.settings_ = CompactionSettings{ 50, 100, 2 };
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-cnet" }, test_model(200),
                              std::move(behaviors));
    st->stream_script = [](Context const&, int) {
        return std::vector<StreamEvent>{ text_delta("ok"), done_text("ok", Usage{ 500, 4, 0, 0, 504 }) };
    };
    st->complete_script = [](Context const&) -> Result<ChatResponse> {
        return std::unexpected(Error{ Errc::NetworkError, "摘要请求超时" });
    };

    for (int i = 0; i < 3; ++i)
        run_all(agent, { user(std::string(200, 'a')) });
    // run 内防爆触发 → complete 失败 → CompactionFailed（message 含原因）
    REQUIRE(agent.last_error().has_value());
    CHECK(agent.last_error()->code == Errc::CompactionFailed);
    CHECK(agent.last_error()->message.find("压缩摘要请求失败") != std::string::npos);
    CHECK(agent.last_error()->message.find("摘要请求超时") != std::string::npos);
}

TEST_CASE("容错：generator break 提前取消 → 不挂 + agent 可复用")
{
    auto st = reset_mock("rt-break");
    st->stream_script = plain_text_script;
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-break" }, test_model());

    int seen = 0;
    for (AgentEvent const& ev : agent.run({ user("hi") })) {
        if (seen++ >= 2)
            break;   // 中途 break（generator 析构 → 取消）
        (void)ev;
    }
    // 不挂；同一 agent 可再次 run
    std::vector<AgentEvent> events = run_all(agent, { user("再来") });
    CHECK(count_type(events, AgentEvent::Type::AgentEnd) == 1);
    CHECK(st->seen_contexts.size() == 2);
    CHECK(!agent.is_streaming());
}

// ───────────────────── 功能补全：剩余扩展点 ─────────────────────

TEST_CASE("功能：after_tool_call 改写结果")
{
    ensure_weather_tool();
    auto st = reset_mock("rt-after");
    st->stream_script = weather_tool_roundtrip;
    Agent<MockProvider, RewriteAfterBehaviors> agent(EndpointConfig{ .name = "rt-after" }, test_model(),
                                                     RewriteAfterBehaviors{});
    agent.set_tools({ "agent_weather" });
    std::vector<AgentEvent> events = run_all(agent, { user("查天气") });

    ToolExecEnd const* exec_end = first_of<ToolExecEnd>(events);
    REQUIRE(exec_end != nullptr);
    CHECK(exec_end->result.output == "改写后的结果");
    CHECK(!exec_end->result.is_error);
}

TEST_CASE("功能：run_start 补丁（改消息 + 系统提示）")
{
    auto st = reset_mock("rt-patch");
    st->stream_script = plain_text_script;
    Agent<MockProvider, PatchStartBehaviors> agent(EndpointConfig{ .name = "rt-patch" }, test_model(),
                                                   PatchStartBehaviors{});
    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    REQUIRE(!st->seen_contexts.empty());
    // 请求 ctx 用补丁后的消息 + 系统提示
    CHECK(st->seen_contexts[0].system_prompt == "补丁系统提示");
    REQUIRE(!st->seen_contexts[0].messages.empty());
    CHECK(message_text(st->seen_contexts[0].messages[0]) == "补丁消息");
    CHECK(agent.system_prompt() == "补丁系统提示");
}

TEST_CASE("功能：before_request 改写 StreamOptions")
{
    auto st = reset_mock("rt-mt");
    st->stream_script = plain_text_script;
    Agent<MockProvider, MaxTokensBehaviors> agent(EndpointConfig{ .name = "rt-mt" }, test_model(),
                                                  MaxTokensBehaviors{});
    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    REQUIRE(!st->seen_opts.empty());
    REQUIRE(st->seen_opts[0].max_tokens.has_value());
    CHECK(*st->seen_opts[0].max_tokens == 123);
}

TEST_CASE("功能：system_prompt 构造传递")
{
    auto st = reset_mock("rt-sys");
    st->stream_script = plain_text_script;
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-sys" }, test_model(),
                              DefaultBehaviors{}, "你是个助手");
    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    REQUIRE(!st->seen_contexts.empty());
    CHECK(st->seen_contexts[0].system_prompt == "你是个助手");
    CHECK(agent.system_prompt() == "你是个助手");
}

TEST_CASE("功能：set_model 生效")
{
    auto st = reset_mock("rt-model");
    st->stream_script = plain_text_script;
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-model" }, test_model());
    static const std::string next_id = "switch-model";
    ModelView next;
    next.id = next_id;
    next.context_window = 8000;
    agent.set_model(next);

    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    REQUIRE(!st->seen_models.empty());
    CHECK(st->seen_models[0] == "switch-model");
    CHECK(agent.model().id == "switch-model");
}

TEST_CASE("功能：set_reasoning 下一轮生效")
{
    auto st = reset_mock("rt-reason");
    st->stream_script = plain_text_script;
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-reason" }, test_model());
    agent.set_reasoning(ThinkingLevel::High);

    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    REQUIRE(!st->seen_opts.empty());
    REQUIRE(st->seen_opts[0].reasoning.has_value());
    CHECK(*st->seen_opts[0].reasoning == ThinkingLevel::High);
}

TEST_CASE("功能：多工具一轮 Parallel（std::async 真并行）")
{
    ensure_weather_tool();
    auto st = reset_mock("rt-2tool");
    st->stream_script = two_tools_script;
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-2tool" }, test_model());
    agent.set_tools({ "agent_weather" });

    std::vector<AgentEvent> events = run_all(agent, { user("查两个城市") });

    CHECK(count_type(events, AgentEvent::Type::ToolExecStart) == 2);
    CHECK(count_type(events, AgentEvent::Type::ToolExecEnd) == 2);
    CHECK(count_type(events, AgentEvent::Type::TurnStart) == 2);   // 工具轮 + 答案轮
    // 两个结果都回传
    REQUIRE(st->seen_contexts.size() == 2);
    CHECK(has_tool_result_with(st->seen_contexts[1].messages, "杭州 30°C 晴"));
    CHECK(has_tool_result_with(st->seen_contexts[1].messages, "上海 30°C 晴"));
    // pending_tools 执行后清空
    CHECK(agent.pending_tools().empty());
}

TEST_CASE("功能：工具 Sequential 模式")
{
    ensure_weather_tool();
    auto st = reset_mock("rt-seq");
    st->stream_script = two_tools_script;
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-seq" }, test_model());
    agent.set_tools({ "agent_weather" });
    agent.set_tool_execution_mode(ToolExecutionMode::Sequential);

    std::vector<AgentEvent> events = run_all(agent, { user("查两个城市") });

    CHECK(count_type(events, AgentEvent::Type::ToolExecStart) == 2);
    CHECK(count_type(events, AgentEvent::Type::ToolExecEnd) == 2);
    CHECK(agent.pending_tools().empty());
}

TEST_CASE("功能：context_usage 用量快照")
{
    auto st = reset_mock("rt-usage");
    st->stream_script = [](Context const&, int) {
        return std::vector<StreamEvent>{ text_delta("hi"), done_text("hi", Usage{ 10, 20, 0, 0, 30 }) };
    };
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-usage" }, test_model(100000, 1.0, 2.0));
    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    Usage usage = agent.context_usage();
    CHECK(usage.total_tokens >= 30);
    CHECK(usage.input_tokens > 0);
    CHECK(agent.context_tokens() >= 30);
}

// ───────────────────── 功能补全：异步接口 / 队列模式 ─────────────────────

TEST_CASE("功能：run_async 异步跑完并收到事件")
{
    auto st = reset_mock("rt-runasync");
    st->stream_script = plain_text_script;
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-runasync" }, test_model());

    asio::io_context io;
    AsyncStream<AgentEvent> sink(io.get_executor());
    bool completed = false;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        co_await agent.run_async({ user("你好") }, sink);
        completed = true;
    }, asio::detached);
    io.run();

    CHECK(completed);
    std::vector<AgentEvent> events = drain(sink);
    CHECK(count_type(events, AgentEvent::Type::AgentStart) == 1);
    CHECK(count_type(events, AgentEvent::Type::AgentEnd) == 1);
    CHECK(agent.messages().size() == 2);
    CHECK(!agent.is_streaming());
}

TEST_CASE("功能：continue_run 续跑")
{
    auto st = reset_mock("rt-cont");
    st->stream_script = plain_text_script;
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-cont" }, test_model());
    run_all(agent, { user("第一问") });

    std::vector<AgentEvent> events = run_continue_all(agent);
    CHECK(count_type(events, AgentEvent::Type::AgentEnd) == 1);
    CHECK(st->seen_contexts.size() == 2);
    // 首轮 2 条 + 续跑只追加 1 条 assistant（continue 不注入 user）
    CHECK(agent.messages().size() == 3);
}

TEST_CASE("功能：continue_async 续跑")
{
    auto st = reset_mock("rt-contas");
    st->stream_script = plain_text_script;
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-contas" }, test_model());
    run_all(agent, { user("第一问") });

    asio::io_context io;
    AsyncStream<AgentEvent> sink(io.get_executor());
    bool completed = false;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        co_await agent.continue_async(sink);
        completed = true;
    }, asio::detached);
    io.run();

    CHECK(completed);
    std::vector<AgentEvent> events = drain(sink);
    CHECK(count_type(events, AgentEvent::Type::AgentEnd) == 1);
    CHECK(st->seen_contexts.size() == 2);
}

TEST_CASE("功能：compact_async 异步压缩")
{
    auto st = reset_mock("rt-cas");
    DefaultBehaviors behaviors;
    behaviors.settings_ = CompactionSettings{ 50, 100, 2 };
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-cas" }, test_model(200),
                              std::move(behaviors));
    st->stream_script = [](Context const&, int) {
        return std::vector<StreamEvent>{ text_delta("ok"), done_text("ok", Usage{ 500, 4, 0, 0, 504 }) };
    };
    st->complete_script = [](Context const&) -> Result<ChatResponse> {
        ChatResponse response;
        response.content.push_back(Text{ "摘要：用户问了天气。" });
        response.stop_reason = StopReason::Stop;
        return response;
    };

    for (int i = 0; i < 3; ++i)
        run_all(agent, { user(std::string(200, 'a')) });

    asio::io_context io;
    std::optional<Result<bool>> outcome;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        outcome = co_await agent.compact_async();
    }, asio::detached);
    io.run();

    REQUIRE(outcome.has_value());
    REQUIRE(*outcome);
    REQUIRE(!agent.messages().empty());
    CHECK(is_compaction_summary(agent.messages().front()));
    CHECK(agent.previous_summary() == "摘要：用户问了天气。");
}

TEST_CASE("功能：wait_for_idle 等 run 结束")
{
    auto st = reset_mock("rt-wfi");
    st->stream_script = weather_tool_roundtrip;
    st->stream_delay_ms = 500;   // 让 run 持续一段时间
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-wfi" }, test_model());
    agent.set_tools({ "agent_weather" });

    std::thread runner([&] {
        for (AgentEvent const& ev : agent.run({ user("查天气") })) {
            (void)ev;
        }
    });

    asio::io_context io;
    bool idle_done = false;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        co_await agent.wait_for_idle();
        idle_done = true;
    }, asio::detached);
    io.run();

    runner.join();
    CHECK(idle_done);
    CHECK(!agent.is_streaming());
}

TEST_CASE("功能：steering 模式 All 一次全给")
{
    auto st = reset_mock("rt-steerall");
    st->stream_script = plain_text_script;
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-steerall" }, test_model());
    agent.set_steering_mode(QueueMode::All);
    agent.steer(user("插话1"));
    agent.steer(user("插话2"));

    std::vector<AgentEvent> events = run_all(agent, { user("你好") });

    REQUIRE(!st->seen_contexts.empty());
    REQUIRE(st->seen_contexts[0].messages.size() >= 3);
    CHECK(message_text(st->seen_contexts[0].messages[1]) == "插话1");
    CHECK(message_text(st->seen_contexts[0].messages[2]) == "插话2");
}

TEST_CASE("功能：follow_up 模式 All 一次全给")
{
    auto st = reset_mock("rt-fuall");
    st->stream_script = plain_text_script;
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-fuall" }, test_model());
    agent.set_follow_up_mode(QueueMode::All);
    agent.follow_up(user("追加1"));
    agent.follow_up(user("追加2"));

    std::vector<AgentEvent> events = run_all(agent, { user("第一问") });

    CHECK(count_type(events, AgentEvent::Type::TurnStart) == 2);
    REQUIRE(st->seen_contexts.size() == 2);
    // 首轮回复后注入：第二请求 ctx = [第一问, ok, 追加1, 追加2]
    REQUIRE(st->seen_contexts[1].messages.size() >= 4);
    CHECK(message_text(st->seen_contexts[1].messages[2]) == "追加1");
    CHECK(message_text(st->seen_contexts[1].messages[3]) == "追加2");
}

TEST_CASE("功能：clear_queues 清空后不注入")
{
    auto st = reset_mock("rt-clearq");
    st->stream_script = plain_text_script;
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-clearq" }, test_model());
    agent.steer(user("插话"));
    agent.follow_up(user("追加"));
    agent.clear_queues();
    CHECK(!agent.has_queued_messages());

    std::vector<AgentEvent> events = run_all(agent, { user("你好") });
    CHECK(count_type(events, AgentEvent::Type::TurnStart) == 1);   // 无插话 → 单轮
    CHECK(st->seen_contexts.size() == 1);
    CHECK(message_text(st->seen_contexts[0].messages.back()) == "你好");
}

// ───────────────────── 补测：剩余缺口 ─────────────────────

TEST_CASE("功能：工具 Default 模式（回退顺序执行）")
{
    ensure_weather_tool();
    auto st = reset_mock("rt-tooldef");
    st->stream_script = two_tools_script;
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-tooldef" }, test_model());
    agent.set_tools({ "agent_weather" });
    agent.set_tool_execution_mode(ToolExecutionMode::Default);

    std::vector<AgentEvent> events = run_all(agent, { user("查两个城市") });
    CHECK(count_type(events, AgentEvent::Type::ToolExecStart) == 2);
    CHECK(count_type(events, AgentEvent::Type::ToolExecEnd) == 2);
    CHECK(agent.pending_tools().empty());
}

TEST_CASE("功能：abort 无 run 在途 → 不影响后续 run（可复用）")
{
    auto st = reset_mock("rt-abortreuse");
    st->stream_script = plain_text_script;
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-abortreuse" }, test_model());

    agent.abort();   // 无 run 在途：best effort，不崩
    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    CHECK(count_type(events, AgentEvent::Type::AgentEnd) == 1);
    CHECK(!agent.last_error().has_value());
    CHECK(agent.messages().size() == 2);
    CHECK(!agent.is_streaming());
}

TEST_CASE("容错：run_async 中途 abort → Aborted")
{
    ensure_weather_tool();
    auto st = reset_mock("rt-asabort");
    st->stream_script = weather_tool_roundtrip;
    st->stream_delay_ms = 2000;   // 阻塞生产协程，给 abort 窗口（取消不生效最坏 2s 兜底）
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-asabort" }, test_model());
    agent.set_tools({ "agent_weather" });

    asio::io_context io;
    bool completed = false;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        AsyncStream<AgentEvent> sink(io.get_executor());
        co_await agent.run_async({ user("查天气") }, sink);
        completed = true;
    }, asio::detached);

    std::thread runner([&] { io.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    agent.abort();
    runner.join();

    CHECK(completed);
    REQUIRE(agent.last_error().has_value());
    CHECK(agent.last_error()->code == Errc::Aborted);
    CHECK(!agent.is_streaming());
}

TEST_CASE("功能：steer 运行中插队（流阻塞时注入，下一轮生效）")
{
    auto st = reset_mock("rt-steermid");
    st->stream_script = plain_text_script;
    st->stream_delay_ms = 1000;   // 首轮阻塞，给插队窗口
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-steermid" }, test_model());

    std::thread runner([&] {
        for (AgentEvent const& ev : agent.run({ user("你好") })) {
            (void)ev;
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    agent.steer(user("插话"));   // 首轮流阻塞期间插队
    runner.join();

    // 首轮完成后注入 → 第二轮请求末尾是插话
    REQUIRE(st->seen_contexts.size() >= 2);
    CHECK(message_text(st->seen_contexts[1].messages.back()) == "插话");
    CHECK(!agent.is_streaming());
}

// ───────────────────── before_payload（引擎原生支持改写）─────────────────────

TEST_CASE("功能：before_payload 改写请求体（构建结果上改，其余保留）")
{
    auto st = reset_mock("rt-bp");
    st->stream_script = plain_text_script;
    Agent<MockProvider, PayloadRewriteBehaviors> agent(EndpointConfig{ .name = "rt-bp" },
                                                       test_model(), PayloadRewriteBehaviors{});

    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    // 引擎 build_params 生成 → 钩子改写 → prebuilt_body 交还引擎发送
    REQUIRE(!st->seen_opts.empty());
    REQUIRE(st->seen_opts[0].prebuilt_body.has_value());
    nlohmann::json const& body = *st->seen_opts[0].prebuilt_body;
    CHECK(body["mock"] == true);              // 构建原样保留（改写不是覆写整份）
    CHECK(body["temperature"] == 0.7);        // 改写生效
    CHECK(body["extra_flag"] == "改写的");    // 新增字段
}

TEST_CASE("容错：before_payload 返回 Error → run 终止")
{
    auto st = reset_mock("rt-bperr");
    st->stream_script = plain_text_script;
    Agent<MockProvider, PayloadErrorBehaviors> agent(EndpointConfig{ .name = "rt-bperr" },
                                                     test_model(), PayloadErrorBehaviors{});

    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    REQUIRE(agent.last_error().has_value());
    CHECK(agent.last_error()->code == Errc::ExecutionFailed);
    CHECK(agent.last_error()->message == "payload boom");
    CHECK(count_type(events, AgentEvent::Type::AgentError) == 1);
    CHECK(events.back().type() == AgentEvent::Type::AgentEnd);
}

TEST_CASE("功能：before_payload nullopt → 不改（构建原样交还引擎）")
{
    auto st = reset_mock("rt-bpnone");
    st->stream_script = plain_text_script;
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-bpnone" }, test_model());

    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    REQUIRE(!st->seen_opts.empty());
    REQUIRE(st->seen_opts[0].prebuilt_body.has_value());
    CHECK((*st->seen_opts[0].prebuilt_body)["mock"] == true);
    CHECK(!(*st->seen_opts[0].prebuilt_body).contains("temperature"));
}


// ───────────────────── per-tool 工具执行模式 ─────────────────────

TEST_CASE("功能：Tools::mode 查询 per-tool 执行模式")
{
    (void)Tools::reg(ToolInfo{ "mode_seq", "串行工具", {{"type", "object"}} },
                     [](nlohmann::json) -> Result<std::string> { return "seq"; },
                     ArgsCheck::Schema, ToolExecutionMode::Sequential);
    (void)Tools::reg(ToolInfo{ "mode_default", "默认工具", {{"type", "object"}} },
                     [](nlohmann::json) -> Result<std::string> { return "default"; });

    Result<ToolExecutionMode> seq = Tools::mode("mode_seq");
    REQUIRE(seq.has_value());
    CHECK(*seq == ToolExecutionMode::Sequential);

    Result<ToolExecutionMode> dflt = Tools::mode("mode_default");
    REQUIRE(dflt.has_value());
    CHECK(*dflt == ToolExecutionMode::Default);

    Result<ToolExecutionMode> missing = Tools::mode("no_such_tool");
    REQUIRE_FALSE(missing.has_value());
    CHECK(missing.error().code == Errc::NotFound);
}

TEST_CASE("功能：per-tool 执行模式覆盖 Agent 全局")
{
    // para_tool 强制 Parallel；agent_weather 默认（跟随全局）
    (void)Tools::reg(ToolInfo{ "para_tool", "强制并行", {{"type", "object"}} },
                     [](nlohmann::json) -> Result<std::string> { return "para"; },
                     ArgsCheck::Schema, ToolExecutionMode::Parallel);
    ensure_weather_tool();
    auto st = reset_mock("rt-permode");
    st->stream_script = [](Context const&, int index) {
        if (index != 0)
            return std::vector<StreamEvent>{ text_delta("ok"), done_text("ok") };
        ChatResponse response;
        response.content.push_back(ToolCall{ "c1", "para_tool", nlohmann::json::object() });
        response.content.push_back(ToolCall{ "c2", "agent_weather", {{"location", "杭州"}} });
        response.stop_reason = StopReason::ToolUse;
        return std::vector<StreamEvent>{
            tool_call_end("c1", "para_tool", nlohmann::json::object()),
            tool_call_end("c2", "agent_weather", {{"location", "杭州"}}),
            StreamEvent{ DoneEvent{ std::move(response) } },
        };
    };
    Agent<MockProvider> agent(EndpointConfig{ .name = "rt-permode" }, test_model());
    agent.set_tools({ "para_tool", "agent_weather" });
    agent.set_tool_execution_mode(ToolExecutionMode::Sequential);   // 全局串行

    std::vector<AgentEvent> events = run_all(agent, { user("干活") });

    // 两个工具都执行成功：para_tool 被 per-tool Parallel 覆盖（走并发分支）、
    // agent_weather 默认跟随全局 Sequential（走内联分支）
    std::vector<ToolExecEnd const*> exec_ends;
    for (auto const& e : events)
        if (auto* te = std::get_if<ToolExecEnd>(&e.data))
            exec_ends.push_back(te);
    REQUIRE(exec_ends.size() == 2);
    for (auto const* te : exec_ends) {
        CHECK(!te->result.is_error);
        if (te->name == "para_tool")
            CHECK(te->result.output == "para");
    }
}
