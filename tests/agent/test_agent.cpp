// Agent 循环 T1 测试：mock Provider 断言事件序列 / 扩展点逐一 / 压缩 / 取消。
// 测试代码豁免 std::function 禁令（AGENTS.md），mock 剧本全用 std::function。

#include <agent/agent/agent.hpp>
#include <agent/agent/compaction.hpp>
#include <agent/tools/tools.hpp>

#include "mock_provider.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

using namespace agent;
using namespace agent::test;

namespace {

// ── 自定义 Behaviors（override 钩子；非虚名字隐藏，签名与默认一致）──

class BlockingBehaviors : public DefaultBehaviors {
public:
    template<typename Provider>
    Result<bool> before_tool_call(Provider const&, ToolCall const& tool_call,
                                  ContextSnapshot const&) const
    {
        if (tool_call.name == "agent_weather")
            return false;
        return true;
    }
};

class StopAfterFirstBehaviors : public DefaultBehaviors {
public:
    template<typename Provider>
    Result<bool> should_stop(Provider const&, Message const&) const
    {
        return true;
    }
};

class CannedCompactBehaviors : public DefaultBehaviors {
public:
    std::vector<Message> replacement;
    template<typename Provider>
    Result<std::optional<std::vector<Message>>> compact(Provider const&, ContextSnapshot const&) const
    {
        return replacement;
    }
};

class InjectingBehaviors : public DefaultBehaviors {
public:
    Message extra;
    template<typename Provider>
    Result<std::vector<Message>> transform_context(Provider const&, ContextSnapshot const& snap) const
    {
        std::vector<Message> out = snap.messages;
        out.push_back(extra);
        return out;
    }
};

class SwitchModelBehaviors : public DefaultBehaviors {
public:
    ModelView next;
    template<typename Provider>
    Result<std::optional<NextTurnUpdate>> prepare_next_turn(
        Provider const&, Message const&, std::vector<ToolResult> const&) const
    {
        NextTurnUpdate update;
        update.model = next;
        return update;
    }
};

class KeyBehaviors : public DefaultBehaviors {
public:
    Result<std::optional<std::string>> get_api_key(std::string_view) const
    {
        return std::string("dynamic-key");
    }
};

/// 收集一次 run 的全部事件（驱动 generator 到结束）。
template<typename AgentT>
std::vector<AgentEvent> run_all(AgentT& agent, std::vector<Message> msgs = {},
                                StreamOptions const& opts = {})
{
    std::vector<AgentEvent> out;
    for (AgentEvent const& ev : agent.run(std::move(msgs), opts))
        out.push_back(ev);
    return out;
}

}  // namespace

TEST_CASE("基础 run：事件序列 + messages + usage/cost")
{
    auto st = reset_mock("t-basic");
    st->stream_script = [](Context const&, int) {
        return std::vector<StreamEvent>{
            text_delta("你好！"),
            done_text("你好！", Usage{ 5, 3, 0, 0, 8 }),
        };
    };
    Agent<MockProvider> agent(EndpointConfig{ .name = "t-basic", .api_key = "k" }, test_model());

    std::vector<AgentEvent> events = run_all(agent, { user("你好") });

    // 事件序列骨架
    CHECK(events.front().type() == AgentEvent::Type::AgentStart);
    CHECK(events.back().type() == AgentEvent::Type::AgentEnd);
    CHECK(count_type(events, AgentEvent::Type::TurnStart) == 1);
    CHECK(count_type(events, AgentEvent::Type::TurnEnd) == 1);
    CHECK(count_type(events, AgentEvent::Type::MessageStart) == 2);   // user + assistant
    CHECK(count_type(events, AgentEvent::Type::MessageEnd) == 2);
    CHECK(count_type(events, AgentEvent::Type::MessageUpdate) == 1);  // 一个 TextDelta
    CHECK(count_type(events, AgentEvent::Type::AgentError) == 0);

    // MessageUpdate 载荷是 TextDelta，partial 消息累积正文
    MessageUpdate const* update = first_of<MessageUpdate>(events);
    REQUIRE(update != nullptr);
    CHECK(std::holds_alternative<TextDelta>(update->delta));
    CHECK(std::get<TextDelta>(update->delta).text == "你好！");
    CHECK(!update->message.content.empty());

    // messages 落库
    REQUIRE(agent.messages().size() == 2);
    CHECK(agent.messages()[0].role == Role::User);
    CHECK(message_text(agent.messages()[1]) == "你好！");

    // usage / cost：单价 1/2 → (5*1 + 3*2)/1e6 = 11e-6
    REQUIRE(agent.last_usage().has_value());
    CHECK(agent.last_usage()->input_tokens == 5);
    CHECK(agent.last_usage()->output_tokens == 3);
    CHECK(agent.last_usage()->cost == doctest::Approx(11e-6));
    CHECK(agent.context_cost() == doctest::Approx(11e-6));
    CHECK(agent.context_tokens() >= 8);
}

TEST_CASE("thinking 增量事件")
{
    auto st = reset_mock("t-think");
    st->stream_script = [](Context const&, int) {
        return std::vector<StreamEvent>{
            thinking_delta("先想想"),
            text_delta("答案"),
            done_text("答案"),
        };
    };
    Agent<MockProvider> agent(EndpointConfig{ .name = "t-think" }, test_model());
    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    CHECK(count_type(events, AgentEvent::Type::MessageUpdate) == 2);
    // 第一个 MessageUpdate 是 ThinkingDelta
    MessageUpdate const* first = first_of<MessageUpdate>(events);
    REQUIRE(first != nullptr);
    CHECK(std::holds_alternative<ThinkingDelta>(first->delta));
    CHECK(std::get<ThinkingDelta>(first->delta).text == "先想想");
}

TEST_CASE("工具往返闭环：事件顺序 + 二次请求带工具结果 + messages")
{
    ensure_weather_tool();
    auto st = reset_mock("t-tool");
    st->stream_script = weather_tool_roundtrip;
    Agent<MockProvider> agent(EndpointConfig{ .name = "t-tool", .api_key = "k" }, test_model());

    std::vector<AgentEvent> events = run_all(agent, { user("查天气杭州") });

    // 两轮（工具轮 + 答案轮）
    CHECK(count_type(events, AgentEvent::Type::TurnStart) == 2);
    CHECK(count_type(events, AgentEvent::Type::ToolCallDelta) == 2);   // 两段参数增量
    CHECK(count_type(events, AgentEvent::Type::ToolCallEnd) == 1);
    CHECK(count_type(events, AgentEvent::Type::ToolExecStart) == 1);
    CHECK(count_type(events, AgentEvent::Type::ToolExecEnd) == 1);
    CHECK(count_type(events, AgentEvent::Type::AgentError) == 0);

    // 顺序：ToolCallEnd 在 ToolExecStart 前
    auto call_end_it = std::find_if(events.begin(), events.end(), [](AgentEvent const& e) {
        return e.type() == AgentEvent::Type::ToolCallEnd;
    });
    auto exec_start_it = std::find_if(events.begin(), events.end(), [](AgentEvent const& e) {
        return e.type() == AgentEvent::Type::ToolExecStart;
    });
    REQUIRE(call_end_it != events.end());
    REQUIRE(exec_start_it != events.end());
    CHECK(call_end_it < exec_start_it);

    // 工具执行成功，结果回传
    ToolExecEnd const* exec_end = first_of<ToolExecEnd>(events);
    REQUIRE(exec_end != nullptr);
    CHECK(exec_end->name == "agent_weather");
    CHECK(exec_end->result.output == "杭州 30°C 晴");
    CHECK(!exec_end->result.is_error);

    // 第二次请求的 ctx 含 tool 角色消息
    REQUIRE(st->seen_contexts.size() == 2);
    CHECK(has_role(st->seen_contexts[1].messages, Role::ToolResult));
    // 请求工具子集含 agent_weather
    CHECK(std::find(st->seen_contexts[0].tools.begin(), st->seen_contexts[0].tools.end(),
                    std::string("agent_weather")) != st->seen_contexts[0].tools.end());

    // messages：user, assistant(tool), toolresult, assistant(text)
    REQUIRE(agent.messages().size() == 4);
    CHECK(agent.messages()[1].role == Role::Assistant);
    CHECK(agent.messages()[2].role == Role::ToolResult);
    CHECK(agent.messages()[3].role == Role::Assistant);
    CHECK(message_text(agent.messages()[3]).find("杭州 30°C 晴") != std::string::npos);
}

TEST_CASE("before_tool_call 拒绝 → 模型收到被拒错误")
{
    ensure_weather_tool();
    auto st = reset_mock("t-block");
    st->stream_script = weather_tool_roundtrip;
    Agent<MockProvider, BlockingBehaviors> agent(EndpointConfig{ .name = "t-block" }, test_model(),
                                                 BlockingBehaviors{});

    std::vector<AgentEvent> events = run_all(agent, { user("查天气") });

    ToolExecEnd const* exec_end = first_of<ToolExecEnd>(events);
    REQUIRE(exec_end != nullptr);
    CHECK(exec_end->result.is_error);
    CHECK(exec_end->result.output.find("拒绝") != std::string::npos);

    // 第二次请求把被拒原因作为工具错误回传
    REQUIRE(st->seen_contexts.size() == 2);
    CHECK(has_tool_result_with(st->seen_contexts[1].messages, "拒绝"));
}

TEST_CASE("should_stop 提前停 → 无第二次请求")
{
    auto st = reset_mock("t-stop");
    st->stream_script = weather_tool_roundtrip;
    Agent<MockProvider, StopAfterFirstBehaviors> agent(EndpointConfig{ .name = "t-stop" }, test_model(),
                                                       StopAfterFirstBehaviors{});

    std::vector<AgentEvent> events = run_all(agent, { user("查天气") });

    CHECK(count_type(events, AgentEvent::Type::TurnStart) == 1);
    CHECK(st->seen_contexts.size() == 1);   // 提前停 → 工具结果未再请求
    CHECK(events.back().type() == AgentEvent::Type::AgentEnd);
}

TEST_CASE("steer 运行前排队 → 首轮请求注入")
{
    auto st = reset_mock("t-steer");
    st->stream_script = [](Context const&, int) {
        return std::vector<StreamEvent>{ text_delta("收到"), done_text("收到") };
    };
    Agent<MockProvider> agent(EndpointConfig{ .name = "t-steer" }, test_model());
    agent.steer(user("插话"));

    std::vector<AgentEvent> events = run_all(agent, { user("你好") });

    REQUIRE(!st->seen_contexts.empty());
    CHECK(message_text(st->seen_contexts[0].messages.back()) == "插话");
    CHECK(has_role(agent.messages(), Role::User));
}

TEST_CASE("follow_up 将停时注入 → 追加一轮")
{
    auto st = reset_mock("t-followup");
    st->stream_script = [](Context const&, int) {
        return std::vector<StreamEvent>{ text_delta("ok"), done_text("ok") };
    };
    Agent<MockProvider> agent(EndpointConfig{ .name = "t-followup" }, test_model());
    agent.follow_up(user("追加问题"));

    std::vector<AgentEvent> events = run_all(agent, { user("第一个问题") });

    CHECK(count_type(events, AgentEvent::Type::TurnStart) == 2);
    REQUIRE(st->seen_contexts.size() == 2);
    CHECK(message_text(st->seen_contexts[1].messages.back()) == "追加问题");
    CHECK(!agent.has_queued_messages());   // 注入后队列空
}

TEST_CASE("transform_context 改写：请求见但不落库")
{
    auto st = reset_mock("t-tc");
    st->stream_script = [](Context const&, int) { return std::vector<StreamEvent>{ done_text("ok") }; };
    InjectingBehaviors behaviors;
    behaviors.extra = user("注入的检索结果");
    Agent<MockProvider, InjectingBehaviors> agent(EndpointConfig{ .name = "t-tc" }, test_model(),
                                                  std::move(behaviors));

    std::vector<AgentEvent> events = run_all(agent, { user("你好") });

    REQUIRE(!st->seen_contexts.empty());
    CHECK(message_text(st->seen_contexts[0].messages.back()) == "注入的检索结果");
    CHECK(agent.messages().size() == 2);   // 注入不落库
}

TEST_CASE("prepare_next_turn 换模型")
{
    auto st = reset_mock("t-pnt");
    st->stream_script = weather_tool_roundtrip;
    SwitchModelBehaviors behaviors;
    static const std::string next_id = "next-model";
    ModelView next;
    next.id = next_id;
    next.context_window = 8000;
    behaviors.next = next;
    Agent<MockProvider, SwitchModelBehaviors> agent(EndpointConfig{ .name = "t-pnt" }, test_model(),
                                                    std::move(behaviors));

    std::vector<AgentEvent> events = run_all(agent, { user("查天气") });

    REQUIRE(st->seen_models.size() == 2);
    CHECK(st->seen_models[0] == "agent-test-model");
    CHECK(st->seen_models[1] == "next-model");
}

TEST_CASE("get_api_key 动态 key")
{
    auto st = reset_mock("t-key");
    st->stream_script = [](Context const&, int) { return std::vector<StreamEvent>{ done_text("ok") }; };
    Agent<MockProvider, KeyBehaviors> agent(EndpointConfig{ .name = "t-key" }, test_model(),
                                            KeyBehaviors{});

    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    REQUIRE(!st->seen_opts.empty());
    CHECK(st->seen_opts[0].api_key == "dynamic-key");
}

TEST_CASE("连续 run：消息累积")
{
    auto st = reset_mock("t-multi");
    st->stream_script = [](Context const&, int) {
        return std::vector<StreamEvent>{ text_delta("答"), done_text("答") };
    };
    Agent<MockProvider> agent(EndpointConfig{ .name = "t-multi" }, test_model());

    run_all(agent, { user("第一问") });
    run_all(agent, { user("第二问") });

    REQUIRE(agent.messages().size() == 4);
    CHECK(agent.messages()[0].role == Role::User);
    CHECK(agent.messages()[2].role == Role::User);
    CHECK(st->seen_contexts.size() == 2);
}

TEST_CASE("流中断（无终结事件）→ NetworkError")
{
    auto st = reset_mock("t-trunc");
    st->stream_script = [](Context const&, int) {
        return std::vector<StreamEvent>{ text_delta("半截") };   // 无 Done
    };
    Agent<MockProvider> agent(EndpointConfig{ .name = "t-trunc" }, test_model());

    std::vector<AgentEvent> events = run_all(agent, { user("hi") });

    REQUIRE(agent.last_error().has_value());
    CHECK(agent.last_error()->code == Errc::NetworkError);
    CHECK(count_type(events, AgentEvent::Type::AgentError) == 1);
    CHECK(events.back().type() == AgentEvent::Type::AgentEnd);
}

TEST_CASE("abort 并发取消 → Aborted")
{
    ensure_weather_tool();
    auto st = reset_mock("t-abort");
    st->stream_script = weather_tool_roundtrip;
    st->stream_delay_ms = 1000;   // 剧本阻塞：让 loop 卡在 receive 上（取消不生效时最坏 1s 兜底）
    Agent<MockProvider> agent(EndpointConfig{ .name = "t-abort" }, test_model());

    // 测试允许 try/catch；捕获到异常 → 直接测试失败（不静默吞）
    std::exception_ptr thread_error;
    std::thread runner([&] {
        try {
            for (AgentEvent const& ev : agent.run({ user("查天气") })) {
                (void)ev;
            }
        } catch (...) {
            thread_error = std::current_exception();
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));   // 等 loop 进入流消费
    agent.abort();
    runner.join();

    // 若 run 消费过程抛异常 → 测试直接失败
    if (thread_error) {
        try {
            std::rethrow_exception(thread_error);
        } catch (std::exception const& e) {
            FAIL("run 消费过程抛异常: " << e.what());
        } catch (...) {
            FAIL("run 消费过程抛未知异常");
        }
    }

    REQUIRE(agent.last_error().has_value());
    CHECK(agent.last_error()->code == Errc::Aborted);
    CHECK(!agent.is_streaming());
}

TEST_CASE("压缩未触发：不压原样")
{
    auto st = reset_mock("t-c-notrig");
    st->stream_script = [](Context const&, int) { return std::vector<StreamEvent>{ done_text("ok") }; };
    Agent<MockProvider> agent(EndpointConfig{ .name = "t-c-notrig" }, test_model(100000));

    run_all(agent, { user("你好") });
    CHECK(!agent.should_compact());
    Result<bool> r = agent.compact();
    REQUIRE(r);
    CHECK(*r == false);
    CHECK(agent.messages().size() == 2);
}

TEST_CASE("压缩触发：override 完全接管")
{
    auto st = reset_mock("t-c-canned");
    CannedCompactBehaviors behaviors;
    behaviors.replacement = { Message{ Role::User,
        { Text{ std::string(kCompactionSummaryPrefix) + "旧内容" } } } };
    Agent<MockProvider, CannedCompactBehaviors> agent(EndpointConfig{ .name = "t-c-canned" },
                                                      test_model(200), std::move(behaviors));
    st->stream_script = [](Context const&, int) { return std::vector<StreamEvent>{ done_text("ok") }; };

    Result<bool> r = agent.compact();
    REQUIRE(r);
    CHECK(*r == true);
    REQUIRE(agent.messages().size() == 1);
    CHECK(is_compaction_summary(agent.messages().front()));
    CHECK(agent.previous_summary() == "旧内容");
}

TEST_CASE("run 内防爆：超限自动压缩")
{
    auto st = reset_mock("t-c-auto");
    CannedCompactBehaviors behaviors;
    behaviors.replacement = { Message{ Role::User,
        { Text{ std::string(kCompactionSummaryPrefix) + "自动压缩" } } } };
    Agent<MockProvider, CannedCompactBehaviors> agent(EndpointConfig{ .name = "t-c-auto" },
                                                      test_model(200), std::move(behaviors));
    st->stream_script = [](Context const&, int) { return std::vector<StreamEvent>{ done_text("ok") }; };

    std::vector<AgentEvent> events = run_all(agent, { user("你好") });

    // 小窗口 → 每轮请求前内防爆触发 → messages 前缀是罐头摘要
    REQUIRE(!agent.messages().empty());
    CHECK(is_compaction_summary(agent.messages().front()));
    CHECK(agent.previous_summary() == "自动压缩");
}

TEST_CASE("默认压缩：mock complete 摘要 + 请求构造断言")
{
    auto st = reset_mock("t-c-summary");
    DefaultBehaviors behaviors;
    behaviors.settings_ = CompactionSettings{ 50, 100, 2 };
    Agent<MockProvider> agent(EndpointConfig{ .name = "t-c-summary" }, test_model(200),
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

    // 攒三轮大上下文（每轮 user 200 字符 = 50 token；锚点 usage 504 > 阈值 150）。
    // 第三轮 run 内防爆触发默认压缩：mock complete 生成摘要 → messages 被替换。
    for (int i = 0; i < 3; ++i)
        run_all(agent, { user(std::string(200, static_cast<char>('a' + i))) });

    CHECK(agent.should_compact());

    // 压缩结果：新对话 = [摘要] + 保留段（cut=2 → [u1, a1, u2]）+ 第三轮助手回复
    REQUIRE(agent.messages().size() == 5);
    CHECK(is_compaction_summary(agent.messages()[0]));
    CHECK(extract_summary_text(agent.messages()[0]) == "摘要：用户问了天气。");
    CHECK(message_text(agent.messages()[1]) == std::string(200, 'b'));
    CHECK(message_text(agent.messages()[3]) == std::string(200, 'c'));
    CHECK(agent.previous_summary() == "摘要：用户问了天气。");

    // 摘要请求构造：tools 留空、thinking 关、max_tokens 钳制、指令作尾部 user、system 不变
    REQUIRE(st->complete_contexts.size() == 1);
    CHECK(st->complete_contexts[0].tools.empty());
    CHECK(st->complete_contexts[0].system_prompt.empty());
    REQUIRE(!st->complete_contexts[0].messages.empty());
    Message const& instruction = st->complete_contexts[0].messages.back();
    CHECK(instruction.role == Role::User);
    CHECK(message_text(instruction).find("压缩成一个简洁的会话摘要") != std::string::npos);
    REQUIRE(st->complete_opts.size() == 1);
    REQUIRE(st->complete_opts[0].reasoning.has_value());
    CHECK(*st->complete_opts[0].reasoning == ThinkingLevel::Off);
    REQUIRE(st->complete_opts[0].max_tokens.has_value());
    CHECK(*st->complete_opts[0].max_tokens == 4096);
}

TEST_CASE("压缩失败：摘要响应调工具")
{
    auto st = reset_mock("t-c-tool");
    DefaultBehaviors behaviors;
    behaviors.settings_ = CompactionSettings{ 50, 100, 2 };
    Agent<MockProvider> agent(EndpointConfig{ .name = "t-c-tool" }, test_model(200),
                              std::move(behaviors));
    st->stream_script = [](Context const&, int) {
        return std::vector<StreamEvent>{ text_delta("ok"), done_text("ok", Usage{ 500, 4, 0, 0, 504 }) };
    };
    st->complete_script = [](Context const&) -> Result<ChatResponse> {
        ChatResponse response;
        response.content.push_back(ToolCall{ "c", "agent_weather", nlohmann::json::object() });
        return response;
    };

    // 第三轮内防爆压缩触发 → 摘要响应调工具 → 压缩失败（run 以 CompactionFailed 中止）
    for (int i = 0; i < 3; ++i)
        run_all(agent, { user(std::string(200, 'a')) });
    REQUIRE(agent.last_error().has_value());
    CHECK(agent.last_error()->code == Errc::CompactionFailed);
    CHECK(agent.last_error()->message.find("工具调用") != std::string::npos);
    // 外部主动压缩同样失败（失败不改 messages）
    Result<bool> r = agent.compact();
    REQUIRE(!r);
    CHECK(r.error().code == Errc::CompactionFailed);
    CHECK(r.error().message.find("工具调用") != std::string::npos);
}

TEST_CASE("压缩失败：空摘要")
{
    auto st = reset_mock("t-c-empty");
    DefaultBehaviors behaviors;
    behaviors.settings_ = CompactionSettings{ 50, 100, 2 };
    Agent<MockProvider> agent(EndpointConfig{ .name = "t-c-empty" }, test_model(200),
                              std::move(behaviors));
    st->stream_script = [](Context const&, int) {
        return std::vector<StreamEvent>{ text_delta("ok"), done_text("ok", Usage{ 500, 4, 0, 0, 504 }) };
    };
    st->complete_script = [](Context const&) -> Result<ChatResponse> {
        return ChatResponse{};   // 空响应 → 空摘要
    };

    // 第三轮内防爆压缩触发 → 摘要为空 → 压缩失败
    for (int i = 0; i < 3; ++i)
        run_all(agent, { user(std::string(200, 'a')) });
    REQUIRE(agent.last_error().has_value());
    CHECK(agent.last_error()->code == Errc::CompactionFailed);
    CHECK(agent.last_error()->message.find("为空") != std::string::npos);
    // 外部主动压缩同样失败
    Result<bool> r = agent.compact();
    REQUIRE(!r);
    CHECK(r.error().code == Errc::CompactionFailed);
    CHECK(r.error().message.find("为空") != std::string::npos);
}

TEST_CASE("reset 清空状态")
{
    auto st = reset_mock("t-reset");
    st->stream_script = [](Context const&, int) { return std::vector<StreamEvent>{ done_text("ok") }; };
    Agent<MockProvider> agent(EndpointConfig{ .name = "t-reset" }, test_model());
    run_all(agent, { user("hi") });
    REQUIRE(agent.messages().size() == 2);

    agent.reset();
    CHECK(agent.messages().empty());
    CHECK(!agent.last_error().has_value());
    CHECK(!agent.last_usage().has_value());
    CHECK(!agent.has_queued_messages());
    CHECK(agent.context_cost() == 0);
}

TEST_CASE("has_queued_messages / clear_queues")
{
    Agent<MockProvider> agent(EndpointConfig{ .name = "t-queue" }, test_model());
    CHECK(!agent.has_queued_messages());
    agent.steer(user("a"));
    agent.follow_up(user("b"));
    CHECK(agent.has_queued_messages());
    agent.clear_queues();
    CHECK(!agent.has_queued_messages());
}

TEST_CASE("Agent 不可拷贝 / 移动（编译期拒绝）")
{
    static_assert(!std::is_copy_constructible_v<Agent<MockProvider>>);
    static_assert(!std::is_copy_assignable_v<Agent<MockProvider>>);
    static_assert(!std::is_move_constructible_v<Agent<MockProvider>>);
    static_assert(!std::is_move_assignable_v<Agent<MockProvider>>);
    CHECK(true);
}
