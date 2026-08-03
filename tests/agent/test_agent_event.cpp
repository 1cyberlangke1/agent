// AgentEvent 类型层测试：13 种事件 type 判别（variant index 与 Type 枚举严格一致）。

#include <agent/agent/agent_event.hpp>

#include <doctest/doctest.h>

#include <string>
#include <variant>

using namespace agent;

TEST_CASE("AgentEvent 载荷类型数与 Type 枚举一致")
{
    static_assert(std::variant_size_v<decltype(AgentEvent::data)> == 13);
    CHECK(std::variant_size_v<decltype(AgentEvent::data)> == 13);
}

TEST_CASE("AgentEvent 13 种事件 type 判别")
{
    Message user_message{ Role::User, { Text{ "你好" } } };
    ToolResult tr{ "call_1", "杭州 30°C", false };

    std::vector<std::pair<AgentEvent, AgentEvent::Type>> cases{
        { AgentEvent{ AgentStart{} }, AgentEvent::Type::AgentStart },
        { AgentEvent{ AgentEnd{ { user_message } } }, AgentEvent::Type::AgentEnd },
        { AgentEvent{ TurnStart{} }, AgentEvent::Type::TurnStart },
        { AgentEvent{ TurnEnd{ user_message, { tr } } }, AgentEvent::Type::TurnEnd },
        { AgentEvent{ MessageStart{ user_message } }, AgentEvent::Type::MessageStart },
        { AgentEvent{ MessageUpdate{ user_message, TextDelta{ "hi" } } }, AgentEvent::Type::MessageUpdate },
        { AgentEvent{ MessageUpdate{ user_message, ThinkingDelta{ "想" } } }, AgentEvent::Type::MessageUpdate },
        { AgentEvent{ MessageEnd{ user_message } }, AgentEvent::Type::MessageEnd },
        { AgentEvent{ ToolCallDelta{ "call_1", "get_weather", "{\"loc" } }, AgentEvent::Type::ToolCallDelta },
        { AgentEvent{ ToolCallEnd{ "call_1", "get_weather", nlohmann::json{{"location", "杭州"}} } },
          AgentEvent::Type::ToolCallEnd },
        { AgentEvent{ ToolExecStart{ "call_1", "get_weather", nlohmann::json{{"location", "杭州"}} } },
          AgentEvent::Type::ToolExecStart },
        { AgentEvent{ ToolExecUpdate{ "call_1", "get_weather", nlohmann::json{{"location", "杭州"}}, "部分" } },
          AgentEvent::Type::ToolExecUpdate },
        { AgentEvent{ ToolExecEnd{ "call_1", "get_weather", tr } }, AgentEvent::Type::ToolExecEnd },
        { AgentEvent{ AgentError{ Error{ Errc::NetworkError, "断流" } } }, AgentEvent::Type::AgentError },
    };

    for (auto const& [event, expected] : cases)
        CHECK(event.type() == expected);
}

TEST_CASE("AgentEvent 载荷可通过 std::get 提取")
{
    AgentEvent event{ ToolExecEnd{ "call_1", "get_weather", ToolResult{ "call_1", "ok", false } } };
    ToolExecEnd const* payload = std::get_if<ToolExecEnd>(&event.data);
    REQUIRE(payload != nullptr);
    CHECK(payload->id == "call_1");
    CHECK(payload->result.output == "ok");
}

TEST_CASE("MessageUpdate.delta 为 TextDelta / ThinkingDelta 变体")
{
    Message partial{ Role::Assistant, { Text{ "你" } } };
    MessageUpdate update{ partial, ThinkingDelta{ "好" } };
    ThinkingDelta const* delta = std::get_if<ThinkingDelta>(&update.delta);
    REQUIRE(delta != nullptr);
    CHECK(delta->text == "好");
}
