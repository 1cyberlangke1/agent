// 类型层测试：枚举（ThinkingLevel 同步）/ StreamEvent（type 判别 + visit 访问）。

#include <agent/llm/types.hpp>

#include <doctest/doctest.h>

#include <string>
#include <variant>

using namespace agent;

TEST_CASE("ThinkingLevel 枚举与 thinking_level_count 同步")
{
    // 枚举按序 0..6，static_assert 已锁定；这里验证实际值
    CHECK(static_cast<int>(ThinkingLevel::Off) == 0);
    CHECK(static_cast<int>(ThinkingLevel::Max) == 6);
    CHECK(thinking_level_count == 7);
}

TEST_CASE("StreamEvent type() 从 variant 下标推导")
{
    StreamEvent ev{ TextDelta{ "hi" } };
    CHECK(ev.type() == StreamEvent::Type::TextDelta);
    ev = StreamEvent{ DoneEvent{ ChatResponse{} } };
    CHECK(ev.type() == StreamEvent::Type::Done);
    ev = StreamEvent{ Error{ Errc::RateLimited, "slow down" } };
    CHECK(ev.type() == StreamEvent::Type::Error);
}

TEST_CASE("StreamEvent visit 访问载荷")
{
    StreamEvent ev{ ToolCallEnd{ "t1", "get_weather", {{"location", "杭州"}} } };
    std::string name;
    std::visit([&name](auto const& v) {
        if constexpr (std::is_same_v<std::decay_t<decltype(v)>, ToolCallEnd>)
            name = v.name;
    }, ev.data);
    CHECK(name == "get_weather");
}
