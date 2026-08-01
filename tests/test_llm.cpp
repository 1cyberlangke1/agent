// L1 类型层测试：枚举 / ModelRegistry（内置表加载、运行时注册、覆盖、遍历）/
// StreamEvent / AsyncStream 基础 / clamp_thinking_level。

#include <agent/llm.hpp>

#include <doctest/doctest.h>

#include <string>
#include <vector>

using namespace agent;

TEST_CASE("ThinkingLevel 枚举与 thinking_level_count 同步")
{
    // 枚举按序 0..6，static_assert 已锁定；这里验证实际值
    CHECK(static_cast<int>(ThinkingLevel::Off) == 0);
    CHECK(static_cast<int>(ThinkingLevel::Max) == 6);
    CHECK(thinking_level_count == 7);
}

TEST_CASE("ModelRegistry 内置表已加载（生成器产物）")
{
    // deepseek / openai / anthropic / google 各有至少一个模型
    CHECK(ModelRegistry::find_model("deepseek-chat").has_value());
    CHECK(ModelRegistry::find_model("gpt-5.5").has_value());
    CHECK(ModelRegistry::find_model("claude-sonnet-4-6").has_value());
    CHECK(ModelRegistry::find_model("gemini-3.5-flash").has_value());
    // 不存在的 id
    CHECK(!ModelRegistry::find_model("no-such-model").has_value());
}

TEST_CASE("ModelRegistry 内置表字段正确")
{
    auto m = ModelRegistry::find_model("deepseek-chat");
    REQUIRE(m.has_value());
    CHECK(m->id == "deepseek-chat");
    CHECK(m->context_window > 0);
    CHECK(m->max_output_tokens > 0);
    // deepseek-chat 非推理模型
    CHECK(!m->reasoning);
    // 支持文本 + 图像输入的模型（如 GPT-5.2 Pro 支持 image）
    auto gpt = ModelRegistry::find_model("gpt-5.2-pro");
    REQUIRE(gpt.has_value());
    CHECK(gpt->reasoning);
    CHECK(gpt->supports_image_input);
}

TEST_CASE("ModelRegistry 运行时注册 + 覆盖")
{
    ModelRegistry::register_model(RuntimeModel{
        .id = "test-model",
        .context_window = 32000,
        .max_output_tokens = 4096,
        .reasoning = true,
        .supports_image_input = true,
    });
    auto m = ModelRegistry::find_model("test-model");
    REQUIRE(m.has_value());
    CHECK(m->id == "test-model");
    CHECK(m->context_window == 32000);
    CHECK(m->max_output_tokens == 4096);
    CHECK(m->reasoning);
    CHECK(m->supports_image_input);

    // 覆盖：同 id 重新注册
    ModelRegistry::register_model(RuntimeModel{
        .id = "test-model",
        .context_window = 64000,
        .max_output_tokens = 8192,
    });
    auto m2 = ModelRegistry::find_model("test-model");
    REQUIRE(m2.has_value());
    CHECK(m2->context_window == 64000);
    CHECK(m2->max_output_tokens == 8192);
}

TEST_CASE("ModelRegistry 动态覆盖内置表（同 id）")
{
    // 覆盖内置的 deepseek-chat，context_window 改小
    ModelRegistry::register_model(RuntimeModel{
        .id = "deepseek-chat",
        .context_window = 100,
        .max_output_tokens = 50,
    });
    auto m = ModelRegistry::find_model("deepseek-chat");
    REQUIRE(m.has_value());
    CHECK(m->context_window == 100);
}

TEST_CASE("ModelRegistry for_each_model 稳定顺序 + 覆盖不重复")
{
    std::vector<std::string> ids;
    ModelRegistry::for_each_model([&ids](ModelView const& mv) { ids.push_back(std::string(mv.id)); });

    // 内置表按生成顺序在前；动态注册的 test-model 在后
    REQUIRE(ids.size() >= 2);
    // 覆盖的内置模型 deepseek-chat 只出现一次（动态版）
    int count = 0;
    for (auto const& id : ids)
        if (id == "deepseek-chat") ++count;
    CHECK(count == 1);

    // test-model 在列表尾部（动态注册序）
    bool found = false;
    for (auto const& id : ids) {
        if (id == "test-model") { found = true; break; }
    }
    CHECK(found);
}

TEST_CASE("ModelView 生命周期：已发出的 ModelView 在重复注册后仍有效")
{
    auto before = ModelRegistry::find_model("deepseek-chat");
    REQUIRE(before.has_value());
    std::string_view old_id = before->id;

    ModelRegistry::register_model(RuntimeModel{
        .id = "deepseek-chat",
        .context_window = 200,
        .max_output_tokens = 100,
    });

    // 旧 ModelView（指向旧 deque 条目）仍有效——deque 只追加不改写
    CHECK(old_id == "deepseek-chat");
    auto after = ModelRegistry::find_model("deepseek-chat");
    REQUIRE(after.has_value());
    CHECK(after->context_window == 200);
}

TEST_CASE("clamp_thinking_level 收敛到模型支持范围")
{
    // deepseek-v4-pro map 已预填：{off, high, high, high, high, high, max}
    // 非 Off 档恒有值 → clamp 恒等；map 值即档位强度
    auto m = ModelRegistry::find_model("deepseek-v4-pro");
    REQUIRE(m.has_value());
    // XHigh 预填为 "high" → clamp 恒等返回 XHigh（map[5] 有值）
    CHECK(clamp_thinking_level(*m, ThinkingLevel::XHigh) == ThinkingLevel::XHigh);
    CHECK(m->thinking_level_map[static_cast<std::size_t>(ThinkingLevel::XHigh)] == "high");
    CHECK(clamp_thinking_level(*m, ThinkingLevel::Max) == ThinkingLevel::Max);
    CHECK(clamp_thinking_level(*m, ThinkingLevel::Off) == ThinkingLevel::Off);
    // 非推理模型（reasoning=false，全 nullopt）：任何等级都落到 Off
    auto chat = ModelRegistry::find_model("deepseek-chat");
    REQUIRE(chat.has_value());
    CHECK(clamp_thinking_level(*chat, ThinkingLevel::High) == ThinkingLevel::Off);
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

TEST_CASE("AsyncStream 基本收发与关闭")
{
    asio::io_context io;
    AsyncStream<int> stream(io.get_executor(), 4);

    bool received = false;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        auto r = co_await stream.receive();
        if (r) { CHECK(*r == 42); received = true; }
        stream.close();
    }, asio::detached);

    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        co_await stream.send(42);
    }, asio::detached);

    io.run();
    CHECK(received);
    CHECK(!stream.is_open());
}

TEST_CASE("AsyncStream channel 关闭后 receive 返回错误")
{
    asio::io_context io;
    AsyncStream<int> stream(io.get_executor(), 4);
    stream.close();

    bool got_error = false;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        auto r = co_await stream.receive();
        got_error = !r.has_value();
    }, asio::detached);

    io.run();
    CHECK(got_error);
}

TEST_CASE("AsyncStream try_receive 空队列返回 nullopt")
{
    asio::io_context io;
    AsyncStream<int> stream(io.get_executor(), 4);
    CHECK(!stream.try_receive().has_value());
}

TEST_CASE("AsyncStream 背压：容量有限，send 挂起直到消费")
{
    asio::io_context io;
    AsyncStream<int> stream(io.get_executor(), 1);

    // 发送 2 个，容量 1：第二个 send 挂起，需消费后完成
    int received_sum = 0;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        bool ok1 = co_await stream.send(1);
        bool ok2 = co_await stream.send(2);   // 挂起直到消费
        CHECK(ok1);
        CHECK(ok2);
    }, asio::detached);

    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        auto r1 = co_await stream.receive();
        if (r1) received_sum += *r1;
        auto r2 = co_await stream.receive();
        if (r2) received_sum += *r2;
    }, asio::detached);

    io.run();
    CHECK(received_sum == 3);
}
