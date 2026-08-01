// 异步通道测试：AsyncStream 基本收发 / 关闭语义 / try_receive / 背压。

#include <agent/llm/stream.hpp>

#include <doctest/doctest.h>

using namespace agent;

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
