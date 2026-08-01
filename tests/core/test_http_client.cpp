// http_client（curl multi_socket × asio 胶水层）测试。
//
// 全部对着进程内 MockServer（127.0.0.1 明文 http）跑：
// - 请求侧：method / target / 头 / body 由 server 线程断言（errors() 收集）
// - 响应侧：status / headers / body 拼接 / EOF / 错误路径
// - 传输语义：首字节前重试（429/5xx/连接失败）、首字节后绝不重试、
//   idle 超时、取消信号
//
// TLS 路径不在此测（schannel 对 mock 需要证书部署；生产 TLS 由 curl 负责）。

#include <agent/core/http_client.hpp>

#include "mock_server.hpp"

#include <doctest/doctest.h>

#include <brotli/encode.h>
#include <zlib.h>

#include <atomic>
#include <chrono>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using agent::HttpRequestOptions;
using agent::HttpResponse;
using agent::HttpStreamReader;
using agent::Result;
using agent::test::MockServer;
using agent::test::RequestView;

namespace {

/// 在独立 io_context 上同步跑一个 awaitable 拿结果（测试便捷壳）。
template<typename T>
T run_awaitable(asio::awaitable<T> coroutine_task)
{
    asio::io_context io;
    std::optional<T> outcome;
    asio::co_spawn(io, std::move(coroutine_task),
        [&outcome](std::exception_ptr, T value) { outcome.emplace(std::move(value)); });
    io.run();
    REQUIRE(outcome.has_value());
    return std::move(*outcome);
}

/// 组装一条规范的 Content-Length 响应剧本（单 Chunk）。
MockServer::Exchange plain_response(int status_code, std::string const& reason,
                                    std::string const& content_type,
                                    std::string const& body,
                                    std::string const& extra_headers = {})
{
    std::string head = "HTTP/1.1 " + std::to_string(status_code) + " " + reason + "\r\n"
        + "Content-Type: " + content_type + "\r\n"
        + "Content-Length: " + std::to_string(body.size()) + "\r\n"
        + extra_headers
        + "Connection: close\r\n\r\n";
    return MockServer::Exchange{.expect = nullptr,
                                .chunks = {{head + body, 0}},
                                .close_abruptly = false};
}

/// zlib 现场压缩为 gzip 格式（windowBits=31 = gzip 包装），供压缩剧本用。
std::string gzip_compress(std::string const& input)
{
    z_stream stream{};
    REQUIRE(deflateInit2(&stream, Z_BEST_SPEED, Z_DEFLATED, 31, 8,
                         Z_DEFAULT_STRATEGY) == Z_OK);
    std::string output(input.size() + 256, '\0');
    stream.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(input.data()));
    stream.avail_in = static_cast<uInt>(input.size());
    stream.next_out = reinterpret_cast<Bytef*>(output.data());
    stream.avail_out = static_cast<uInt>(output.size());
    REQUIRE(deflate(&stream, Z_FINISH) == Z_STREAM_END);
    output.resize(stream.total_out);
    deflateEnd(&stream);
    return output;
}

/// zlib 现场压缩为 zlib 格式（windowBits=15 = zlib 包装），供 deflate 剧本用。
std::string zlib_compress(std::string const& input)
{
    z_stream stream{};
    REQUIRE(deflateInit2(&stream, Z_BEST_SPEED, Z_DEFLATED, 15, 8,
                         Z_DEFAULT_STRATEGY) == Z_OK);
    std::string output(input.size() + 256, '\0');
    stream.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(input.data()));
    stream.avail_in = static_cast<uInt>(input.size());
    stream.next_out = reinterpret_cast<Bytef*>(output.data());
    stream.avail_out = static_cast<uInt>(output.size());
    REQUIRE(deflate(&stream, Z_FINISH) == Z_STREAM_END);
    output.resize(stream.total_out);
    deflateEnd(&stream);
    return output;
}

/// brotli 现场压缩（BrotliEncoderCompress），供 br 剧本用。
std::string brotli_compress(std::string const& input)
{
    std::size_t bound = BrotliEncoderMaxCompressedSize(input.size());
    std::string output(bound, '\0');
    std::size_t out_len = bound;
    REQUIRE(BrotliEncoderCompress(BROTLI_DEFAULT_QUALITY, BROTLI_DEFAULT_WINDOW, BROTLI_MODE_GENERIC,
                                  input.size(),
                                  reinterpret_cast<std::uint8_t const*>(input.data()),
                                  &out_len,
                                  reinterpret_cast<std::uint8_t*>(output.data()))
            == BROTLI_TRUE);
    output.resize(out_len);
    return output;
}

}  // namespace

TEST_CASE("async_http_post 基本请求与响应")
{
    MockServer server;
    MockServer::Exchange exchange = plain_response(200, "OK", "application/json",
                                                   R"({"echo":true})");
    exchange.expect = [&server](RequestView const& request) {
        if (request.method != "POST")
            server.record_error("method != POST: " + request.method);
        if (request.target != "/v1/chat/completions")
            server.record_error("target 错误: " + request.target);
        if (request.header("content-type") != "application/json")
            server.record_error("content-type 错误");
        if (request.body != R"({"model":"test-model"})")
            server.record_error("body 错误: " + request.body);
        // Expect: 100-continue 必须被禁用（否则流式首包延迟一个 RTT）
        if (!request.header("expect").empty())
            server.record_error("Expect 头未被禁用");
    };
    server.enqueue(std::move(exchange));

    Result<HttpResponse> response = run_awaitable([&]() -> asio::awaitable<Result<HttpResponse>> {
        co_return co_await agent::async_http_post(
            co_await asio::this_coro::executor,
            server.base_url() + "/v1/chat/completions",
            nlohmann::json{{"model", "test-model"}}, HttpRequestOptions{});
    }());

    REQUIRE(response.has_value());
    CHECK(response->status == 200);
    CHECK(response->body == R"({"echo":true})");
    bool found_content_type = false;
    for (auto const& [name, value] : response->headers) {
        if (name == "Content-Type" && value == "application/json")
            found_content_type = true;
    }
    CHECK(found_content_type);
    CHECK(server.request_count() == 1);
    CHECK(server.errors().empty());
}

TEST_CASE("HttpStreamReader 分块流式读取到 EOF")
{
    std::string const part1 = "first-part-";
    std::string const part2 = "second-part-";
    std::string const part3 = "third-part";
    std::string const full_body = part1 + part2 + part3;

    MockServer server;
    std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        "Content-Length: " + std::to_string(full_body.size()) + "\r\n"
        "Connection: close\r\n\r\n";
    // 头与三段 body 分四次写入、带间隔——验证跨 TCP 包的流式交付
    server.enqueue(MockServer::Exchange{
        .expect = nullptr,
        .chunks = {{head, 20}, {part1, 20}, {part2, 20}, {part3, 0}},
        .close_abruptly = false});

    struct StreamOutcome {
        int status = 0;
        std::string collected;
        int chunk_count = 0;
        bool clean_eof = false;
    };
    Result<StreamOutcome> outcome = run_awaitable([&]() -> asio::awaitable<Result<StreamOutcome>> {
        Result<HttpStreamReader> reader = co_await HttpStreamReader::open(
            co_await asio::this_coro::executor, server.base_url() + "/stream",
            nlohmann::json::object(), HttpRequestOptions{});
        if (!reader)
            co_return std::unexpected(reader.error());
        StreamOutcome result;
        result.status = reader->status();
        while (true) {
            Result<std::optional<std::string>> chunk = co_await reader->next_chunk();
            if (!chunk)
                co_return std::unexpected(chunk.error());
            if (!chunk->has_value()) {
                result.clean_eof = true;
                break;
            }
            result.collected += **chunk;
            result.chunk_count += 1;
        }
        co_return result;
    }());

    REQUIRE(outcome.has_value());
    CHECK(outcome->status == 200);
    CHECK(outcome->collected == full_body);
    CHECK(outcome->chunk_count >= 2);   // 分次交付（TCP 合并允许少于 4）
    CHECK(outcome->clean_eof);
    CHECK(server.errors().empty());
}

TEST_CASE("chunked 编码由 curl 透明解码")
{
    MockServer server;
    // 手写 chunked 帧：两块 + 终止块
    std::string response_bytes =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/plain\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Connection: close\r\n\r\n"
        "b\r\nhello-chunk\r\n"
        "6\r\n-world\r\n"
        "0\r\n\r\n";
    server.enqueue(MockServer::Exchange{
        .expect = nullptr, .chunks = {{response_bytes, 0}}, .close_abruptly = false});

    Result<HttpResponse> response = run_awaitable([&]() -> asio::awaitable<Result<HttpResponse>> {
        co_return co_await agent::async_http_post(
            co_await asio::this_coro::executor, server.base_url() + "/chunked",
            nlohmann::json::object(), HttpRequestOptions{});
    }());

    REQUIRE(response.has_value());
    CHECK(response->status == 200);
    CHECK(response->body == "hello-chunk-world");   // chunked 帧头已被 curl 剥掉
    CHECK(server.errors().empty());
}

TEST_CASE("429 带 Retry-After 重试后成功")
{
    MockServer server;
    server.enqueue(plain_response(429, "Too Many Requests", "application/json",
                                  R"({"error":"rate limited"})",
                                  "Retry-After: 0\r\n"));
    server.enqueue(plain_response(200, "OK", "application/json", R"({"ok":true})"));

    Result<HttpResponse> response = run_awaitable([&]() -> asio::awaitable<Result<HttpResponse>> {
        co_return co_await agent::async_http_post(
            co_await asio::this_coro::executor, server.base_url() + "/retry",
            nlohmann::json::object(), HttpRequestOptions{});
    }());

    REQUIRE(response.has_value());
    CHECK(response->status == 200);
    CHECK(response->body == R"({"ok":true})");
    CHECK(server.request_count() == 2);
    CHECK(server.errors().empty());
}

TEST_CASE("5xx 重试耗尽后返回最后一次响应")
{
    MockServer server;
    for (int i = 0; i < 3; ++i)
        server.enqueue(plain_response(500, "Internal Server Error", "application/json",
                                      R"({"error":"boom"})", "Retry-After: 0\r\n"));

    HttpRequestOptions options;
    options.max_retries = 2;   // 1 次原始 + 2 次重试 = 3 个请求
    Result<HttpResponse> response = run_awaitable([&]() -> asio::awaitable<Result<HttpResponse>> {
        co_return co_await agent::async_http_post(
            co_await asio::this_coro::executor, server.base_url() + "/exhaust",
            nlohmann::json::object(), std::move(options));
    }());

    REQUIRE(response.has_value());   // 重试耗尽 ≠ 传输失败：响应交上层读详情
    CHECK(response->status == 500);
    CHECK(response->body == R"({"error":"boom"})");
    CHECK(server.request_count() == 3);
    CHECK(server.errors().empty());
}

TEST_CASE("4xx 不重试且错误 body 可读")
{
    MockServer server;
    server.enqueue(plain_response(401, "Unauthorized", "application/json",
                                  R"({"error":{"message":"bad key"}})"));

    Result<HttpResponse> response = run_awaitable([&]() -> asio::awaitable<Result<HttpResponse>> {
        co_return co_await agent::async_http_post(
            co_await asio::this_coro::executor, server.base_url() + "/auth",
            nlohmann::json::object(), HttpRequestOptions{});
    }());

    REQUIRE(response.has_value());
    CHECK(response->status == 401);
    CHECK(response->body == R"({"error":{"message":"bad key"}})");
    CHECK(server.request_count() == 1);   // 4xx（非 429）绝不重试
    CHECK(server.errors().empty());
}

TEST_CASE("首字节后断流：错误终结且绝不重发")
{
    MockServer server;
    // 声明 100 字节实发 10 字节后 RST——流中断连。
    // 关键时序：写完 head+10 字节后留 150ms 窗口，确保 curl 已读完响应头
    // （open 成功）再断连——否则 RST 撞在读头阶段会被 curl 当成「首字节前
    // 失败」而重试（request_count 变成 3），测的不是目标行为。
    std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        "Content-Length: 100\r\nConnection: close\r\n\r\n";
    server.enqueue(MockServer::Exchange{
        .expect = nullptr,
        .chunks = {{head + "partial-10", 250}},
        .close_abruptly = true});

    struct BrokenStreamOutcome {
        bool open_ok = false;
        std::string received;
        bool got_error = false;
    };
    BrokenStreamOutcome outcome = run_awaitable([&]() -> asio::awaitable<BrokenStreamOutcome> {
        BrokenStreamOutcome result;
        Result<HttpStreamReader> reader = co_await HttpStreamReader::open(
            co_await asio::this_coro::executor, server.base_url() + "/broken",
            nlohmann::json::object(), HttpRequestOptions{});
        if (!reader)
            co_return result;
        result.open_ok = true;
        while (true) {
            Result<std::optional<std::string>> chunk = co_await reader->next_chunk();
            if (!chunk) {
                result.got_error = true;
                break;
            }
            if (!chunk->has_value())
                break;
            result.received += **chunk;
        }
        co_return result;
    }());

    CHECK(outcome.open_ok);                    // 头已到，open 成功
    CHECK(outcome.received == "partial-10");   // 断前字节完整交付
    CHECK(outcome.got_error);                  // 断流 = 错误终结，不是 EOF
    CHECK(server.request_count() == 1);        // 首字节后绝不重发
    CHECK(server.errors().empty());
}

TEST_CASE("连接失败返回 NetworkError")
{
    // 拿一个刚释放的端口（server 析构后连接必被拒）
    unsigned short dead_port = 0;
    {
        MockServer temporary_server;
        dead_port = temporary_server.port();
    }

    HttpRequestOptions options;
    options.connect_timeout_ms = 2000;
    options.max_retries = 0;
    Result<HttpResponse> response = run_awaitable([&]() -> asio::awaitable<Result<HttpResponse>> {
        co_return co_await agent::async_http_post(
            co_await asio::this_coro::executor,
            "http://127.0.0.1:" + std::to_string(dead_port) + "/nowhere",
            nlohmann::json::object(), std::move(options));
    }());

    REQUIRE_FALSE(response.has_value());
    CHECK(response.error().code == agent::Errc::NetworkError);
}

TEST_CASE("自定义请求头到达服务器")
{
    MockServer server;
    MockServer::Exchange exchange = plain_response(200, "OK", "application/json", "{}");
    exchange.expect = [&server](RequestView const& request) {
        if (request.header("authorization") != "Bearer test-key")
            server.record_error("Authorization 头缺失或错误");
        if (request.header("x-custom") != "custom-value")
            server.record_error("x-custom 头缺失或错误");
    };
    server.enqueue(std::move(exchange));

    HttpRequestOptions options;
    options.headers = {{"Authorization", "Bearer test-key"}, {"x-custom", "custom-value"}};
    Result<HttpResponse> response = run_awaitable([&]() -> asio::awaitable<Result<HttpResponse>> {
        co_return co_await agent::async_http_post(
            co_await asio::this_coro::executor, server.base_url() + "/headers",
            nlohmann::json::object(), std::move(options));
    }());

    REQUIRE(response.has_value());
    CHECK(server.errors().empty());
}

TEST_CASE("idle 超时：流中静默超限报错")
{
    MockServer server;
    std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        "Content-Length: 20\r\nConnection: close\r\n\r\n";
    // 头 + 5 字节后停 3 秒（超 idle 上限），剩余永不到达
    server.enqueue(MockServer::Exchange{
        .expect = nullptr,
        .chunks = {{head + "early", 3000}, {"never-arrives!!", 0}},
        .close_abruptly = false});

    HttpRequestOptions options;
    options.idle_timeout_ms = 1000;   // LOW_SPEED 秒粒度：1 秒
    options.max_retries = 0;

    struct IdleOutcome {
        bool open_ok = false;
        bool got_error = false;
    };
    IdleOutcome outcome = run_awaitable([&]() -> asio::awaitable<IdleOutcome> {
        IdleOutcome result;
        Result<HttpStreamReader> reader = co_await HttpStreamReader::open(
            co_await asio::this_coro::executor, server.base_url() + "/idle",
            nlohmann::json::object(), std::move(options));
        if (!reader)
            co_return result;
        result.open_ok = true;
        while (true) {
            Result<std::optional<std::string>> chunk = co_await reader->next_chunk();
            if (!chunk) {
                result.got_error = true;
                break;
            }
            if (!chunk->has_value())
                break;
        }
        co_return result;
    }());

    CHECK(outcome.open_ok);
    CHECK(outcome.got_error);   // idle 超时 → 错误终结（而非无限等待）
}

TEST_CASE("取消信号中断流式读取")
{
    MockServer server;
    std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        "Content-Length: 50\r\nConnection: close\r\n\r\n";
    // 头 + 少量数据后长时间停顿——取消必须能在停顿中打断等待
    server.enqueue(MockServer::Exchange{
        .expect = nullptr,
        .chunks = {{head + "before-cancel", 5000}, {"rest", 0}},
        .close_abruptly = false});

    asio::cancellation_signal cancel_signal;
    HttpRequestOptions options;
    options.cancel = cancel_signal.slot();
    options.max_retries = 0;

    struct CancelOutcome {
        bool got_error = false;
        std::string error_message;
        std::chrono::steady_clock::duration elapsed{};
    };
    CancelOutcome outcome = run_awaitable([&]() -> asio::awaitable<CancelOutcome> {
        asio::any_io_executor executor = co_await asio::this_coro::executor;

        // 300ms 后从旁路协程发取消
        asio::co_spawn(executor, [&cancel_signal]() -> asio::awaitable<void> {
            asio::steady_timer timer{co_await asio::this_coro::executor};
            timer.expires_after(std::chrono::milliseconds(300));
            co_await timer.async_wait(asio::as_tuple(asio::use_awaitable));
            cancel_signal.emit(asio::cancellation_type::terminal);
        }, asio::detached);

        CancelOutcome result;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        Result<HttpStreamReader> reader = co_await HttpStreamReader::open(
            executor, server.base_url() + "/cancel",
            nlohmann::json::object(), std::move(options));
        if (!reader) {
            result.got_error = true;
            result.error_message = reader.error().message;
            result.elapsed = std::chrono::steady_clock::now() - start;
            co_return result;
        }
        while (true) {
            Result<std::optional<std::string>> chunk = co_await reader->next_chunk();
            if (!chunk) {
                result.got_error = true;
                result.error_message = chunk.error().message;
                break;
            }
            if (!chunk->has_value())
                break;
        }
        result.elapsed = std::chrono::steady_clock::now() - start;
        co_return result;
    }());

    CHECK(outcome.got_error);
    CHECK(outcome.error_message == "request cancelled");
    // 取消必须及时生效（远早于 5 秒的剧本停顿）
    CHECK(outcome.elapsed < std::chrono::seconds(3));
}

TEST_CASE("gzip 响应自动解压")
{
    std::string const plain_body =
        "{\"choices\":[{\"delta\":{\"content\":\"你好，世界\"}}]}";
    std::string const gzip_body = gzip_compress(plain_body);

    MockServer server;
    MockServer::Exchange exchange =
        plain_response(200, "OK", "application/json", gzip_body,
                       "Content-Encoding: gzip\r\n");
    exchange.expect = [&server](RequestView const& request) {
        // 客户端必须声明支持的编码（否则服务器不压缩，但我们要验证声明）
        if (request.header("accept-encoding").empty())
            server.record_error("Accept-Encoding 头缺失");
    };
    server.enqueue(std::move(exchange));

    Result<HttpResponse> response = run_awaitable([&]() -> asio::awaitable<Result<HttpResponse>> {
        co_return co_await agent::async_http_post(
            co_await asio::this_coro::executor, server.base_url() + "/gzip",
            nlohmann::json::object(), HttpRequestOptions{});
    }());

    REQUIRE(response.has_value());
    CHECK(response->status == 200);
    CHECK(response->body == plain_body);   // 压缩字节已被 curl 透明解压
    CHECK(server.errors().empty());
}

TEST_CASE("deflate 响应自动解压")
{
    std::string const plain_body = "deflate-encoded-response-body";
    std::string const zlib_body = zlib_compress(plain_body);

    MockServer server;
    server.enqueue(plain_response(200, "OK", "text/plain", zlib_body,
                                  "Content-Encoding: deflate\r\n"));

    Result<HttpResponse> response = run_awaitable([&]() -> asio::awaitable<Result<HttpResponse>> {
        co_return co_await agent::async_http_post(
            co_await asio::this_coro::executor, server.base_url() + "/deflate",
            nlohmann::json::object(), HttpRequestOptions{});
    }());

    REQUIRE(response.has_value());
    CHECK(response->body == plain_body);
    CHECK(server.errors().empty());
}

TEST_CASE("压缩流式响应逐块解压")
{
    std::string const plain_body =
        "data: {\"x\":1}\n\n"
        "data: {\"y\":2}\n\n"
        "data: {\"z\":3}\n\n";
    std::string const gzip_body = gzip_compress(plain_body);

    MockServer server;
    std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        "Content-Encoding: gzip\r\n"
        "Content-Length: " + std::to_string(gzip_body.size()) + "\r\n"
        "Connection: close\r\n\r\n";
    // 压缩字节分两包送达——解压器必须跨包工作
    std::size_t split = gzip_body.size() / 2;
    server.enqueue(MockServer::Exchange{
        .expect = nullptr,
        .chunks = {{head + gzip_body.substr(0, split), 0},
                   {gzip_body.substr(split), 0}},
        .close_abruptly = false});

    struct CompressedStreamOutcome {
        bool open_ok = false;
        std::string collected;
        bool clean_eof = false;
    };
    CompressedStreamOutcome outcome = run_awaitable(
        [&]() -> asio::awaitable<CompressedStreamOutcome> {
            CompressedStreamOutcome result;
            Result<HttpStreamReader> reader = co_await HttpStreamReader::open(
                co_await asio::this_coro::executor, server.base_url() + "/gzip-stream",
                nlohmann::json::object(), HttpRequestOptions{});
            if (!reader)
                co_return result;
            result.open_ok = true;
            while (true) {
                Result<std::optional<std::string>> chunk = co_await reader->next_chunk();
                if (!chunk)
                    break;
                if (!chunk->has_value()) {
                    result.clean_eof = true;
                    break;
                }
                result.collected += **chunk;
            }
            co_return result;
        }());

    CHECK(outcome.open_ok);
    CHECK(outcome.collected == plain_body);
    CHECK(outcome.clean_eof);
    CHECK(server.errors().empty());
}

TEST_CASE("brotli 响应自动解压")
{
    std::string const plain_body = "brotli-encoded-response-body-你好";
    std::string const br_body = brotli_compress(plain_body);

    MockServer server;
    server.enqueue(plain_response(200, "OK", "text/plain", br_body,
                                  "Content-Encoding: br\r\n"));

    Result<HttpResponse> response = run_awaitable([&]() -> asio::awaitable<Result<HttpResponse>> {
        co_return co_await agent::async_http_post(
            co_await asio::this_coro::executor, server.base_url() + "/br",
            nlohmann::json::object(), HttpRequestOptions{});
    }());

    REQUIRE(response.has_value());
    CHECK(response->status == 200);
    CHECK(response->body == plain_body);   // br 字节已被 curl 透明解压
    CHECK(server.errors().empty());
}

TEST_CASE("并发压力：20 线程 × 25 请求全部成功且数据正确")
{
    // 验证传输层无共享可变状态：多线程并发请求同一 server，全部成功、响应不串扰
    constexpr int kThreads = 20;
    constexpr int kPerThread = 25;
    constexpr int kTotal = kThreads * kPerThread;

    MockServer server;
    for (int i = 0; i < kTotal; ++i)
        server.enqueue(plain_response(200, "OK", "text/plain", "ok-" + std::to_string(i)));

    std::atomic<int> succeeded{0};
    std::atomic<int> mismatched{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            asio::io_context io;
            for (int i = 0; i < kPerThread; ++i) {
                // 每请求独立 io_context（单线程模型），跨线程并发无共享状态
                Result<HttpResponse> response;
                asio::co_spawn(io, [&]() -> asio::awaitable<void> {
                    response = co_await agent::async_http_post(
                        co_await asio::this_coro::executor, server.base_url() + "/concurrent",
                        nlohmann::json::object(), HttpRequestOptions{});
                }, asio::detached);
                io.restart();
                io.run();
                if (response && response->status == 200)
                    ++succeeded;
                else
                    ++mismatched;
            }
        });
    }
    for (auto& th : threads)
        th.join();

    CHECK(succeeded == kTotal);
    CHECK(mismatched == 0);
    CHECK(server.request_count() == kTotal);
    CHECK(server.errors().empty());
}
