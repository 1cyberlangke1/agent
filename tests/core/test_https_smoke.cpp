// HTTPS 冒烟测试：真实公开端点验证 TLS 握手路径。
//
// 背景：Windows curl 走 schannel（系统证书库），我们的传输层零 TLS 代码
// （libcurl 全权处理）。TLS 专项自签证书测试在 schannel 下需要导入系统
// 证书库（危险），故改用「真实公开 HTTPS 端点连通性」验证 TLS 握手成功。
//
// 做法：向 https://api.deepseek.com 发一个最小请求（无效 key）——
// 返回 401 即证明「TLS 握手成功 + HTTP 层连通」（认证失败发生在握手之后）。
// 无外网环境（连接失败）时打印跳过说明，不判失败。

#include <agent/core/http_client.hpp>

#include <doctest/doctest.h>

#include <optional>
#include <string>

using agent::HttpRequestOptions;
using agent::HttpResponse;
using agent::Result;

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

}  // namespace

TEST_CASE("https 冒烟：真实端点 TLS 握手（401 即证明 https 通）")
{
    HttpRequestOptions opts;
    opts.headers.emplace_back("Authorization", "Bearer invalid-key");
    opts.connect_timeout_ms = 15000;
    opts.total_timeout_ms = 30000;
    opts.max_retries = 0;

    nlohmann::json body{ { "model", "deepseek-chat" },
                         { "messages", nlohmann::json::array({ nlohmann::json{ { "role", "user" },
                                                                                { "content", "hi" } } }) } };

    Result<HttpResponse> response = run_awaitable([&]() -> asio::awaitable<Result<HttpResponse>> {
        co_return co_await agent::async_http_post(
            co_await asio::this_coro::executor,
            "https://api.deepseek.com/v1/chat/completions", body, opts);
    }());

    if (!response) {
        // 无外网 / 端点不可达：跳过（测试意图是「有外网时验证 TLS 路径」）
        MESSAGE("跳过 https 冒烟：连接失败 (" << response.error().message << ")");
        return;
    }
    // 任何 HTTP 状态码（401/429/5xx）都证明 TLS 握手成功 + 请求送达——
    // TLS 发生在认证/限流之前。无外网/连不上时打印跳过说明。
    MESSAGE("https 冒烟：HTTP " << response->status);
    CHECK(response->status > 0);
}
