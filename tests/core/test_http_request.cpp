// 完整 curl 封装（PLAN_4.md）测试：方法 / body 形态（json/form/multipart/流式上传）/
// 查询参数 / 认证 / 重定向 / cookie jar / 错误归一化 / HttpClient 连接复用 / 响应便捷。
//
// 全部对着进程内 MockServer（127.0.0.1 明文 http）；TLS 路径不在此测
//（schannel 对 mock 需证书部署；verify_tls 开关仅测 option 映射不实际握手）。

#include <agent/core/http_client.hpp>

#include "mock_server.hpp"

#include <doctest/doctest.h>

#include <fstream>
#include <optional>
#include <string>
#include <utility>

using agent::HttpAuth;
using agent::HttpClient;
using agent::HttpProxyType;
using agent::HttpRequest;
using agent::HttpRequestOptions;
using agent::HttpResponse;
using agent::MultipartPart;
using agent::Result;
using agent::test::MockServer;
using agent::test::RequestView;

namespace {

/// 在独立 io_context 上同步跑一个 awaitable（单个协程内可串多个 HttpClient 请求）。
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

/// void 版本（co_spawn 完成回调不携带值）。
void run_awaitable_void(asio::awaitable<void> coroutine_task)
{
    asio::io_context io;
    bool done = false;
    asio::co_spawn(io, std::move(coroutine_task),
        [&done](std::exception_ptr) { done = true; });
    io.run();
    REQUIRE(done);
}

/// 规范响应（Content-Length + Connection: close）。
MockServer::Exchange plain_response(int status_code, std::string const& reason,
                                    std::string const& body,
                                    std::string const& extra_headers = {})
{
    std::string head = "HTTP/1.1 " + std::to_string(status_code) + " " + reason + "\r\n"
        + "Content-Type: application/json\r\n"
        + "Content-Length: " + std::to_string(body.size()) + "\r\n"
        + extra_headers
        + "Connection: close\r\n\r\n";
    return MockServer::Exchange{.expect = nullptr,
                                .chunks = {{head + body, 0}},
                                .close_abruptly = false};
}

/// keep-alive 响应（不带 Connection: close，连接复用测试用）。
MockServer::Exchange keepalive_json(std::string const& body)
{
    std::string head = std::string("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n")
        + "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n";
    return MockServer::Exchange{.expect = nullptr,
                                .chunks = {{head + body, 0}},
                                .close_abruptly = false,
                                .close_after = false};
}

/// 请求侧断言用的 expect 助手：断言 method 与 body（失败经 server.record_error 报告）。
MockServer::Exchange expect_method_body(MockServer& server, std::string const& method,
                                        std::string const& body, std::string const& response_body)
{
    MockServer::Exchange exchange = plain_response(200, "OK", response_body);
    exchange.expect = [&server, method, body](RequestView const& request) {
        if (request.method != method)
            server.record_error("method != " + method + ": " + request.method);
        if (request.body != body)
            server.record_error("body != " + body + ": " + request.body);
    };
    return exchange;
}

/// 简易临时文件（multipart / 上传测试用）。
std::string make_temp_file(std::string const& content)
{
    static int counter = 0;
    std::string path = std::string("build/tests/tmp_upload_") + std::to_string(counter++) + ".txt";
    std::ofstream file(path);
    file << content;
    file.close();
    return path;
}

}  // namespace

// ───────────────────── 方法 ─────────────────────

TEST_CASE("方法：PUT/PATCH/DELETE 带 body + 自定义方法")
{
    MockServer server;
    // PUT
    server.enqueue(expect_method_body(server, "PUT", "put-payload", R"({"ok":"put"})"));
    HttpRequest put;
    put.method = "PUT";
    put.url = server.base_url() + "/res";
    put.body = "put-payload";
    put.content_type = "text/plain";
    Result<HttpResponse> r_put = agent::sync_http_request(put, {});
    REQUIRE(r_put.has_value());
    CHECK(r_put->status == 200);

    // PATCH
    server.enqueue(expect_method_body(server, "PATCH", "patch-payload", R"({"ok":"patch"})"));
    HttpRequest patch;
    patch.method = "PATCH";
    patch.url = server.base_url() + "/res";
    patch.body = "patch-payload";
    patch.content_type = "text/plain";
    Result<HttpResponse> r_patch = agent::sync_http_request(patch, {});
    REQUIRE(r_patch.has_value());
    CHECK(r_patch->status == 200);

    // DELETE（无 body）
    server.enqueue(expect_method_body(server, "DELETE", "", R"({"ok":"delete"})"));
    HttpRequest del;
    del.method = "DELETE";
    del.url = server.base_url() + "/res";
    Result<HttpResponse> r_del = agent::sync_http_request(del, {});
    REQUIRE(r_del.has_value());
    CHECK(r_del->status == 200);

    // 自定义方法（CUSTOMREQUEST）
    server.enqueue(expect_method_body(server, "PROPFIND", "prop-payload", R"({"ok":"prop"})"));
    HttpRequest prop;
    prop.method = "PROPFIND";
    prop.url = server.base_url() + "/res";
    prop.body = "prop-payload";
    prop.content_type = "text/plain";
    Result<HttpResponse> r_prop = agent::sync_http_request(prop, {});
    REQUIRE(r_prop.has_value());
    CHECK(r_prop->status == 200);

    CHECK(server.errors().empty());
}

TEST_CASE("方法：HEAD（CURLOPT_NOBODY，无响应体）")
{
    MockServer server;
    MockServer::Exchange exchange{
        .expect = nullptr,
        .chunks = {{"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                    "Content-Length: 5\r\nConnection: close\r\n\r\n", 0}},
        .close_abruptly = false,
    };
    exchange.expect = [&server](RequestView const& request) {
        if (request.method != "HEAD")
            server.record_error("method != HEAD: " + request.method);
    };
    server.enqueue(std::move(exchange));

    HttpRequest req;
    req.method = "HEAD";
    req.url = server.base_url() + "/head";
    Result<HttpResponse> response = agent::sync_http_request(req, {});
    REQUIRE(response.has_value());
    CHECK(response->status == 200);
    CHECK(response->body.empty());   // NOBODY 无响应体
    CHECK(server.errors().empty());
}

// ───────────────────── body 形态 ─────────────────────

TEST_CASE("body：form-urlencoded 自动编码")
{
    MockServer server;
    MockServer::Exchange exchange = plain_response(200, "OK", R"({"ok":true})");
    exchange.expect = [&server](RequestView const& request) {
        if (request.header("content-type") != "application/x-www-form-urlencoded")
            server.record_error("content-type 错误");
        std::string expected = "city=%E6%9D%AD%E5%B7%9E&q=hello%20world";
        if (request.body != expected)
            server.record_error("form body 错误: " + request.body);
    };
    server.enqueue(std::move(exchange));

    Result<nlohmann::json> result = agent::http_post_form(
        server.base_url() + "/form", {{"city", "杭州"}, {"q", "hello world"}}, {});
    REQUIRE(result.has_value());
    CHECK((*result)["ok"] == true);
    CHECK(server.errors().empty());
}

TEST_CASE("body：multipart（值 + 内存 + 文件 part）")
{
    std::string const file_content = "file-content-123";
    std::string const file_path = make_temp_file(file_content);

    MockServer server;
    MockServer::Exchange exchange = plain_response(200, "OK", R"({"ok":true})");
    exchange.expect = [&](RequestView const& request) {
        if (request.header("content-type").find("multipart/form-data; boundary=") == std::string::npos)
            server.record_error("content-type 不是 multipart");
        if (request.body.find("name=\"field\"") == std::string::npos)
            server.record_error("缺 field part");
        if (request.body.find("field-value") == std::string::npos)
            server.record_error("field part 值缺失");
        if (request.body.find("name=\"data\"") == std::string::npos)
            server.record_error("缺 data part");
        if (request.body.find("buffer-bytes") == std::string::npos)
            server.record_error("data part 值缺失");
        if (request.body.find("name=\"file\"") == std::string::npos
            || request.body.find("filename=\"a.txt\"") == std::string::npos)
            server.record_error("缺 file part / filename");
        if (request.body.find(file_content) == std::string::npos)
            server.record_error("file part 内容缺失");
    };
    server.enqueue(std::move(exchange));

    HttpRequest request;
    request.method = "POST";
    request.url = server.base_url() + "/upload";
    request.multipart = {
        MultipartPart{ .name = "field", .value = "field-value" },
        MultipartPart{ .name = "data", .buffer = "buffer-bytes" },
        MultipartPart{ .name = "file", .file_path = file_path,
                       .content_type = "text/plain", .filename = "a.txt" },
    };
    Result<HttpResponse> response = agent::sync_http_request(request, {});
    REQUIRE(response.has_value());
    CHECK(response->status == 200);
    CHECK(server.errors().empty());
    std::remove(file_path.c_str());
}

TEST_CASE("body：http_upload_file（multipart 单文件 part）")
{
    std::string const file_content = "upload-body-content";
    std::string const file_path = make_temp_file(file_content);

    MockServer server;
    MockServer::Exchange exchange = plain_response(200, "OK", R"({"ok":true})");
    exchange.expect = [&](RequestView const& request) {
        if (request.header("content-type").find("multipart/form-data") == std::string::npos)
            server.record_error("content-type 不是 multipart");
        if (request.body.find("name=\"file\"") == std::string::npos)
            server.record_error("缺 file part");
        if (request.body.find(file_content) == std::string::npos)
            server.record_error("file 内容缺失");
    };
    server.enqueue(std::move(exchange));

    Result<void> result = agent::http_upload_file(server.base_url() + "/up", file_path, {});
    REQUIRE(result.has_value());
    CHECK(server.errors().empty());
    std::remove(file_path.c_str());
}

// ───────────────────── 查询参数 / 认证 / 重定向 ─────────────────────

TEST_CASE("查询参数 append_query 自动编码")
{
    MockServer server;
    MockServer::Exchange exchange = plain_response(200, "OK", R"({"ok":true})");
    exchange.expect = [&server](RequestView const& request) {
        std::string expected = "/search?city=%E6%9D%AD%E5%B7%9E&q=a%26b";
        if (request.target != expected)
            server.record_error("target 错误: " + request.target);
    };
    server.enqueue(std::move(exchange));

    HttpRequest request;
    request.method = "GET";
    request.url = server.base_url() + "/search";
    request.query = {{"city", "杭州"}, {"q", "a&b"}};
    Result<HttpResponse> response = agent::sync_http_request(request, {});
    REQUIRE(response.has_value());
    CHECK(server.errors().empty());
}

TEST_CASE("认证：Basic / Bearer")
{
    MockServer server;
    // Basic
    MockServer::Exchange basic_exchange = plain_response(200, "OK", R"({"ok":true})");
    basic_exchange.expect = [&server](RequestView const& request) {
        if (request.header("authorization") != "Basic dXNlcjpwYXNz")
            server.record_error("Basic 头错误: " + std::string(request.header("authorization")));
    };
    server.enqueue(std::move(basic_exchange));
    HttpRequest basic;
    basic.url = server.base_url() + "/basic";
    basic.auth = HttpAuth::Basic;
    basic.auth_credentials = "user:pass";
    Result<HttpResponse> r_basic = agent::sync_http_request(basic, {});
    REQUIRE(r_basic.has_value());

    // Bearer
    MockServer::Exchange bearer_exchange = plain_response(200, "OK", R"({"ok":true})");
    bearer_exchange.expect = [&server](RequestView const& request) {
        if (request.header("authorization") != "Bearer token123")
            server.record_error("Bearer 头错误");
    };
    server.enqueue(std::move(bearer_exchange));
    HttpRequest bearer;
    bearer.url = server.base_url() + "/bearer";
    bearer.auth = HttpAuth::Bearer;
    bearer.auth_credentials = "token123";
    Result<HttpResponse> r_bearer = agent::sync_http_request(bearer, {});
    REQUIRE(r_bearer.has_value());

    CHECK(server.errors().empty());
}

TEST_CASE("重定向：默认不跟随；follow 后走最终 + effective_url")
{
    // 不跟随
    {
        MockServer server;
        server.enqueue(plain_response(302, "Found", "", "Location: /final\r\n"));
        HttpRequest request;
        request.url = server.base_url() + "/start";
        Result<HttpResponse> response = agent::sync_http_request(request, {});
        REQUIRE(response.has_value());
        CHECK(response->status == 302);
        CHECK(server.request_count() == 1);
    }
    // 跟随
    {
        MockServer server;
        server.enqueue(plain_response(302, "Found", "", "Location: /final\r\n"));
        server.enqueue(plain_response(200, "OK", R"({"final":true})"));
        HttpRequest request;
        request.url = server.base_url() + "/start";
        request.follow_redirects = true;
        Result<HttpResponse> response = agent::sync_http_request(request, {});
        REQUIRE(response.has_value());
        CHECK(response->status == 200);
        CHECK(response->body == R"({"final":true})");
        CHECK(response->effective_url == server.base_url() + "/final");
        CHECK(server.request_count() == 2);
        CHECK(server.errors().empty());
    }
}

// ───────────────────── 便捷层错误归一化 ─────────────────────

TEST_CASE("便捷层错误归一化：429→RateLimited / 401→AuthError / 500→ProviderError / 非JSON→ParseError")
{
    // 429
    {
        MockServer server;
        server.enqueue(plain_response(429, "Too Many Requests", R"({"error":"slow"})"));
        HttpRequestOptions options;
        options.max_retries = 0;   // 禁重试：让 429 响应直接返回给归一化层
        Result<nlohmann::json> r = agent::http_get_json(server.base_url() + "/a", options);
        REQUIRE_FALSE(r.has_value());
        CHECK(r.error().code == agent::Errc::RateLimited);
    }
    // 401
    {
        MockServer server;
        server.enqueue(plain_response(401, "Unauthorized", R"({"error":"bad key"})"));
        Result<nlohmann::json> r = agent::http_get_json(server.base_url() + "/b", {});
        REQUIRE_FALSE(r.has_value());
        CHECK(r.error().code == agent::Errc::AuthError);
    }
    // 500
    {
        MockServer server;
        server.enqueue(plain_response(500, "Server Error", R"({"error":"boom"})"));
        HttpRequestOptions options;
        options.max_retries = 0;
        Result<nlohmann::json> r = agent::http_get_json(server.base_url() + "/c", options);
        REQUIRE_FALSE(r.has_value());
        CHECK(r.error().code == agent::Errc::ProviderError);
    }
    // 200 但 body 非 JSON → ParseError
    {
        MockServer server;
        server.enqueue(plain_response(200, "OK", "not-json-at-all"));
        Result<nlohmann::json> r = agent::http_get_json(server.base_url() + "/d", {});
        REQUIRE_FALSE(r.has_value());
        CHECK(r.error().code == agent::Errc::ParseError);
    }
}

// ───────────────────── HttpResponse 便捷 ─────────────────────

TEST_CASE("HttpResponse 便捷：ok()/header()/json()")
{
    MockServer server;
    MockServer::Exchange exchange = plain_response(200, "OK", R"({"a":1,"b":[2,3]})",
                                                    "X-Test: hello\r\n");
    server.enqueue(std::move(exchange));

    HttpRequest request;
    request.url = server.base_url() + "/meta";
    Result<HttpResponse> response = agent::sync_http_request(request, {});
    REQUIRE(response.has_value());
    CHECK(response->ok());
    CHECK(response->header("x-test") == "hello");
    Result<nlohmann::json> parsed = response->json();
    REQUIRE(parsed.has_value());
    CHECK((*parsed)["a"] == 1);
    CHECK((*parsed)["b"].size() == 2);
    CHECK(response->total_time_seconds >= 0);
}

// ───────────────────── HttpClient（协程：cookie jar + 连接复用）─────────────────────

TEST_CASE("HttpClient cookie jar：Set-Cookie → 下次请求带上")
{
    MockServer server;
    MockServer::Exchange first = plain_response(200, "OK", R"({"ok":1})",
                                                "Set-Cookie: sid=abc123; Path=/\r\n");
    server.enqueue(std::move(first));
    MockServer::Exchange second = plain_response(200, "OK", R"({"ok":2})");
    second.expect = [&server](RequestView const& request) {
        if (request.header("cookie") != "sid=abc123")
            server.record_error("cookie 未携带: " + std::string(request.header("cookie")));
    };
    server.enqueue(std::move(second));

    HttpClient client;
    bool r1_ok = false;
    bool r2_ok = false;
    run_awaitable_void([&]() -> asio::awaitable<void> {
        HttpRequest req1;
        req1.url = server.base_url() + "/a";
        r1_ok = (co_await client.async_request(req1, {})).has_value();
        HttpRequest req2;
        req2.url = server.base_url() + "/b";
        r2_ok = (co_await client.async_request(req2, {})).has_value();
    }());

    CHECK(r1_ok);
    CHECK(r2_ok);
    CHECK(server.errors().empty());
}

TEST_CASE("HttpClient 连接复用：同 host 两请求走同一 TCP 连接")
{
    MockServer server;
    server.enqueue(keepalive_json(R"({"n":1})"));
    server.enqueue(keepalive_json(R"({"n":2})"));

    HttpClient client;
    run_awaitable_void([&]() -> asio::awaitable<void> {
        HttpRequest req1;
        req1.url = server.base_url() + "/x";
        HttpRequest req2;
        req2.url = server.base_url() + "/y";
        (void)co_await client.async_request(req1, {});
        (void)co_await client.async_request(req2, {});
    }());

    CHECK(server.request_count() == 2);
    CHECK(server.connection_count() == 1);   // 常驻 multi 连接池 → 同一连接
    CHECK(server.errors().empty());
}

TEST_CASE("HttpClient cookie jar load/save")
{
    HttpClient client;
    run_awaitable_void([&]() -> asio::awaitable<void> {
        co_await client.load_cookies("build/tests/cookie_test.txt");
        co_await client.save_cookies("build/tests/cookie_out.txt");
    }());
    // load 不存在的文件 → 返回错误（不崩）
    bool loaded = run_awaitable([&]() -> asio::awaitable<bool> {
        Result<void> r = co_await client.load_cookies("build/tests/nonexistent_cookie.txt");
        co_return r.has_value();
    }());
    CHECK(!loaded);
}
