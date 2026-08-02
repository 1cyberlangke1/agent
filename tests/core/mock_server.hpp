#pragma once

// 单元测试进程内的本地 HTTP mock server。
//
// - asio acceptor 绑定 127.0.0.1:0（系统分配临时端口），base_url() 交给被测方
// - 跑在自己的后台线程（同步 asio 读写）——与被测客户端的 io_context 完全
//   隔离，才能测出真实的跨端分包/延迟行为
// - 响应完全字节级控制：状态行也是剧本（Chunk）的一部分，可构造畸形响应
// - 请求断言（Exchange::expect）在 server 线程执行，不得调用 doctest 宏
//   （非线程安全）——失败信息 push 进 errors()，测试主线程末尾统一
//   CHECK(errors().empty())
// - 每连接处理一个请求后关闭（剧本响应应带 Connection: close，
//   客户端不会复用连接）
//
// 测试代码豁免 std::function 禁令（AGENTS.md 代码风格）。

#include <asio.hpp>

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace agent::test {

/// server 线程解析出的请求（拷贝件，供 expect 断言）。
class RequestView {
public:
    std::string method;
    std::string target;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;

    /// 头取值（大小写不敏感）；不存在返回空。
    std::string_view header(std::string_view name) const
    {
        auto lower_eq = [](std::string_view a, std::string_view b) {
            if (a.size() != b.size()) return false;
            for (std::size_t i = 0; i < a.size(); ++i) {
                char x = a[i], y = b[i];
                if (x >= 'A' && x <= 'Z') x = static_cast<char>(x - 'A' + 'a');
                if (y >= 'A' && y <= 'Z') y = static_cast<char>(y - 'A' + 'a');
                if (x != y) return false;
            }
            return true;
        };
        for (auto const& [header_name, value] : headers)
            if (lower_eq(header_name, name))
                return value;
        return {};
    }
};

class MockServer {
public:
    /// 响应分片：bytes 原样写入 socket，写完后停顿 delay_ms。
    /// 分片 + 延迟是构造一切流式难样例的原语。
    struct Chunk {
        std::string bytes;
        int delay_ms = 0;
    };

    /// 一次 HTTP 交换的剧本。
    struct Exchange {
        std::function<void(RequestView const&)> expect;   // 可空
        std::vector<Chunk> chunks;      // 状态行+头+body 全部自己写
        bool close_abruptly = false;    // 发完后 RST（SO_LINGER=0）而非优雅关闭
    };

    MockServer()
        : acceptor_(io_, asio::ip::tcp::endpoint(asio::ip::make_address("127.0.0.1"), 0))
    {
        port_ = acceptor_.local_endpoint().port();
        server_thread_ = std::thread([this] { accept_loop(); });
    }

    ~MockServer()
    {
        stopping_ = true;
        asio::error_code ignored;
        acceptor_.close(ignored);
        if (server_thread_.joinable())
            server_thread_.join();
    }

    MockServer(MockServer const&) = delete;
    MockServer& operator=(MockServer const&) = delete;

    /// 按顺序响应后续请求。
    void enqueue(Exchange exchange)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.push_back(std::move(exchange));
    }

    std::string base_url() const
    {
        return "http://127.0.0.1:" + std::to_string(port_);
    }

    unsigned short port() const { return port_; }

    /// 已收到的请求数（重试断言用）。
    int request_count() const { return request_count_.load(); }

    /// server 线程收集的断言失败（主线程末尾统一 CHECK 为空）。
    std::vector<std::string> errors() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return errors_;
    }

private:
    void accept_loop()
    {
        while (!stopping_) {
            asio::ip::tcp::socket socket(io_);
            asio::error_code ec;
            acceptor_.accept(socket, ec);
            if (ec)
                return;   // acceptor 关闭 = 停止
            handle_connection(socket);
        }
    }

    void handle_connection(asio::ip::tcp::socket& socket)
    {
        std::optional<RequestView> request = read_request(socket);
        if (!request.has_value())
            return;   // 连接被对端放弃（如取消测试），不消耗剧本
        request_count_.fetch_add(1);

        Exchange exchange;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (pending_.empty()) {
                errors_.push_back("unexpected request (no scripted exchange): "
                                  + request->method + " " + request->target);
                return;
            }
            exchange = std::move(pending_.front());
            pending_.pop_front();
        }

        if (exchange.expect) {
            exchange.expect(*request);   // 内部失败经 record_error 报告
        }

        for (Chunk const& chunk : exchange.chunks) {
            asio::error_code write_ec;
            asio::write(socket, asio::buffer(chunk.bytes), write_ec);
            if (write_ec)
                return;   // 对端提前断开（取消/超时测试的正常路径）
            if (chunk.delay_ms > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(chunk.delay_ms));
        }

        asio::error_code ignored;
        if (exchange.close_abruptly) {
            // SO_LINGER=0：close 发 RST 而非 FIN，模拟粗暴断连
            socket.set_option(asio::socket_base::linger(true, 0), ignored);
        } else {
            socket.shutdown(asio::ip::tcp::socket::shutdown_both, ignored);
        }
        socket.close(ignored);
    }

    /// 阻塞读一个完整请求：头至 \r\n\r\n，body 按 Content-Length。
    /// 简化解析（测试工具，客户端是规范的 curl）：不支持请求侧 chunked。
    std::optional<RequestView> read_request(asio::ip::tcp::socket& socket)
    {
        std::string data;
        asio::error_code ec;
        // 读到头结束
        while (data.find("\r\n\r\n") == std::string::npos) {
            char buffer[4096];
            std::size_t n = socket.read_some(asio::buffer(buffer), ec);
            if (ec)
                return std::nullopt;
            data.append(buffer, n);
            if (data.size() > (std::size_t{8} << 20))
                return std::nullopt;   // 防失控
        }
        std::size_t header_end = data.find("\r\n\r\n");
        std::string head = data.substr(0, header_end);
        std::string body = data.substr(header_end + 4);

        RequestView request;
        std::size_t line_start = 0;
        bool first_line = true;
        while (line_start <= head.size()) {
            std::size_t line_end = head.find("\r\n", line_start);
            std::string line = head.substr(line_start,
                line_end == std::string::npos ? std::string::npos : line_end - line_start);
            if (first_line) {
                std::size_t space1 = line.find(' ');
                std::size_t space2 = line.find(' ', space1 + 1);
                if (space1 != std::string::npos && space2 != std::string::npos) {
                    request.method = line.substr(0, space1);
                    request.target = line.substr(space1 + 1, space2 - space1 - 1);
                }
                first_line = false;
            } else if (!line.empty()) {
                std::size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    std::string name = line.substr(0, colon);
                    std::string value = line.substr(colon + 1);
                    while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
                        value.erase(value.begin());
                    request.headers.emplace_back(std::move(name), std::move(value));
                }
            }
            if (line_end == std::string::npos)
                break;
            line_start = line_end + 2;
        }

        // 按 Content-Length 补齐 body
        std::string_view length_text = request.header("content-length");
        std::size_t content_length = 0;
        for (char c : length_text) {
            if (c < '0' || c > '9') break;
            content_length = content_length * 10 + static_cast<std::size_t>(c - '0');
        }
        while (body.size() < content_length) {
            char buffer[4096];
            std::size_t n = socket.read_some(asio::buffer(buffer), ec);
            if (ec)
                return std::nullopt;
            body.append(buffer, n);
        }
        request.body = std::move(body);
        return request;
    }

public:
    /// expect 回调内报告断言失败（doctest 宏非线程安全，不可在 server 线程用）。
    void record_error(std::string message)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        errors_.push_back(std::move(message));
    }

private:
    asio::io_context io_;
    asio::ip::tcp::acceptor acceptor_;
    unsigned short port_ = 0;
    std::thread server_thread_;
    std::atomic<bool> stopping_{false};
    std::atomic<int> request_count_{0};
    mutable std::mutex mutex_;
    std::deque<Exchange> pending_;
    std::vector<std::string> errors_;
};

}  // namespace agent::test
