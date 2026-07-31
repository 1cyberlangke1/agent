// L0 传输层实现：libcurl multi_socket 模式嫁接 asio io_context。
//
// 线程模型（关键约束）：一次传输（CurlSession）的全部状态运行在单一
// executor 上——curl 回调只在 curl_multi_socket_action 内同步执行，而
// socket_action 只被该 executor 上的 asio 完成回调调用。容器无锁。
// 唯一跨线程入口是取消信号（emit 可来自任意线程），经 asio::post 跳转。
// 调用方要求：executor 须为单线程 io_context（或 strand）。
//
// socket 归属：socket 由 asio 创建（CURLOPT_OPENSOCKETFUNCTION），curl 只
// 借用 native handle 做连接与读写；监听（可读/可写）由 asio async_wait
// 驱动，事件到达后回调 curl_multi_socket_action 推进 curl 状态机。
// 关闭（CURLOPT_CLOSESOCKETFUNCTION）时从表中移除，asio socket 析构收尾。
//
// 生命周期：CurlSession 由 shared_ptr 管理——所有挂起的 asio 回调各持
// 一份引用；HttpStreamReader 析构（或重试放弃本次传输）时调 shutdown()
// 拆除 curl 句柄并取消监听，其后到达的回调看到 stopped 标志空转返回，
// 最后一个回调释放引用后 CurlSession 析构。

#include <agent/http_client.hpp>

#include <curl/curl.h>

#include <chrono>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace agent {

namespace {

// ───────────────────────── 常量与小工具 ─────────────────────────

/// 背压水位：body 缓冲超过 high 时暂停 curl 交付（CURL_WRITEFUNC_PAUSE），
/// next_chunk 消费到 low 以下时恢复（curl_easy_pause CONT）。
constexpr std::size_t kHighWaterBytes = std::size_t{1} << 20;   // 1 MiB
constexpr std::size_t kLowWaterBytes = std::size_t{256} << 10;  // 256 KiB

/// curl_global_init 进程级一次（线程安全）。
void ensure_curl_global()
{
    static std::once_flag flag;
    std::call_once(flag, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

/// ASCII 大小写不敏感比较（HTTP 头名匹配用）。
bool equals_ignore_case(std::string_view left, std::string_view right)
{
    if (left.size() != right.size())
        return false;
    for (std::size_t i = 0; i < left.size(); ++i) {
        char a = left[i];
        char b = right[i];
        if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = static_cast<char>(b - 'A' + 'a');
        if (a != b)
            return false;
    }
    return true;
}

// ───────────────────────── 会话状态 ─────────────────────────

/// 单个 socket 的 asio 包装与监听状态。
struct SocketState {
    asio::ip::tcp::socket socket;
    int watch = 0;     // curl 当前要求的监听位（CURL_POLL_IN/OUT 组合）
    int pending = 0;   // 已挂起的 async_wait 位（防同方向重复挂起）
};

/// 一次 HTTP 传输的完整状态（单 executor，见文件头注释）。
struct CurlSession {
    asio::any_io_executor executor;
    asio::steady_timer curl_timer;   // curl 的超时驱动（CURLMOPT_TIMERFUNCTION）
    asio::steady_timer event;        // 等待/唤醒原语：常驻 max，cancel() = 唤醒
    asio::steady_timer idle_timer;   // 流式块间 idle 超时（next_chunk 自管）
    int idle_timeout_ms = 0;         // idle 超时阈值（0 = 不限）
    bool idle_expired = false;       // idle 已到期（下次 wake 返回错误）
    std::weak_ptr<CurlSession> weak_self;   // C 回调内发起 asio 操作时取 shared

    CURLM* multi = nullptr;
    CURL* easy = nullptr;
    curl_slist* header_list = nullptr;
    std::string request_body;                  // CURLOPT_POSTFIELDS 指向的存储
    char error_buffer[CURL_ERROR_SIZE] = {};

    std::map<curl_socket_t, SocketState> sockets;

    // 响应头阶段（header 回调填充）
    long status = 0;
    std::vector<std::pair<std::string, std::string>> response_headers;
    bool headers_done = false;   // 已收到最终响应（非 1xx）的完整头

    // body 缓冲（write 回调生产，next_chunk 消费；同 executor 无锁）
    std::deque<std::string> body_chunks;
    std::size_t buffered_bytes = 0;
    bool paused = false;

    // 终态
    bool finished = false;            // CURLMSG_DONE 已到
    CURLcode curl_result = CURLE_OK;  // finished 时的传输结果
    bool aborted = false;             // 用户取消
    bool stopped = false;             // 会话已拆除，回调空转

    explicit CurlSession(asio::any_io_executor ex)
        : executor(ex), curl_timer(ex), event(ex), idle_timer(ex)
    {
        event.expires_at(std::chrono::steady_clock::time_point::max());
    }

    ~CurlSession() { shutdown(); }

    CurlSession(CurlSession const&) = delete;
    CurlSession& operator=(CurlSession const&) = delete;

    /// 唤醒 open/next_chunk 的等待循环。
    void wake() { event.cancel(); }

    /// 拆除传输（幂等）：取消监听、清理 curl 句柄、唤醒等待者。
    /// 之后所有挂起回调看到 stopped 直接返回。
    void shutdown()
    {
        if (stopped)
            return;
        stopped = true;
        curl_timer.cancel();
        idle_timer.cancel();
        for (auto& [fd, state] : sockets) {
            asio::error_code ignored;
            state.socket.cancel(ignored);
        }
        if (multi != nullptr && easy != nullptr)
            curl_multi_remove_handle(multi, easy);
        if (easy != nullptr) {
            curl_easy_cleanup(easy);   // 触发 close_socket_callback → sockets 清空
            easy = nullptr;
        }
        if (multi != nullptr) {
            curl_multi_cleanup(multi);
            multi = nullptr;
        }
        if (header_list != nullptr) {
            curl_slist_free_all(header_list);
            header_list = nullptr;
        }
        wake();
    }
};

// ───────────────────── curl ↔ asio 桥接（前置声明） ─────────────────────

void process_curl_messages(std::shared_ptr<CurlSession> const& session);
void arm_all_sockets(std::shared_ptr<CurlSession> const& session);

/// 某 socket 就绪（或监听被取消）→ 驱动 curl 状态机并重新挂监听。
void on_socket_ready(std::shared_ptr<CurlSession> const& session, curl_socket_t fd,
                     int which, asio::error_code const& ec)
{
    if (session->stopped)
        return;
    auto found = session->sockets.find(fd);
    if (found == session->sockets.end())
        return;                                     // 连接已被 curl 关闭
    found->second.pending &= ~which;
    if (ec == asio::error::operation_aborted)
        return;                                     // REMOVE/shutdown 取消，无事件

    int bitmask = 0;
    if (which & CURL_POLL_IN)
        bitmask |= CURL_CSELECT_IN;
    if (which & CURL_POLL_OUT)
        bitmask |= CURL_CSELECT_OUT;
    if (ec)
        bitmask |= CURL_CSELECT_ERR;

    int running = 0;
    curl_multi_socket_action(session->multi, fd, bitmask, &running);
    process_curl_messages(session);
    if (!session->stopped)
        arm_all_sockets(session);
}

/// 按各 socket 的 watch 需求补挂缺失方向的 async_wait。
/// 在每次 socket_action 之后调用（action 期间 curl 可能新建连接或改监听需求）。
void arm_all_sockets(std::shared_ptr<CurlSession> const& session)
{
    for (auto& [fd, state] : session->sockets) {
        if ((state.watch & CURL_POLL_IN) && !(state.pending & CURL_POLL_IN)) {
            state.pending |= CURL_POLL_IN;
            state.socket.async_wait(asio::ip::tcp::socket::wait_read,
                [session, fd = fd](asio::error_code const& ec) {
                    on_socket_ready(session, fd, CURL_POLL_IN, ec);
                });
        }
        if ((state.watch & CURL_POLL_OUT) && !(state.pending & CURL_POLL_OUT)) {
            state.pending |= CURL_POLL_OUT;
            state.socket.async_wait(asio::ip::tcp::socket::wait_write,
                [session, fd = fd](asio::error_code const& ec) {
                    on_socket_ready(session, fd, CURL_POLL_OUT, ec);
                });
        }
    }
}

/// 收割 curl 完成消息：CURLMSG_DONE → 置终态并唤醒等待者。
void process_curl_messages(std::shared_ptr<CurlSession> const& session)
{
    if (session->stopped || session->multi == nullptr)
        return;
    int queued = 0;
    while (CURLMsg* message = curl_multi_info_read(session->multi, &queued)) {
        if (message->msg == CURLMSG_DONE) {
            session->finished = true;
            session->curl_result = message->data.result;
            session->wake();
        }
    }
}

// ───────────────────── curl C 回调 ─────────────────────

/// CURLOPT_OPENSOCKETFUNCTION：由 asio 创建 socket，curl 借用 native handle。
curl_socket_t open_socket_callback(void* clientp, curlsocktype purpose,
                                   curl_sockaddr* address)
{
    CurlSession* session = static_cast<CurlSession*>(clientp);
    if (purpose != CURLSOCKTYPE_IPCXN)
        return CURL_SOCKET_BAD;
    if (address->family != AF_INET && address->family != AF_INET6)
        return CURL_SOCKET_BAD;

    asio::ip::tcp::socket socket{session->executor};
    asio::error_code ec;
    socket.open(address->family == AF_INET6 ? asio::ip::tcp::v6()
                                            : asio::ip::tcp::v4(), ec);
    if (ec)
        return CURL_SOCKET_BAD;
    curl_socket_t fd = socket.native_handle();
    session->sockets.emplace(fd, SocketState{std::move(socket)});
    return fd;
}

/// CURLOPT_CLOSESOCKETFUNCTION：curl 要求关闭连接。
/// erase 后 asio socket 析构负责真正 close（挂起的 wait 收 operation_aborted）。
int close_socket_callback(void* clientp, curl_socket_t item)
{
    CurlSession* session = static_cast<CurlSession*>(clientp);
    session->sockets.erase(item);
    return 0;
}

/// CURLMOPT_SOCKETFUNCTION：curl 声明对 fd 监听需求的变化。
/// 只更新 watch / 取消监听——补挂 async_wait 统一由 socket_action 调用点
/// 尾部的 arm_all_sockets 完成（C 回调内拿不到 shared_ptr 捕获）。
int multi_socket_callback(CURL*, curl_socket_t fd, int what, void* userp, void*)
{
    CurlSession* session = static_cast<CurlSession*>(userp);
    auto found = session->sockets.find(fd);
    if (found == session->sockets.end())
        return 0;
    if (what == CURL_POLL_REMOVE) {
        found->second.watch = 0;
        asio::error_code ignored;
        found->second.socket.cancel(ignored);
        return 0;
    }
    found->second.watch = what;   // CURL_POLL_IN/OUT/INOUT 与位掩码语义一致
    return 0;
}

/// CURLMOPT_TIMERFUNCTION：curl 的内部超时调度 → asio steady_timer。
/// timeout_ms == -1 取消；到期后以 CURL_SOCKET_TIMEOUT 驱动状态机。
int multi_timer_callback(CURLM*, long timeout_ms, void* userp)
{
    CurlSession* session = static_cast<CurlSession*>(userp);
    if (timeout_ms < 0) {
        session->curl_timer.cancel();
        return 0;
    }
    std::shared_ptr<CurlSession> self = session->weak_self.lock();
    if (!self)
        return 0;
    session->curl_timer.expires_after(std::chrono::milliseconds(timeout_ms));
    session->curl_timer.async_wait([self](asio::error_code const& ec) {
        if (ec || self->stopped)
            return;
        int running = 0;
        curl_multi_socket_action(self->multi, CURL_SOCKET_TIMEOUT, 0, &running);
        process_curl_messages(self);
        if (!self->stopped)
            arm_all_sockets(self);
    });
    return 0;
}

/// CURLOPT_WRITEFUNCTION：body 字节到达 → 入队并唤醒消费者；超水位则暂停。
/// 返回 PAUSE 时 curl 缓存本块，恢复后重新交付，数据不丢失。
size_t write_callback(char* data, size_t size, size_t nmemb, void* clientp)
{
    CurlSession* session = static_cast<CurlSession*>(clientp);
    if (session->stopped)
        return 0;   // 中止传输（CURLE_WRITE_ERROR）
    size_t total = size * nmemb;
    if (session->buffered_bytes >= kHighWaterBytes) {
        session->paused = true;
        return CURL_WRITEFUNC_PAUSE;
    }
    session->headers_done = true;   // 有 body 必有完整头（保险，正常由空行置位）
    session->body_chunks.emplace_back(data, total);
    session->buffered_bytes += total;
    session->wake();
    return total;
}

/// CURLOPT_HEADERFUNCTION：逐行收头。状态行重置收集（1xx 会有多组），
/// 终止空行且状态 ≥ 200 时置 headers_done（1xx 中间响应被跳过）。
size_t header_callback(char* buffer, size_t size, size_t nitems, void* clientp)
{
    CurlSession* session = static_cast<CurlSession*>(clientp);
    if (session->stopped)
        return 0;
    std::string_view line{buffer, size * nitems};

    if (line.starts_with("HTTP/")) {
        session->response_headers.clear();
        return size * nitems;
    }
    if (line == "\r\n" || line == "\n") {
        long code = 0;
        curl_easy_getinfo(session->easy, CURLINFO_RESPONSE_CODE, &code);
        if (code >= 200) {
            session->status = code;
            session->headers_done = true;
            session->wake();
        }
        return size * nitems;
    }
    std::size_t colon = line.find(':');
    if (colon != std::string_view::npos) {
        std::string_view name = line.substr(0, colon);
        std::string_view value = line.substr(colon + 1);
        while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
            value.remove_prefix(1);
        while (!value.empty()
               && (value.back() == '\r' || value.back() == '\n' || value.back() == ' '))
            value.remove_suffix(1);
        session->response_headers.emplace_back(std::string{name}, std::string{value});
    }
    return size * nitems;
}

// ───────────────────── 传输装配与等待原语 ─────────────────────

/// 装配 curl 句柄并启动传输（add_handle 后 curl 经 timer 回调自行开跑）。
/// @return false = curl 句柄创建失败（极罕见，内存耗尽级别）
bool setup_transfer(std::shared_ptr<CurlSession> const& session, std::string const& url,
                    nlohmann::json const& body, HttpRequestOptions const& options)
{
    session->request_body = body.dump();
    session->idle_timeout_ms = options.idle_timeout_ms;
    session->multi = curl_multi_init();
    session->easy = curl_easy_init();
    if (session->multi == nullptr || session->easy == nullptr)
        return false;

    CURLM* multi = session->multi;
    CURL* easy = session->easy;
    curl_multi_setopt(multi, CURLMOPT_SOCKETFUNCTION, multi_socket_callback);
    curl_multi_setopt(multi, CURLMOPT_SOCKETDATA, session.get());
    curl_multi_setopt(multi, CURLMOPT_TIMERFUNCTION, multi_timer_callback);
    curl_multi_setopt(multi, CURLMOPT_TIMERDATA, session.get());

    curl_easy_setopt(easy, CURLOPT_URL, url.c_str());
    curl_easy_setopt(easy, CURLOPT_POST, 1L);
    curl_easy_setopt(easy, CURLOPT_POSTFIELDS, session->request_body.c_str());
    curl_easy_setopt(easy, CURLOPT_POSTFIELDSIZE,
                     static_cast<long>(session->request_body.size()));

    // Expect: 空头禁用 100-continue 等待；Content-Type 固定 JSON；
    // 引擎合并后的头原样追加（认证头也在其中，http 层不理解语义）。
    session->header_list =
        curl_slist_append(session->header_list, "Content-Type: application/json");
    session->header_list = curl_slist_append(session->header_list, "Expect:");
    for (auto const& [name, value] : options.headers) {
        std::string header_line = name + ": " + value;
        session->header_list = curl_slist_append(session->header_list, header_line.c_str());
    }
    curl_easy_setopt(easy, CURLOPT_HTTPHEADER, session->header_list);

    curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(easy, CURLOPT_WRITEDATA, session.get());
    curl_easy_setopt(easy, CURLOPT_HEADERFUNCTION, header_callback);
    curl_easy_setopt(easy, CURLOPT_HEADERDATA, session.get());
    curl_easy_setopt(easy, CURLOPT_OPENSOCKETFUNCTION, open_socket_callback);
    curl_easy_setopt(easy, CURLOPT_OPENSOCKETDATA, session.get());
    curl_easy_setopt(easy, CURLOPT_CLOSESOCKETFUNCTION, close_socket_callback);
    curl_easy_setopt(easy, CURLOPT_CLOSESOCKETDATA, session.get());

    curl_easy_setopt(easy, CURLOPT_ERRORBUFFER, session->error_buffer);
    curl_easy_setopt(easy, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(easy, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
    // 空串 = 声明 curl 支持的全部编码（gzip/deflate/br），响应自动解压
    curl_easy_setopt(easy, CURLOPT_ACCEPT_ENCODING, "");

    // 超时映射：connect → CONNECTTIMEOUT_MS；total → TIMEOUT_MS。
    // idle（流式块间静默）不用 curl 的 LOW_SPEED（秒粒度 speeder 平均会
    // 被首块撑住，且需要 socket 活动驱动，不可靠）——由 next_chunk 层
    // 用 idle_timer 自管（见 next_chunk），阈值存 session。
    curl_easy_setopt(easy, CURLOPT_CONNECTTIMEOUT_MS,
                     static_cast<long>(options.connect_timeout_ms));
    if (options.total_timeout_ms > 0)
        curl_easy_setopt(easy, CURLOPT_TIMEOUT_MS,
                         static_cast<long>(options.total_timeout_ms));

    curl_multi_add_handle(multi, easy);
    return true;
}

/// 等待状态变化（headers_done / body 到达 / finished / aborted）。
/// event 常驻 max，wake() 以 cancel 唤醒。单线程模型下条件检查与挂起之间
/// 无回调可插入，不存在丢失唤醒。
asio::awaitable<void> wait_for_wake(std::shared_ptr<CurlSession> const& session)
{
    auto [ec] = co_await session->event.async_wait(asio::as_tuple(asio::use_awaitable));
    (void)ec;   // operation_aborted 即正常唤醒
}

/// 本协程是否已被 asio 取消（co_spawn 层的 cancellation 传播）。
asio::awaitable<bool> coroutine_cancelled()
{
    asio::cancellation_state state = co_await asio::this_coro::cancellation_state;
    co_return state.cancelled() != asio::cancellation_type::none;
}

/// 传输级 CURLcode → agent::Error。4xx/5xx 不经此路（响应正常返回给上层）。
Error map_curl_error(CURLcode code, char const* detail)
{
    std::string message = curl_easy_strerror(code);
    if (detail != nullptr && detail[0] != '\0') {
        message += ": ";
        message += detail;
    }
    return Error{Errc::NetworkError, std::move(message)};
}

/// 值得整次重试的 HTTP 状态（首字节前语义：headers 刚到、body 未消费）。
bool retryable_status(long status)
{
    return status == 429 || (status >= 500 && status < 600);
}

/// Retry-After 解析：仅秒数形态；HTTP 日期/缺失返回 nullopt（走指数退避）。
std::optional<int> parse_retry_after_ms(
    std::vector<std::pair<std::string, std::string>> const& headers)
{
    for (auto const& [name, value] : headers) {
        if (!equals_ignore_case(name, "retry-after"))
            continue;
        int seconds = 0;
        bool numeric = !value.empty();
        for (char c : value) {
            if (c < '0' || c > '9') {
                numeric = false;
                break;
            }
            seconds = seconds * 10 + (c - '0');
            if (seconds > 24 * 3600)
                break;   // 上限一天，防溢出
        }
        if (numeric)
            return seconds * 1000;
        return std::nullopt;
    }
    return std::nullopt;
}

/// 指数退避：0.5s · 2^attempt · (1 − rand·0.25)。
int backoff_delay_ms(int attempt)
{
    static std::minstd_rand engine{std::random_device{}()};
    std::uniform_real_distribution<double> jitter{0.0, 0.25};
    double delay = 500.0 * static_cast<double>(1 << attempt) * (1.0 - jitter(engine));
    return static_cast<int>(delay);
}

asio::awaitable<void> async_sleep(int delay_ms)
{
    asio::steady_timer timer{co_await asio::this_coro::executor};
    timer.expires_after(std::chrono::milliseconds(delay_ms));
    co_await timer.async_wait(asio::as_tuple(asio::use_awaitable));
}

/// 单次传输尝试的结果。
struct AttemptOutcome {
    std::shared_ptr<CurlSession> session;   // 成功（headers_done）时非空
    Error error{Errc::NetworkError, {}};    // session 为空时有效
    bool cancelled = false;                 // 取消（不重试）
};

/// 单次尝试：装配传输，等到最终响应头或传输失败。
asio::awaitable<AttemptOutcome> run_single_attempt(
    asio::any_io_executor ex, std::string const& url,
    nlohmann::json const& body, HttpRequestOptions& options)
{
    std::shared_ptr<CurlSession> session = std::make_shared<CurlSession>(ex);
    session->weak_self = session;

    // 用户取消信号（可能来自任意线程）→ post 到会话 executor 拆传输。
    if (options.cancel.is_connected()) {
        options.cancel.assign(
            [weak = std::weak_ptr<CurlSession>(session), ex](asio::cancellation_type) {
                asio::post(ex, [weak] {
                    if (std::shared_ptr<CurlSession> locked = weak.lock()) {
                        locked->aborted = true;
                        locked->shutdown();
                    }
                });
            });
    }

    if (!setup_transfer(session, url, body, options)) {
        co_return AttemptOutcome{nullptr,
            Error{Errc::NetworkError, "curl handle init failed"}, false};
    }

    while (!session->headers_done && !session->finished && !session->aborted) {
        if (co_await coroutine_cancelled()) {
            session->aborted = true;
            break;
        }
        co_await wait_for_wake(session);
    }

    if (session->aborted) {
        session->shutdown();
        co_return AttemptOutcome{nullptr,
            Error{Errc::NetworkError, "request cancelled"}, true};
    }
    if (session->headers_done)
        co_return AttemptOutcome{std::move(session), Error{}, false};

    // finished 而无响应头 = 首字节前传输失败（可重试）
    Error error = map_curl_error(session->curl_result, session->error_buffer);
    session->shutdown();
    co_return AttemptOutcome{nullptr, std::move(error), false};
}

/// 带首字节前重试的打开流程。重试仅针对：连接级失败、429/5xx 响应头。
/// 重试耗尽后仍有响应头 → 把响应交给上层（status/body 可读，详情归引擎）。
asio::awaitable<Result<std::shared_ptr<CurlSession>>> open_with_retries(
    asio::any_io_executor ex, std::string url,
    nlohmann::json body, HttpRequestOptions options)
{
    Error last_error{Errc::NetworkError, "no attempt executed"};
    for (int attempt = 0; attempt <= options.max_retries; ++attempt) {
        AttemptOutcome outcome = co_await run_single_attempt(ex, url, body, options);

        if (outcome.cancelled)
            co_return std::unexpected(std::move(outcome.error));

        if (outcome.session == nullptr) {
            last_error = std::move(outcome.error);
            if (attempt == options.max_retries)
                co_return std::unexpected(std::move(last_error));
            co_await async_sleep(backoff_delay_ms(attempt));
            continue;
        }

        long status = outcome.session->status;
        if (retryable_status(status) && attempt < options.max_retries) {
            std::optional<int> retry_after =
                parse_retry_after_ms(outcome.session->response_headers);
            int delay_ms = retry_after.value_or(backoff_delay_ms(attempt));
            if (delay_ms > options.max_retry_delay_ms)
                co_return std::move(outcome.session);   // 等不起，响应交上层
            outcome.session->shutdown();
            co_await async_sleep(delay_ms);
            if (co_await coroutine_cancelled())
                co_return std::unexpected(Error{Errc::NetworkError, "request cancelled"});
            continue;
        }
        co_return std::move(outcome.session);
    }
    co_return std::unexpected(std::move(last_error));
}

}   // namespace

// ───────────────────── HttpStreamReader ─────────────────────

struct HttpStreamReader::Impl {
    std::shared_ptr<CurlSession> session;
};

HttpStreamReader::HttpStreamReader(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
HttpStreamReader::HttpStreamReader(HttpStreamReader&& other) noexcept = default;
HttpStreamReader& HttpStreamReader::operator=(HttpStreamReader&& other) noexcept = default;

HttpStreamReader::~HttpStreamReader()
{
    // 提前丢弃（流未读完）：拆传输，挂起回调空转后自然回收会话。
    if (impl_ != nullptr && impl_->session != nullptr)
        impl_->session->shutdown();
}

asio::awaitable<Result<HttpStreamReader>> HttpStreamReader::open(
    asio::any_io_executor ex, std::string_view url,
    nlohmann::json body, HttpRequestOptions options)
{
    ensure_curl_global();
    Result<std::shared_ptr<CurlSession>> session = co_await open_with_retries(
        ex, std::string{url}, std::move(body), std::move(options));
    if (!session)
        co_return std::unexpected(std::move(session).error());
    std::unique_ptr<Impl> impl{new Impl{std::move(*session)}};
    co_return HttpStreamReader{std::move(impl)};
}

asio::awaitable<Result<std::optional<std::string>>> HttpStreamReader::next_chunk()
{
    std::shared_ptr<CurlSession> const& session = impl_->session;
    while (true) {
        if (!session->body_chunks.empty()) {
            session->idle_timer.cancel();
            session->idle_expired = false;   // 收到块 → idle 计时清零
            std::string chunk = std::move(session->body_chunks.front());
            session->body_chunks.pop_front();
            session->buffered_bytes -= chunk.size();
            // 背压恢复：消费到低水位以下时让 curl 继续交付，并踢一下状态机
            //（unpause 后 curl 需要一次 action 才恢复读 socket）。
            if (session->paused && session->buffered_bytes < kLowWaterBytes
                && !session->stopped && !session->finished) {
                session->paused = false;
                curl_easy_pause(session->easy, CURLPAUSE_CONT);
                int running = 0;
                curl_multi_socket_action(session->multi, CURL_SOCKET_TIMEOUT, 0, &running);
                process_curl_messages(session);
                if (!session->stopped)
                    arm_all_sockets(session);
            }
            co_return std::optional<std::string>{std::move(chunk)};
        }
        if (session->finished) {
            if (session->curl_result == CURLE_OK)
                co_return std::optional<std::string>{};   // EOF，流正常结束
            co_return std::unexpected(
                map_curl_error(session->curl_result, session->error_buffer));
        }
        if (session->aborted)
            co_return std::unexpected(Error{Errc::NetworkError, "request cancelled"});
        if (session->idle_expired)
            co_return std::unexpected(Error{Errc::NetworkError, "idle timeout"});
        if (co_await coroutine_cancelled())
            co_return std::unexpected(Error{Errc::NetworkError, "request cancelled"});

        // 挂 idle 超时：expires_after 会取消旧的 pending wait（asio 语义），
        // 无需手动 cancel。回调里用 weak 防 session 析构后悬空。
        // 计时只覆盖「等待数据」——收到数据/终态即返回，下次进入重挂。
        if (session->idle_timeout_ms > 0) {
            session->idle_timer.expires_after(
                std::chrono::milliseconds(session->idle_timeout_ms));
            session->idle_timer.async_wait(
                [weak = std::weak_ptr<CurlSession>(session)](asio::error_code const& ec) {
                    if (ec)
                        return;   // cancel/shutdown 后不动作
                    if (std::shared_ptr<CurlSession> locked = weak.lock()) {
                        locked->idle_expired = true;
                        locked->wake();
                    }
                });
        }
        co_await wait_for_wake(session);
    }
}

int HttpStreamReader::status() const
{
    return static_cast<int>(impl_->session->status);
}

std::vector<std::pair<std::string, std::string>> const& HttpStreamReader::headers() const
{
    return impl_->session->response_headers;
}

std::string_view HttpStreamReader::header(std::string_view name) const
{
    for (auto const& [header_name, value] : impl_->session->response_headers) {
        if (equals_ignore_case(header_name, name))
            return value;
    }
    return {};
}

// ───────────────────── async_http_post ─────────────────────

asio::awaitable<Result<HttpResponse>> async_http_post(
    asio::any_io_executor ex, std::string_view url,
    nlohmann::json body, HttpRequestOptions options)
{
    Result<HttpStreamReader> reader =
        co_await HttpStreamReader::open(ex, url, std::move(body), std::move(options));
    if (!reader)
        co_return std::unexpected(std::move(reader).error());

    HttpResponse response;
    response.status = reader->status();
    while (true) {
        Result<std::optional<std::string>> chunk = co_await reader->next_chunk();
        if (!chunk)
            co_return std::unexpected(std::move(chunk).error());
        if (!chunk->has_value())
            break;
        response.body += **chunk;
    }
    response.headers = reader->headers();
    co_return response;
}

}   // namespace agent
