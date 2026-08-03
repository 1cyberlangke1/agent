// L0 传输层实现 + 完整 curl 封装：libcurl multi_socket 模式嫁接 asio io_context。
//
// 线程模型（关键约束）：一次传输（CurlSession）的全部状态运行在单一
// executor 上——curl 回调只在 curl_multi_socket_action 内同步执行，而
// socket_action 只被该 executor 上的 asio 完成回调调用。容器无锁。
// 唯一跨线程入口是取消信号（emit 可来自任意线程），经 asio::post 跳转。
// 调用方要求：executor 须为单线程 io_context（或 strand）。HttpClient 同样。
//
// socket 归属：socket 由 asio 创建（CURLOPT_OPENSOCKETFUNCTION），curl 只
// 借用 native handle 做连接与读写；监听（可读/可写）由 asio async_wait
// 驱动，事件到达后回调 curl_multi_socket_action 推进 curl 状态机。
// 关闭（CURLOPT_CLOSESOCKETFUNCTION）时从表中移除，asio socket 析构收尾。
//
// 生命周期：CurlSession 由 shared_ptr 管理——所有挂起的 asio 回调各持
// 一份引用；shutdown() 拆除 curl 句柄并取消监听，其后到达的回调看到
// stopped 标志空转返回，最后一个回调释放引用后 CurlSession 析构。
//
// 能力覆盖（PLAN_4.md）：方法（POST/CUSTOMREQUEST/NOBODY）、body（字符串 /
// multipart curl_mime / 大文件流式上传 READFUNCTION）、auth（Basic/Bearer/
// Digest/Ntlm/Negotiate）、TLS（verify/CA/客户端证书）、重定向、代理（含
// SOCKS）、cookie、响应元数据（effective_url/total_time）。便捷层把非 2xx
// 归一化成 Error（RateLimited/AuthError/ProviderError）+ ParseError。
//
// HttpClient 连接复用：curl_easy_reset 保留 alive 连接，easy 句柄池跨请求
// 复用 → 同一 host 走 keep-alive。

#include <agent/core/http_client.hpp>

#include <curl/curl.h>

#include <chrono>
#include <cstdio>
#include <deque>
#include <fstream>
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

constexpr std::size_t kHighWaterBytes = std::size_t{1} << 20;   // 1 MiB
constexpr std::size_t kLowWaterBytes = std::size_t{256} << 10;  // 256 KiB

void ensure_curl_global()
{
    static std::once_flag flag;
    std::call_once(flag, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

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

struct CurlSession;

struct SocketState {
    asio::ip::tcp::socket socket;
    int watch = 0;
    int pending = 0;
};

/// multi 的 socket/timer 回调上下文 + **连接池持有的 socket 表**。
/// socket 归属 multi 而非 session：curl 池化连接后 socket 必须存活到连接关闭
///（close_socket_callback），否则复用连接变成死连接（curl "shutting down"）。
/// 私有 multi：ctx 是 session 成员（随 session 析构 → socket 关闭）；共享 multi（HttpClient）：
/// ctx 常驻客户端，socket 跨请求存活实现 keep-alive 复用。
struct MultiCtx {
    CurlSession* active = nullptr;
    std::map<curl_socket_t, SocketState> sockets;
};

class CurlSession {
public:
    asio::any_io_executor executor;
    asio::steady_timer curl_timer;
    asio::steady_timer event;
    asio::steady_timer idle_timer;
    int idle_timeout_ms = 0;
    bool idle_expired = false;
    std::weak_ptr<CurlSession> weak_self;

    CURLM* multi = nullptr;
    bool owns_multi = true;               // false = 共享 multi（HttpClient，shutdown 不 cleanup）
    MultiCtx ctx;                         // 私有 multi 的回调上下文（active = this）
    MultiCtx* ctx_ptr = nullptr;          // 实际使用的 ctx（私有 = &ctx；共享 = HttpClient 的）
    CURL* easy = nullptr;
    bool owns_easy = true;                // false = 来自 HttpClient easy 池（shutdown 不 cleanup）
    curl_slist* header_list = nullptr;
    curl_mime* mime = nullptr;            // multipart（curl_mime 持有）
    FILE* upload_file = nullptr;          // 流式上传文件
    std::string request_body;             // CURLOPT_POSTFIELDS 指向的存储
    char error_buffer[CURL_ERROR_SIZE] = {};

    long status = 0;
    std::vector<std::pair<std::string, std::string>> response_headers;
    bool headers_done = false;
    std::string effective_url;
    double total_time = 0;

    std::deque<std::string> body_chunks;
    std::size_t buffered_bytes = 0;
    bool paused = false;

    bool finished = false;
    CURLcode curl_result = CURLE_OK;
    bool aborted = false;
    bool stopped = false;

    explicit CurlSession(asio::any_io_executor ex)
        : executor(ex), curl_timer(ex), event(ex), idle_timer(ex)
    {
        event.expires_at(std::chrono::steady_clock::time_point::max());
    }

    ~CurlSession() { shutdown(); }

    CurlSession(CurlSession const&) = delete;
    CurlSession& operator=(CurlSession const&) = delete;

    void wake() { event.cancel(); }

    void shutdown()
    {
        if (stopped)
            return;
        stopped = true;
        curl_timer.cancel();
        idle_timer.cancel();
        if (ctx_ptr != nullptr) {
            for (auto& [fd, state] : ctx_ptr->sockets) {
                asio::error_code ignored;
                state.socket.cancel(ignored);   // 取消监听；socket 归 ctx（连接），不关闭
            }
        }
        if (multi != nullptr && easy != nullptr)
            curl_multi_remove_handle(multi, easy);
        if (owns_easy && easy != nullptr) {
            curl_easy_cleanup(easy);
            easy = nullptr;
        }
        if (owns_multi && multi != nullptr) {
            curl_multi_cleanup(multi);
            multi = nullptr;
        }
        if (header_list != nullptr) {
            curl_slist_free_all(header_list);
            header_list = nullptr;
        }
        if (mime != nullptr) {
            curl_mime_free(mime);
            mime = nullptr;
        }
        if (upload_file != nullptr) {
            fclose(upload_file);
            upload_file = nullptr;
        }
        wake();
    }
};

// ───────────────────── curl ↔ asio 桥接 ─────────────────────

void process_curl_messages(std::shared_ptr<CurlSession> const& session);
void arm_all_sockets(std::shared_ptr<CurlSession> const& session);

void on_socket_ready(std::shared_ptr<CurlSession> const& session, curl_socket_t fd,
                     int which, asio::error_code const& ec)
{
    if (session->stopped || session->ctx_ptr == nullptr)
        return;
    auto found = session->ctx_ptr->sockets.find(fd);
    if (found == session->ctx_ptr->sockets.end())
        return;
    found->second.pending &= ~which;
    if (ec == asio::error::operation_aborted)
        return;

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

void arm_all_sockets(std::shared_ptr<CurlSession> const& session)
{
    if (session->ctx_ptr == nullptr)
        return;
    for (auto& [fd, state] : session->ctx_ptr->sockets) {
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

void process_curl_messages(std::shared_ptr<CurlSession> const& session)
{
    if (session->stopped || session->multi == nullptr)
        return;
    int queued = 0;
    while (CURLMsg* message = curl_multi_info_read(session->multi, &queued)) {
        if (message->msg == CURLMSG_DONE) {
            session->finished = true;
            session->curl_result = message->data.result;
            char* eff = nullptr;
            curl_easy_getinfo(session->easy, CURLINFO_EFFECTIVE_URL, &eff);
            session->effective_url = eff != nullptr ? eff : "";
            curl_easy_getinfo(session->easy, CURLINFO_TOTAL_TIME, &session->total_time);
            session->wake();
        }
    }
}

// ───────────────────── curl C 回调 ─────────────────────

curl_socket_t open_socket_callback(void* clientp, curlsocktype purpose,
                                   curl_sockaddr* address)
{
    CurlSession* session = static_cast<CurlSession*>(clientp);
    if (session->ctx_ptr == nullptr)
        return CURL_SOCKET_BAD;
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
    session->ctx_ptr->sockets.emplace(fd, SocketState{std::move(socket)});
    return fd;
}

int close_socket_callback(void* clientp, curl_socket_t item)
{
    CurlSession* session = static_cast<CurlSession*>(clientp);
    if (session->ctx_ptr != nullptr)
        session->ctx_ptr->sockets.erase(item);   // 连接关闭 → socket 析构关闭
    return 0;
}

int multi_socket_callback(CURL*, curl_socket_t fd, int what, void* userp, void*)
{
    MultiCtx* ctx = static_cast<MultiCtx*>(userp);
    CurlSession* session = ctx->active;
    if (session == nullptr || session->stopped)
        return 0;
    auto found = ctx->sockets.find(fd);
    if (found == ctx->sockets.end())
        return 0;
    if (what == CURL_POLL_REMOVE) {
        found->second.watch = 0;
        asio::error_code ignored;
        found->second.socket.cancel(ignored);
        return 0;
    }
    found->second.watch = what;
    return 0;
}

int multi_timer_callback(CURLM*, long timeout_ms, void* userp)
{
    MultiCtx* ctx = static_cast<MultiCtx*>(userp);
    CurlSession* session = ctx->active;
    if (session == nullptr || session->stopped)
        return 0;
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

size_t write_callback(char* data, size_t size, size_t nmemb, void* clientp)
{
    CurlSession* session = static_cast<CurlSession*>(clientp);
    if (session->stopped)
        return 0;
    size_t total = size * nmemb;
    if (session->buffered_bytes >= kHighWaterBytes) {
        session->paused = true;
        return CURL_WRITEFUNC_PAUSE;
    }
    session->headers_done = true;
    session->body_chunks.emplace_back(data, total);
    session->buffered_bytes += total;
    session->wake();
    return total;
}

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

/// 大文件流式上传的读回调（CURLOPT_READFUNCTION）。
size_t upload_read_callback(char* buffer, size_t size, size_t nitems, void* userp)
{
    FILE* file = static_cast<FILE*>(userp);
    return fread(buffer, size, nitems, file);
}

// ───────────────────── 传输装配（完整 curl 映射）─────────────────────

/// 按请求装配 curl easy 并加入 multi（私有或共享）。
/// shared_multi 非空 = HttpClient 的常驻 multi（连接池复用，owns_multi=false，
/// callbacks 走 shared_ctx->active）；否则每请求自建 multi。
/// reuse_easy 非空 = HttpClient easy 池句柄（owns_easy=false，shutdown 不 cleanup，
/// 调用方负责 curl_easy_reset 还池——reset 清 option、不动连接，连接在 multi 池）。
bool setup_transfer(std::shared_ptr<CurlSession> const& session,
                    HttpRequest const& request, HttpRequestOptions const& options,
                    MultiCtx* shared_ctx, CURLM* shared_multi, CURL* reuse_easy)
{
    std::string url = append_query(request.url, request.query);
    std::string method = request.method.empty() ? "GET" : request.method;

    session->idle_timeout_ms = options.idle_timeout_ms;
    session->easy = reuse_easy != nullptr ? reuse_easy : curl_easy_init();
    session->owns_easy = (reuse_easy == nullptr);
    if (session->easy == nullptr)
        return false;

    if (shared_multi != nullptr) {
        session->multi = shared_multi;
        session->owns_multi = false;
        shared_ctx->active = session.get();
        session->ctx_ptr = shared_ctx;
    } else {
        session->multi = curl_multi_init();
        session->owns_multi = true;
        session->ctx.active = session.get();
        session->ctx_ptr = &session->ctx;
    }
    if (session->multi == nullptr)
        return false;

    CURLM* multi = session->multi;
    CURL* easy = session->easy;
    MultiCtx* ctx = session->ctx_ptr;
    curl_multi_setopt(multi, CURLMOPT_SOCKETFUNCTION, multi_socket_callback);
    curl_multi_setopt(multi, CURLMOPT_SOCKETDATA, ctx);
    curl_multi_setopt(multi, CURLMOPT_TIMERFUNCTION, multi_timer_callback);
    curl_multi_setopt(multi, CURLMOPT_TIMERDATA, ctx);

    curl_easy_setopt(easy, CURLOPT_URL, url.c_str());

    // ── 方法 ──
    bool has_body = request.body.has_value() || !request.multipart.empty()
                 || request.upload_file.has_value();
    if (method == "POST")
        curl_easy_setopt(easy, CURLOPT_POST, 1L);
    else if (method == "HEAD")
        curl_easy_setopt(easy, CURLOPT_NOBODY, 1L);
    else if (method != "GET")
        curl_easy_setopt(easy, CURLOPT_CUSTOMREQUEST, method.c_str());

    // ── body：大文件流式 / multipart / 字符串 ──
    if (request.upload_file.has_value()) {
        FILE* file = fopen(request.upload_file->c_str(), "rb");
        if (file == nullptr)
            return false;
        session->upload_file = file;
        fseek(file, 0, SEEK_END);
        long size = ftell(file);
        fseek(file, 0, SEEK_SET);
        curl_easy_setopt(easy, CURLOPT_READFUNCTION, upload_read_callback);
        curl_easy_setopt(easy, CURLOPT_READDATA, file);
        curl_easy_setopt(easy, CURLOPT_INFILESIZE_LARGE, static_cast<curl_off_t>(size));
        if (method == "POST")
            curl_easy_setopt(easy, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(size));
        else if (method == "PUT") {
            curl_easy_setopt(easy, CURLOPT_UPLOAD, 1L);
        } else {
            curl_easy_setopt(easy, CURLOPT_UPLOAD, 1L);
            curl_easy_setopt(easy, CURLOPT_CUSTOMREQUEST, method.c_str());
        }
        if (!request.content_type.has_value())
            session->header_list = curl_slist_append(session->header_list,
                "Content-Type: application/octet-stream");
    } else if (!request.multipart.empty()) {
        curl_mime* mime = curl_mime_init(easy);
        for (auto const& part : request.multipart) {
            curl_mimepart* mime_part = curl_mime_addpart(mime);
            if (!part.name.empty())
                curl_mime_name(mime_part, part.name.c_str());
            if (part.value.has_value())
                curl_mime_data(mime_part, part.value->c_str(), CURL_ZERO_TERMINATED);
            else if (part.buffer.has_value())
                curl_mime_data(mime_part, part.buffer->c_str(), part.buffer->size());
            else if (part.file_path.has_value())
                curl_mime_filedata(mime_part, part.file_path->c_str());
            if (part.content_type.has_value())
                curl_mime_type(mime_part, part.content_type->c_str());
            if (part.filename.has_value())
                curl_mime_filename(mime_part, part.filename->c_str());
        }
        session->mime = mime;
        curl_easy_setopt(easy, CURLOPT_MIMEPOST, mime);
        if (request.content_type.has_value())
            session->header_list = curl_slist_append(session->header_list,
                ("Content-Type: " + *request.content_type).c_str());
    } else if (request.body.has_value()) {
        session->request_body = *request.body;
        curl_easy_setopt(easy, CURLOPT_POSTFIELDS, session->request_body.c_str());
        curl_easy_setopt(easy, CURLOPT_POSTFIELDSIZE,
                         static_cast<long>(session->request_body.size()));
        std::string content_type = request.content_type.value_or("application/json");
        session->header_list = curl_slist_append(session->header_list,
            ("Content-Type: " + content_type).c_str());
    }

    // POST 带 body：禁用 100-continue 等待
    if (has_body && method == "POST")
        session->header_list = curl_slist_append(session->header_list, "Expect:");

    // ── 认证 ──
    if (request.auth == HttpAuth::Bearer) {
        if (request.auth_credentials.has_value())
            session->header_list = curl_slist_append(session->header_list,
                ("Authorization: Bearer " + *request.auth_credentials).c_str());
    } else if (request.auth != HttpAuth::None) {
        long auth_type = CURLAUTH_BASIC;
        switch (request.auth) {
            case HttpAuth::Digest: auth_type = CURLAUTH_DIGEST; break;
            case HttpAuth::Ntlm: auth_type = CURLAUTH_NTLM; break;
            case HttpAuth::Negotiate: auth_type = CURLAUTH_NEGOTIATE; break;
            default: auth_type = CURLAUTH_BASIC;
        }
        curl_easy_setopt(easy, CURLOPT_HTTPAUTH, auth_type);
        if (request.auth_credentials.has_value())
            curl_easy_setopt(easy, CURLOPT_USERPWD, request.auth_credentials->c_str());
    }

    // ── 用户头（request + options）──
    for (auto const& [name, value] : request.headers)
        session->header_list = curl_slist_append(session->header_list,
            (name + ": " + value).c_str());
    for (auto const& [name, value] : options.headers)
        session->header_list = curl_slist_append(session->header_list,
            (name + ": " + value).c_str());
    if (session->header_list != nullptr)
        curl_easy_setopt(easy, CURLOPT_HTTPHEADER, session->header_list);

    // ── TLS ──
    curl_easy_setopt(easy, CURLOPT_SSL_VERIFYPEER, request.verify_tls ? 1L : 0L);
    curl_easy_setopt(easy, CURLOPT_SSL_VERIFYHOST, request.verify_tls ? 2L : 0L);
    if (request.ca_file.has_value())
        curl_easy_setopt(easy, CURLOPT_CAINFO, request.ca_file->c_str());
    if (request.client_cert.has_value())
        curl_easy_setopt(easy, CURLOPT_SSLCERT, request.client_cert->c_str());
    if (request.client_key.has_value())
        curl_easy_setopt(easy, CURLOPT_SSLKEY, request.client_key->c_str());
    if (request.client_key_password.has_value())
        curl_easy_setopt(easy, CURLOPT_KEYPASSWD, request.client_key_password->c_str());

    // ── 重定向 ──
    if (request.follow_redirects) {
        curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(easy, CURLOPT_MAXREDIRS, static_cast<long>(request.max_redirects));
        if (request.auto_referer)
            curl_easy_setopt(easy, CURLOPT_AUTOREFERER, 1L);
    }

    // ── 代理 ──
    if (request.proxy.has_value()) {
        curl_easy_setopt(easy, CURLOPT_PROXY, request.proxy->c_str());
        long proxy_type = CURLPROXY_HTTP;
        switch (request.proxy_type) {
            case HttpProxyType::Https: proxy_type = CURLPROXY_HTTPS; break;
            case HttpProxyType::Socks4: proxy_type = CURLPROXY_SOCKS4; break;
            case HttpProxyType::Socks5: proxy_type = CURLPROXY_SOCKS5; break;
            default: proxy_type = CURLPROXY_HTTP;
        }
        curl_easy_setopt(easy, CURLOPT_PROXYTYPE, proxy_type);
        if (request.proxy_auth.has_value()) {
            curl_easy_setopt(easy, CURLOPT_PROXYUSERPWD, request.proxy_auth->c_str());
            curl_easy_setopt(easy, CURLOPT_PROXYAUTH, CURLAUTH_BASIC);
        }
    }

    // ── Cookie ──
    if (request.cookie.has_value())
        curl_easy_setopt(easy, CURLOPT_COOKIE, request.cookie->c_str());
    if (request.cookie_file.has_value())
        curl_easy_setopt(easy, CURLOPT_COOKIEFILE, request.cookie_file->c_str());
    if (request.cookie_jar.has_value())
        curl_easy_setopt(easy, CURLOPT_COOKIEJAR, request.cookie_jar->c_str());
    if (request.cookie_session)
        curl_easy_setopt(easy, CURLOPT_COOKIESESSION, 1L);

    // ── 基础传输选项 ──
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
    curl_easy_setopt(easy, CURLOPT_ACCEPT_ENCODING, "");

    curl_easy_setopt(easy, CURLOPT_CONNECTTIMEOUT_MS,
                     static_cast<long>(options.connect_timeout_ms));
    if (options.total_timeout_ms > 0)
        curl_easy_setopt(easy, CURLOPT_TIMEOUT_MS,
                         static_cast<long>(options.total_timeout_ms));

    curl_multi_add_handle(multi, easy);
    return true;
}

// ───────────────────── 等待与重试原语 ─────────────────────

asio::awaitable<void> wait_for_wake(std::shared_ptr<CurlSession> const& session)
{
    auto [ec] = co_await session->event.async_wait(asio::as_tuple(asio::use_awaitable));
    (void)ec;
}

asio::awaitable<bool> coroutine_cancelled()
{
    asio::cancellation_state state = co_await asio::this_coro::cancellation_state;
    co_return state.cancelled() != asio::cancellation_type::none;
}

Error map_curl_error(CURLcode code, char const* detail)
{
    std::string message = curl_easy_strerror(code);
    if (detail != nullptr && detail[0] != '\0') {
        message += ": ";
        message += detail;
    }
    return Error{Errc::NetworkError, std::move(message)};
}

bool retryable_status(long status)
{
    return status == 429 || (status >= 500 && status < 600);
}

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
                break;
        }
        if (numeric)
            return seconds * 1000;
        return std::nullopt;
    }
    return std::nullopt;
}

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

struct AttemptOutcome {
    std::shared_ptr<CurlSession> session;
    Error error{Errc::NetworkError, {}};
    bool cancelled = false;
};

asio::awaitable<AttemptOutcome> run_single_attempt(
    asio::any_io_executor ex, HttpRequest const& request,
    HttpRequestOptions& options, MultiCtx* shared_ctx, CURLM* shared_multi, CURL* reuse_easy)
{
    std::shared_ptr<CurlSession> session = std::make_shared<CurlSession>(ex);
    session->weak_self = session;

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

    if (!setup_transfer(session, request, options, shared_ctx, shared_multi, reuse_easy)) {
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

    Error error = map_curl_error(session->curl_result, session->error_buffer);
    session->shutdown();
    co_return AttemptOutcome{nullptr, std::move(error), false};
}

asio::awaitable<Result<std::shared_ptr<CurlSession>>> open_with_retries(
    asio::any_io_executor ex, HttpRequest const& request,
    HttpRequestOptions options, MultiCtx* shared_ctx, CURLM* shared_multi, CURL* reuse_easy)
{
    Error last_error{Errc::NetworkError, "no attempt executed"};
    for (int attempt = 0; attempt <= options.max_retries; ++attempt) {
        AttemptOutcome outcome =
            co_await run_single_attempt(ex, request, options, shared_ctx, shared_multi, reuse_easy);

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
                co_return std::move(outcome.session);
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

/// 排空 session body 到字符串（非流式收集）。
asio::awaitable<Result<std::string>> drain_all(std::shared_ptr<CurlSession> const& session)
{
    std::string body;
    while (true) {
        if (!session->body_chunks.empty()) {
            std::string chunk = std::move(session->body_chunks.front());
            session->body_chunks.pop_front();
            session->buffered_bytes -= chunk.size();
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
            body += chunk;
            continue;
        }
        if (session->finished) {
            if (session->curl_result == CURLE_OK)
                co_return body;
            co_return std::unexpected(
                map_curl_error(session->curl_result, session->error_buffer));
        }
        if (session->aborted)
            co_return std::unexpected(Error{Errc::NetworkError, "request cancelled"});
        if (co_await coroutine_cancelled())
            co_return std::unexpected(Error{Errc::NetworkError, "request cancelled"});
        co_await wait_for_wake(session);
    }
}

/// 非流式收集：open（带重试）→ 排空 → 填 HttpResponse（含元数据）。
/// shared_ctx/shared_multi 非空 = HttpClient 常驻 multi（连接池复用）。
/// reuse_easy 非空 = HttpClient easy 池句柄（调用方负责 reset 还池）。
asio::awaitable<Result<HttpResponse>> collect_response(
    asio::any_io_executor ex, HttpRequest const& request,
    HttpRequestOptions options, MultiCtx* shared_ctx, CURLM* shared_multi, CURL* reuse_easy)
{
    ensure_curl_global();
    Result<std::shared_ptr<CurlSession>> session = co_await open_with_retries(
        ex, request, std::move(options), shared_ctx, shared_multi, reuse_easy);
    if (!session)
        co_return std::unexpected(std::move(session).error());

    Result<std::string> body = co_await drain_all(*session);
    if (!body) {
        (*session)->shutdown();
        co_return std::unexpected(std::move(body).error());
    }

    HttpResponse response;
    response.status = static_cast<int>((*session)->status);
    response.body = std::move(*body);
    response.headers = (*session)->response_headers;
    response.effective_url = (*session)->effective_url;
    response.total_time_seconds = (*session)->total_time;
    (*session)->shutdown();
    co_return response;
}

}  // namespace

// ───────────────────── HttpResponse::json ─────────────────────

Result<nlohmann::json> HttpResponse::json() const
{
    nlohmann::json parsed = nlohmann::json::parse(body, nullptr, false);
    if (parsed.is_discarded())
        return std::unexpected(Error{ Errc::ParseError, "response body is not valid JSON" });
    return parsed;
}

// ───────────────────── URL 工具 ─────────────────────

std::string url_encode(std::string_view s)
{
    std::string out;
    for (unsigned char c : s) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
            || c == '-' || c == '_' || c == '.' || c == '~') {
            out += static_cast<char>(c);
        } else {
            char buf[4];
            std::snprintf(buf, sizeof(buf), "%%%02X", c);
            out += buf;
        }
    }
    return out;
}

std::string append_query(std::string url,
                         std::vector<std::pair<std::string, std::string>> params)
{
    if (params.empty())
        return url;
    bool first = url.find('?') == std::string::npos;
    for (auto const& [key, value] : params) {
        url += first ? '?' : '&';
        first = false;
        url += url_encode(key);
        url += '=';
        url += url_encode(value);
    }
    return url;
}

// ───────────────────── 通用请求原语 ─────────────────────

asio::awaitable<Result<HttpResponse>> async_http_request(
    asio::any_io_executor ex, HttpRequest const& request, HttpRequestOptions options)
{
    co_return co_await collect_response(ex, request, std::move(options), nullptr, nullptr, nullptr);
}

Result<HttpResponse> sync_http_request(HttpRequest const& request, HttpRequestOptions options)
{
    asio::io_context io;
    std::optional<Result<HttpResponse>> outcome;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        outcome = co_await async_http_request(
            co_await asio::this_coro::executor, request, std::move(options));
    }, asio::detached);
    io.run();
    if (!outcome)
        return std::unexpected(Error{ Errc::NetworkError, "sync_http_request: no outcome" });
    return std::move(*outcome);
}

// ───────────────────── 便捷层（非 2xx → Error）─────────────────────

namespace {

/// 便捷层错误归一化：非 2xx → RateLimited/AuthError/ProviderError。
Result<nlohmann::json> normalize_json(Result<HttpResponse> const& response)
{
    if (!response)
        return std::unexpected(response.error());
    if (!response->ok()) {
        Errc code = Errc::ProviderError;
        if (response->status == 429)
            code = Errc::RateLimited;
        else if (response->status == 401 || response->status == 403)
            code = Errc::AuthError;
        std::string snippet = response->body.substr(0, 512);
        return std::unexpected(Error{ code, "HTTP " + std::to_string(response->status)
                                           + ": " + snippet });
    }
    return response->json();
}

}  // namespace

Result<nlohmann::json> http_get_json(std::string url, HttpRequestOptions options)
{
    HttpRequest request;
    request.method = "GET";
    request.url = std::move(url);
    return normalize_json(sync_http_request(request, std::move(options)));
}

Result<nlohmann::json> http_post_json(std::string url, nlohmann::json body,
                                      HttpRequestOptions options)
{
    HttpRequest request;
    request.method = "POST";
    request.url = std::move(url);
    request.body = body.dump();
    request.content_type = "application/json";
    return normalize_json(sync_http_request(request, std::move(options)));
}

Result<nlohmann::json> http_post_form(
    std::string url, std::vector<std::pair<std::string, std::string>> fields,
    HttpRequestOptions options)
{
    std::string form;
    bool first = true;
    for (auto const& [key, value] : fields) {
        if (!first)
            form += '&';
        first = false;
        form += url_encode(key);
        form += '=';
        form += url_encode(value);
    }
    HttpRequest request;
    request.method = "POST";
    request.url = std::move(url);
    request.body = std::move(form);
    request.content_type = "application/x-www-form-urlencoded";
    return normalize_json(sync_http_request(request, std::move(options)));
}

Result<void> http_upload_file(std::string url, std::string file_path,
                              HttpRequestOptions options)
{
    HttpRequest request;
    request.method = "POST";
    request.url = std::move(url);
    request.multipart.push_back(MultipartPart{
        .name = "file",
        .file_path = std::move(file_path),
        .content_type = "application/octet-stream",
    });
    Result<HttpResponse> response = sync_http_request(request, std::move(options));
    if (!response)
        return std::unexpected(response.error());
    if (!response->ok())
        return std::unexpected(Error{ Errc::ProviderError,
            "HTTP " + std::to_string(response->status) + ": " + response->body.substr(0, 512) });
    return {};
}

asio::awaitable<Result<nlohmann::json>> async_http_get_json(
    asio::any_io_executor ex, std::string url, HttpRequestOptions options)
{
    HttpRequest request;
    request.method = "GET";
    request.url = std::move(url);
    Result<HttpResponse> response = co_await async_http_request(ex, request, std::move(options));
    co_return normalize_json(response);
}

asio::awaitable<Result<nlohmann::json>> async_http_post_json(
    asio::any_io_executor ex, std::string url, nlohmann::json body,
    HttpRequestOptions options)
{
    HttpRequest request;
    request.method = "POST";
    request.url = std::move(url);
    request.body = body.dump();
    request.content_type = "application/json";
    Result<HttpResponse> response = co_await async_http_request(ex, request, std::move(options));
    co_return normalize_json(response);
}

// ───────────────────── 非流式 POST / GET（向后兼容）─────────────────────

asio::awaitable<Result<HttpResponse>> async_http_post(
    asio::any_io_executor ex, std::string_view url,
    nlohmann::json body, HttpRequestOptions options)
{
    HttpRequest request;
    request.method = "POST";
    request.url = std::string(url);
    request.body = body.dump();
    request.content_type = "application/json";
    co_return co_await collect_response(ex, request, std::move(options), nullptr, nullptr, nullptr);
}

asio::awaitable<Result<HttpResponse>> async_http_get(
    asio::any_io_executor ex, std::string_view url, HttpRequestOptions options)
{
    HttpRequest request;
    request.method = "GET";
    request.url = std::string(url);
    co_return co_await collect_response(ex, request, std::move(options), nullptr, nullptr, nullptr);
}

Result<HttpResponse> sync_http_get(std::string_view url, HttpRequestOptions options)
{
    asio::io_context io;
    std::optional<Result<HttpResponse>> outcome;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        outcome = co_await async_http_get(
            co_await asio::this_coro::executor, url, std::move(options));
    }, asio::detached);
    io.run();
    if (!outcome)
        return std::unexpected(Error{ Errc::NetworkError, "sync_http_get: no outcome" });
    return std::move(*outcome);
}

Result<HttpResponse> sync_http_post(std::string_view url, nlohmann::json body, HttpRequestOptions options)
{
    asio::io_context io;
    std::optional<Result<HttpResponse>> outcome;
    asio::co_spawn(io, [&]() -> asio::awaitable<void> {
        outcome = co_await async_http_post(
            co_await asio::this_coro::executor, url, std::move(body), std::move(options));
    }, asio::detached);
    io.run();
    if (!outcome)
        return std::unexpected(Error{ Errc::NetworkError, "sync_http_post: no outcome" });
    return std::move(*outcome);
}

// ───────────────────── HttpClient（常驻 multi 连接复用 + cookie jar）─────────────────────

HttpClient::HttpClient(std::optional<asio::any_io_executor> executor)
{
    if (executor.has_value()) {
        ex_ = *executor;
        has_executor_ = true;
    }
}

HttpClient::~HttpClient()
{
    if (ctx_ != nullptr) {
        // 连接池 socket 收尾（curl_multi_cleanup 会经 close_socket_callback 清，这里兜底）
        static_cast<MultiCtx*>(ctx_)->sockets.clear();
        delete static_cast<MultiCtx*>(ctx_);
        ctx_ = nullptr;
    }
    while (!easy_pool_.empty()) {
        curl_easy_cleanup(static_cast<CURL*>(easy_pool_.front()));
        easy_pool_.pop_front();
    }
    if (multi_ != nullptr) {
        curl_multi_cleanup(static_cast<CURLM*>(multi_));
        multi_ = nullptr;
    }
}

asio::awaitable<Result<HttpResponse>> HttpClient::async_request(
    HttpRequest request, HttpRequestOptions options)
{
    if (!has_executor_) {
        ex_ = co_await asio::this_coro::executor;
        has_executor_ = true;
    }
    if (multi_ == nullptr) {
        CURLM* multi = curl_multi_init();
        curl_multi_setopt(multi, CURLMOPT_MAXCONNECTS, 8L);   // 连接池
        multi_ = multi;
        ctx_ = new MultiCtx();
    }

    // 合并 cookie jar（request 未显式指定 cookie 时）
    if (!cookie_string_.empty() && !request.cookie.has_value() && !request.cookie_file.has_value())
        request.cookie = cookie_string_;

    // easy 池：reset 清 option 但保留连接（连接在共享 multi 的 cpool），复用连接
    CURL* easy = nullptr;
    if (!easy_pool_.empty()) {
        easy = static_cast<CURL*>(easy_pool_.front());
        easy_pool_.pop_front();
    }

    Result<HttpResponse> response = co_await collect_response(
        ex_, request, std::move(options), static_cast<MultiCtx*>(ctx_),
        static_cast<CURLM*>(multi_), easy);

    // easy 还池：复用连接靠共享 multi 的连接池（easy 每次重建，option 无残留问题）
    if (easy != nullptr)
        easy_pool_.push_back(easy);

    // 响应 Set-Cookie 累积进 jar（简单 name=value 解析）
    if (response) {
        for (auto const& [name, value] : response->headers) {
            if (equals_ignore_case(name, "set-cookie")) {
                std::string_view cookie = value;
                std::size_t semi = cookie.find(';');
                std::string_view pair = cookie.substr(0, semi);
                std::size_t eq = pair.find('=');
                if (eq != std::string_view::npos) {
                    std::string key(pair.substr(0, eq));
                    std::string val(pair.substr(eq + 1));
                    if (!cookie_string_.empty())
                        cookie_string_ += "; ";
                    cookie_string_ += key + "=" + val;
                }
            }
        }
    }
    co_return response;
}

asio::awaitable<Result<void>> HttpClient::load_cookies(std::string path)
{
    std::ifstream file(path);
    if (!file)
        co_return std::unexpected(Error{ Errc::NotFound, "cookie file not readable: " + path });
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    // 去掉末尾换行 / 注释
    while (!content.empty() && (content.back() == '\n' || content.back() == '\r' || content.back() == ' '))
        content.pop_back();
    cookie_string_ = std::move(content);
    co_return Result<void>{};
}

asio::awaitable<Result<void>> HttpClient::save_cookies(std::string path)
{
    std::ofstream file(path);
    if (!file)
        co_return std::unexpected(Error{ Errc::NetworkError, "cookie file not writable: " + path });
    file << cookie_string_ << "\n";
    co_return Result<void>{};
}

// ───────────────────── HttpStreamReader ─────────────────────

struct HttpStreamReader::Impl {
    std::shared_ptr<CurlSession> session;
};

HttpStreamReader::HttpStreamReader(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
HttpStreamReader::HttpStreamReader(HttpStreamReader&& other) noexcept = default;
HttpStreamReader& HttpStreamReader::operator=(HttpStreamReader&& other) noexcept = default;

HttpStreamReader::~HttpStreamReader()
{
    if (impl_ != nullptr && impl_->session != nullptr)
        impl_->session->shutdown();
}

asio::awaitable<Result<HttpStreamReader>> HttpStreamReader::open(
    asio::any_io_executor ex, std::string_view url,
    nlohmann::json body, HttpRequestOptions options, std::string_view method)
{
    HttpRequest request;
    request.method = std::string(method);
    request.url = std::string(url);
    request.body = body.dump();
    request.content_type = "application/json";

    ensure_curl_global();
    Result<std::shared_ptr<CurlSession>> session =
        co_await open_with_retries(ex, request, std::move(options), nullptr, nullptr, nullptr);
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
            session->idle_expired = false;
            std::string chunk = std::move(session->body_chunks.front());
            session->body_chunks.pop_front();
            session->buffered_bytes -= chunk.size();
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
                co_return std::optional<std::string>{};
            co_return std::unexpected(
                map_curl_error(session->curl_result, session->error_buffer));
        }
        if (session->aborted)
            co_return std::unexpected(Error{Errc::NetworkError, "request cancelled"});
        if (session->idle_expired)
            co_return std::unexpected(Error{Errc::NetworkError, "idle timeout"});
        if (co_await coroutine_cancelled())
            co_return std::unexpected(Error{Errc::NetworkError, "request cancelled"});

        if (session->idle_timeout_ms > 0) {
            session->idle_timer.expires_after(
                std::chrono::milliseconds(session->idle_timeout_ms));
            session->idle_timer.async_wait(
                [weak = std::weak_ptr<CurlSession>(session)](asio::error_code const& ec) {
                    if (ec)
                        return;
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

namespace detail {

void SseParser::feed(std::string_view chunk)
{
    buffer_ += chunk;
}

std::optional<std::string> SseParser::next_event()
{
    std::size_t sep = std::string::npos;
    for (std::size_t i = 0; i + 1 < buffer_.size(); ++i) {
        char a = buffer_[i], b = buffer_[i + 1];
        if (a == '\n' && b == '\n') { sep = i + 2; break; }
        if (a == '\r' && b == '\r') { sep = i + 2; break; }
        if (a == '\n' && b == '\r' && i + 2 < buffer_.size() && buffer_[i + 2] == '\n') { sep = i + 3; break; }
    }
    if (sep == std::string::npos)
        return std::nullopt;

    std::string block = buffer_.substr(0, sep);
    buffer_.erase(0, sep);

    std::string data;
    std::size_t pos = 0;
    while (pos < block.size()) {
        std::size_t eol = block.find('\n', pos);
        if (eol == std::string::npos)
            eol = block.size();
        std::string line = block.substr(pos, eol - pos);
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.starts_with("data:")) {
            std::string value = line.substr(5);
            if (!value.empty() && value.front() == ' ')
                value.erase(value.begin());
            if (!data.empty())
                data += '\n';
            data += value;
        }
        pos = eol + 1;
    }
    return data;
}

}  // namespace detail

}   // namespace agent
