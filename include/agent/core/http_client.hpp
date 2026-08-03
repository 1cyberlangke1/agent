#pragma once

// L0 传输层 + 完整 curl 封装（纯协程，零线程）：
//
//   a) 传输原语（LLM 引擎 / 流式用）：async_http_post / async_http_get / HttpStreamReader。
//   b) 通用请求原语：HttpRequest（curl 能力全映射：方法 / body / multipart / 流式上传 /
//      auth / TLS / 重定向 / 代理 / cookie）+ async_http_request / sync_http_request。
//   c) 便捷层（工具友好，非 2xx → Error）：http_get_json / http_post_json / http_post_form /
//      http_upload_file。
//   d) HttpClient（状态化，协程）：easy 句柄池连接复用（keep-alive）+ cookie jar。
//
// 传输后端：libcurl multi_socket 模式嫁接 asio io_context——零线程桥、单事件循环，
// socket 由 asio 创建（CURLOPT_OPENSOCKETFUNCTION）并以 async_wait 驱动
// curl_multi_socket_action。C API 全部封在 .cpp 的 detail 实现内。
// TLS：Windows 走 schannel（系统证书库）；其他平台 OpenSSL。
//
// L0 域无关硬约束：本层不认识 StreamEvent/ChatResponse 等 chat 类型，
// 引擎协程拿「原始 body 字节」，自行喂 SseParser → 推域事件。
//
// 纯协程不变式（HttpClient 同样适用）：所有 curl 回调只在绑定 executor 上执行；
// HttpClient 非线程安全，须由单一 io_context（或 strand）驱动。

#include <agent/core/result.hpp>

#include <asio.hpp>

#include <nlohmann/json.hpp>

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace agent {

// ───────────────────── 请求配置（curl 能力全映射）─────────────────────

/// 认证方式（对应 curl CURLAUTH_*；Bearer 走 Authorization 头）。
enum class HttpAuth { None, Basic, Bearer, Digest, Ntlm, Negotiate };

/// 代理类型（对应 curl CURLPROXY_*）。
enum class HttpProxyType { Http, Https, Socks4, Socks5 };

/// multipart 表单的一个 part：value（文本）/ buffer（内存字节）/ file_path（文件）三选一。
struct MultipartPart {
    std::string name;
    std::optional<std::string> value;
    std::optional<std::string> buffer;
    std::optional<std::string> file_path;
    std::optional<std::string> content_type;
    std::optional<std::string> filename;
};

/// 一次 HTTP 请求「发什么 + 怎么发」。
struct HttpRequest {
    std::string method = "GET";                         ///< GET/POST/PUT/PATCH/DELETE/HEAD/OPTIONS/任意
    std::string url;
    std::vector<std::pair<std::string, std::string>> query;    ///< 自动 url_encode 拼接
    std::vector<std::pair<std::string, std::string>> headers;

    ///< body 三选一：字符串（+ content_type）/ multipart / 大文件流式上传
    std::optional<std::string> body;
    std::optional<std::string> content_type;
    std::vector<MultipartPart> multipart;
    std::optional<std::string> upload_file;

    ///< 认证
    HttpAuth auth = HttpAuth::None;
    std::optional<std::string> auth_credentials;        ///< "user:pass"（Basic/Digest/Ntlm/Negotiate）

    // TLS
    bool verify_tls = true;
    std::optional<std::string> ca_file;
    std::optional<std::string> client_cert;
    std::optional<std::string> client_key;
    std::optional<std::string> client_key_password;

    ///< 重定向
    bool follow_redirects = false;
    int max_redirects = 5;
    bool auto_referer = false;

    ///< 代理
    std::optional<std::string> proxy;
    HttpProxyType proxy_type = HttpProxyType::Http;
    std::optional<std::string> proxy_auth;

    ///< Cookie
    std::optional<std::string> cookie;                  ///< 一次性 "k=v; k2=v2"
    std::optional<std::string> cookie_file;             ///< 读 jar 文件（空串 = 内存 jar）
    std::optional<std::string> cookie_jar;              ///< 响应后写 jar 文件
    bool cookie_session = false;
};

// ───────────────────── 非流式响应 ─────────────────────

/// 一次非流式响应。任何 HTTP 状态码都算成功返回（status/body 可查）；
/// Result 错误仅代表传输层失败（连不上 / 超时 / 取消 / 重试耗尽）。
class HttpResponse {
public:
    int status = 0;
    std::string body;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string effective_url;          ///< 重定向后最终 URL（CURLINFO_EFFECTIVE_URL）
    double total_time_seconds = 0;      ///< 本次传输耗时（CURLINFO_TOTAL_TIME）

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

    bool ok() const { return status / 100 == 2; }

    /// 解析 body 为 JSON；失败 → Errc::ParseError。
    Result<nlohmann::json> json() const;
};

// ───────────────────── 传输 I/O 配置（超时 / 重试 / 取消）─────────────────────

struct HttpRequestOptions {
    std::vector<std::pair<std::string, std::string>> headers;
    int connect_timeout_ms = 30000;
    int idle_timeout_ms = 120000;
    int total_timeout_ms = 600000;
    int max_retries = 2;
    int max_retry_delay_ms = 60000;
    asio::cancellation_slot cancel;
};

// ───────────────────── 通用请求原语 ─────────────────────

/// 异步通用请求（任意方法 / body 形态 / auth / TLS / 重定向 / 代理 / cookie）。
/// 错误语义：传输失败 → Result 错误；任何 HTTP 状态码都算成功返回（status/body 可查）。
asio::awaitable<Result<HttpResponse>> async_http_request(
    asio::any_io_executor ex, HttpRequest const& request, HttpRequestOptions options);

/// 同步通用请求（同步壳包 async 核心，内部自建 io_context）。
Result<HttpResponse> sync_http_request(HttpRequest const& request, HttpRequestOptions options);

// ───────────────────── 便捷层（工具友好：非 2xx → Error）─────────────────────

/// 错误归一化：429 → RateLimited；401/403 → AuthError；其他非 2xx → ProviderError
/// （message 带状态码 + body 前 512 字节）；body 解析失败 → ParseError。
Result<nlohmann::json> http_get_json(std::string url, HttpRequestOptions options = {});
Result<nlohmann::json> http_post_json(std::string url, nlohmann::json body,
                                      HttpRequestOptions options = {});
Result<nlohmann::json> http_post_form(
    std::string url, std::vector<std::pair<std::string, std::string>> fields,
    HttpRequestOptions options = {});
/// 大文件流式上传（multipart 单 part：file part）。
Result<void> http_upload_file(std::string url, std::string file_path,
                              HttpRequestOptions options = {});

asio::awaitable<Result<nlohmann::json>> async_http_get_json(
    asio::any_io_executor ex, std::string url, HttpRequestOptions options = {});
asio::awaitable<Result<nlohmann::json>> async_http_post_json(
    asio::any_io_executor ex, std::string url, nlohmann::json body,
    HttpRequestOptions options = {});

// ───────────────────── URL 工具 ─────────────────────

/// UTF-8 百分号编码（查询参数名/值、路径段）。
std::string url_encode(std::string_view s);

/// 追加查询参数（自动编码；url 已带 ? 时用 &）。
std::string append_query(std::string url,
                         std::vector<std::pair<std::string, std::string>> params);

///< ───────────────────── 非流式 POST / GET（向后兼容）─────────────────────

asio::awaitable<Result<HttpResponse>> async_http_post(
    asio::any_io_executor ex, std::string_view url,
    nlohmann::json body, HttpRequestOptions options);

asio::awaitable<Result<HttpResponse>> async_http_get(
    asio::any_io_executor ex, std::string_view url, HttpRequestOptions options);

Result<HttpResponse> sync_http_get(std::string_view url, HttpRequestOptions options);
Result<HttpResponse> sync_http_post(std::string_view url, nlohmann::json body, HttpRequestOptions options);

// ───────────────────── HttpClient（纯协程：连接复用 + cookie jar）─────────────────────

/// 状态化 HTTP 客户端。纯协程、零线程：executor 懒绑（co_await this_coro::executor）
/// 或构造传入。持有**常驻 curl multi 句柄**（连接池 CURLMOPT_MAXCONNECTS）——curl 8.x
/// 连接缓存挂在 multi 上，跨请求存活，同一 host 后续请求走 keep-alive 复用。
/// cookie jar 也随客户端持久。
/// ⚠️ 单 executor 驱动（与 CurlSession 同不变式），非线程安全，须由单一 io_context
///    （或 strand）驱动；多次 async_request 需在同一 executor 上跑（且串行——单活跃请求）。
class HttpClient {
public:
    /// @param executor 可选：绑定客户端 executor（懒绑 = 首个 async_request 的 this_coro）。
    explicit HttpClient(std::optional<asio::any_io_executor> executor = std::nullopt);
    ~HttpClient();

    HttpClient(HttpClient const&) = delete;
    HttpClient& operator=(HttpClient const&) = delete;

    /// 异步请求：加 easy 到常驻 multi（连接池复用）→ 装配跑完 → 移除。
    /// 内部合并本客户端的 cookie jar（request 未显式设 cookie/cookie_file 时）。
    asio::awaitable<Result<HttpResponse>> async_request(
        HttpRequest request, HttpRequestOptions options = {});

    /// cookie jar 持久化：load 读入内存 jar（"k=v; k2=v2" 文本）；save 写出。
    asio::awaitable<Result<void>> load_cookies(std::string path);
    asio::awaitable<Result<void>> save_cookies(std::string path);

private:
    asio::any_io_executor ex_;
    bool has_executor_ = false;
    void* multi_ = nullptr;        ///< 常驻 CURLM*（连接池 CURLMOPT_MAXCONNECTS）
    void* ctx_ = nullptr;          ///< detail::MultiCtx*（单活跃请求槽）
    std::deque<void*> easy_pool_;  ///< easy 句柄池（curl_easy_reset 清 option、保留连接——连接在 multi）
    std::string cookie_string_;    ///< 客户端维护的 cookie jar（响应 Set-Cookie 累积）
};

// ───────────────────── 流式响应读取器（保留）─────────────────────

class HttpStreamReader {
public:
    static asio::awaitable<Result<HttpStreamReader>> open(
        asio::any_io_executor ex, std::string_view url,
        nlohmann::json body, HttpRequestOptions options,
        std::string_view method = "POST");
    asio::awaitable<Result<std::optional<std::string>>> next_chunk();
    int status() const;
    std::vector<std::pair<std::string, std::string>> const& headers() const;
    std::string_view header(std::string_view name) const;
    HttpStreamReader(HttpStreamReader&& other) noexcept;
    HttpStreamReader& operator=(HttpStreamReader&& other) noexcept;
    HttpStreamReader(HttpStreamReader const&) = delete;
    HttpStreamReader& operator=(HttpStreamReader const&) = delete;
    ~HttpStreamReader();

private:
    struct Impl;
    explicit HttpStreamReader(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

namespace detail {

/// SSE 事件切分器（引擎侧使用）。
class SseParser {
public:
    void feed(std::string_view chunk);
    std::optional<std::string> next_event();
private:
    std::string buffer_;
};

}  // namespace detail

}  // namespace agent
