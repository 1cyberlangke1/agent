#pragma once

// L0 传输原语（域无关）：
//   async_http_post       —— 非流式 POST（测试/未来接口用，complete 走流式）
//   HttpStreamReader      —— 流式响应读取器（pull 形态，open + next_chunk）
//   detail::SseParser     —— SSE 事件切分（data: 内容 join，引擎侧使用）
//
// 传输后端：libcurl multi_socket 模式嫁接 asio io_context——零线程桥、
// 单事件循环，socket 由 asio 创建（CURLOPT_OPENSOCKETFUNCTION）并以
// async_wait 驱动 curl_multi_socket_action。C API 全部封在 .cpp 的
// detail 实现内，不泄漏到本头文件。
// TLS：Windows 走 schannel（系统证书库，无需手动导入）；其他平台 OpenSSL。
//
// L0 域无关硬约束：本层不认识 StreamEvent/ChatResponse 等 chat 类型，
// 引擎协程拿「原始 body 字节」，自行喂 SseParser → 推域事件。

#include <agent/core/result.hpp>

#include <asio.hpp>

#include <nlohmann/json.hpp>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace agent {

/// @brief 非流式响应。
struct HttpResponse {
    int status = 0;
    std::string body;
    std::vector<std::pair<std::string, std::string>> headers;
};

/// @brief 请求配置。engine 负责合并（EndpointConfig.default_headers + StreamOptions.headers
///        − suppress_headers），认证头（Bearer / x-api-key）也由 engine 放进 headers——
///        http 层不理解认证语义（Gemini 的 ?key= 由 engine 拼进 URL）。
struct HttpRequestOptions {
    std::vector<std::pair<std::string, std::string>> headers;
    /// 连接建立超时（curl CONNECTTIMEOUT_MS）。
    int connect_timeout_ms = 30000;
    /// 流式块间静默上限（curl LOW_SPEED_TIME 近似，秒粒度取整；0 = 不限）。
    int idle_timeout_ms = 120000;
    /// 整体上限（curl TIMEOUT_MS；0 = 不限）。
    int total_timeout_ms = 600000;
    /// 首字节前重试次数（429/5xx/连接失败）。首字节后绝不重试。
    int max_retries = 2;
    /// Retry-After 超过此值立即放弃重试。
    int max_retry_delay_ms = 60000;
    /// 取消信号槽。空 slot = 不可取消。emit 可来自任意线程（内部 post 跳转）。
    asio::cancellation_slot cancel;
};

/// @brief 非流式 POST。返回完整响应体。仅供测试与未来接口（embeddings 等）；
///        生产 complete 走流式收集，不发非流式请求。
///
///        错误语义：任何 HTTP 状态（含 4xx/5xx）都算成功返回（status/body 可查，
///        错误详情 JSON 归引擎解析）；Result 错误仅代表传输层失败
///        （连不上 / 超时 / 取消 / 重试耗尽仍连不上）。
/// @param ex      io_context executor（调用协程须运行在其上）
/// @param url     完整 URL（http/https）
/// @param body    请求体（JSON，序列化后以 application/json 发送）
/// @param options 请求配置（头/超时/重试/取消）
asio::awaitable<Result<HttpResponse>> async_http_post(
    asio::any_io_executor ex, std::string_view url,
    nlohmann::json body, HttpRequestOptions options);

/// @brief 流式响应逐块读取器（pull 形态）。**L0 域无关**：http 层不认识 chat 类型，
///        引擎协程 co_await 逐块拿「原始 body 字节」，自行解析。
///        对象持有传输会话（curl multi/easy + asio socket），仅可移动（pimpl）。
class HttpStreamReader {
public:
    /// @brief 连接 + 发请求 + 收响应头。首字节前重试（连接失败/429/5xx，含
    ///        Retry-After 与指数退避）全部发生在 open 内部；open 成功 = 已拿到
    ///        最终响应头（1xx 中间响应已被跳过）。
    ///        4xx/5xx（重试耗尽后）也算 open 成功——status()/body 可读，
    ///        错误详情归引擎；Result 错误仅传输层失败。
    static asio::awaitable<Result<HttpStreamReader>> open(
        asio::any_io_executor ex, std::string_view url,
        nlohmann::json body, HttpRequestOptions options);

    /// @brief 下一块 body 字节（chunked 已由 curl 解码）；nullopt = 流正常结束。
    ///        首字节后的错误（断流 / idle 超时 / 取消）→ Result 错误，绝不重试。
    asio::awaitable<Result<std::optional<std::string>>> next_chunk();

    /// @brief 最终响应状态码（open 成功后有效）。
    int status() const;

    /// @brief 全部响应头（open 成功后有效；1xx 中间响应的头已被最终响应覆盖）。
    std::vector<std::pair<std::string, std::string>> const& headers() const;

    /// @brief 响应头取值（open 成功后有效；名字大小写不敏感）；不存在返回空。
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

/// @brief SSE 事件切分器。
///
///        流式喂字节，产出完整 SSE 事件（data: 行内容，多行 data: 用 \n join）。
///        注释行（以 : 开头）跳过；空行触发一个事件输出。
///        事件名（event:）暂不解析——当前三家都是纯 data: JSON。
class SseParser {
public:
    /// @brief 喂入一段字节。
    void feed(std::string_view chunk);
    /// @brief 取出一个完整事件；尚无完整事件返回 nullopt（需继续 feed）。
    std::optional<std::string> next_event();

private:
    std::string buffer_;
};

}  // namespace detail

}  // namespace agent
