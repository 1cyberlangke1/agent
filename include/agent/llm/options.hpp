#pragma once

// LLM 调用层请求配置（L1 配置层）。
// 包含：EndpointConfig（连接信息）/ Context（对话内容）/ StreamOptions（采样+传输+透传）。
// 不包含模型表（model.hpp）、类型层（types.hpp）、异步通道（stream.hpp）。

#include <agent/llm/content.hpp>
#include <agent/llm/types.hpp>
#include <agent/tools/tools.hpp>

#include <asio.hpp>

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace agent {

// ─────────────────────────────────────────────────────────────
// EndpointConfig — 厂商连接信息
// ─────────────────────────────────────────────────────────────

/// @brief 厂商连接信息。不绑定 model（model 是请求参数）。
struct EndpointConfig {
    std::string name;                   // "deepseek"
    std::string api_key;
    std::string base_url;               // "https://api.deepseek.com"
    /// 该厂商固定默认头：如 Anthropic 的 `anthropic-version`、
    /// 企业代理的 `Organization`/`Project`。StreamOptions.headers 在此之上覆盖。
    std::vector<std::pair<std::string, std::string>> default_headers;
};

// ─────────────────────────────────────────────────────────────
// Context — 对话内容（无采样参数）
// ─────────────────────────────────────────────────────────────

/// @brief 一次 LLM 调用的完整输入。
struct Context {
    std::string system_prompt;
    std::vector<Message> messages;
    std::vector<ToolInfo> tools;
};

// ─────────────────────────────────────────────────────────────
// StreamOptions — 公约数归一化 + extra 透传
// ─────────────────────────────────────────────────────────────

/// @brief 流式/非流式调用的采样、缓存、传输与透传配置。
struct StreamOptions {
    // ── 采样参数（公约数，三家都有）──
    std::optional<double> temperature;                 // 不传 → 不上传
    std::optional<int> max_tokens;                     // 不传 → 用 Model.max_output_tokens
    std::optional<ThinkingLevel> reasoning;            // 统一思考等级 → 各引擎映射

    // ── 缓存（公约数归一化，各引擎内部实现）──
    std::optional<CacheRetention> cache_retention;     // 映射到各家缓存机制
    std::optional<std::string> session_id;             // 会话关联，跨轮复用缓存

    // ── 传输层（都有）──
    std::optional<std::string> api_key;                // 覆盖 EndpointConfig.api_key
    std::optional<std::string> base_url;               // 覆盖 EndpointConfig.base_url
    std::vector<std::pair<std::string, std::string>> headers;     // 追加/覆盖请求头
    std::vector<std::string> suppress_headers;                     // 抑制默认头
    /// @brief 超时分层：单一整体超时会误杀合法长流（流式可跑数分钟）。
    int connect_timeout_ms = 30000;                    // 连接建立 + 收到首字节
    int idle_timeout_ms = 120000;                      // 流式块间静默上限（0 = 不限）
    int total_timeout_ms = 600000;                     // 整体上限，兜底（0 = 不限）
    int max_retries = 2;                               // HTTP 层重试（429/5xx）
    int max_retry_delay_ms = 60000;                    // Retry-After 超此值立即失败
    /// 取消信号。生命周期约定：指针指向的 signal 必须在本次调用完全结束
    /// （generator 析构 / awaitable 完成）之前保持有效，由调用方保证。
    asio::cancellation_signal* cancel = nullptr;

    // ── 原始响应捕获（可选，默认不存，零开销）──
    /// true 时 ChatResponse.raw 填充上游原始响应。
    bool capture_raw_response = false;
    /// capture_raw_response 开启时的原始响应字节上限（默认 1MB），超出丢弃 raw。
    size_t max_raw_bytes = 1 << 20;

    // ── 非公约数：透传接口，用户自己决定 ──
    /// 用户塞 store / metadata / provider 特有字段。
    /// 引擎不解释、不推断，原样并入请求体（provider 不认识的字段自行忽略或报错）。
    nlohmann::json extra;
};

}  // namespace agent
