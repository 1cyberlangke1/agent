#pragma once

#include <expected>
#include <string>

namespace agent {

/// @brief 错误码
///
/// 覆盖工具注册、执行、Provider 调用三层所有可能错误。
enum class Errc {
    /// 工具不存在
    NotFound,
    /// 工具名重复注册
    Duplicate,
    /// 参数校验不通过（JSON 类型不匹配等）
    InvalidArgs,
    /// 工具执行内部出错
    ExecutionFailed,
    /// 网络请求失败（连接超时、DNS 错误等）
    NetworkError,
    /// 上游限流（HTTP 429）
    RateLimited,
    /// 认证鉴权失败（API key 无效等）
    AuthError,
    /// Provider 返回服务端错误
    ProviderError,
    /// MCP 服务器未连接
    ServerNotConnected,
    /// 压缩失败（摘要响应调工具 / 流中断 / 空摘要 / 请求超限），message 写明具体原因
    CompactionFailed,
    /// 用户取消 / 超时中止（abort() 触发，AgentError 用它区分取消与失败）
    Aborted,
};

/// @brief 错误信息
///
/// @param code    错误码，用于程序做分支判断（重试 / 跳过 / 报错）
/// @param message 上游原始详情文本，如 HTTP 429 的 Retry-After 提示，
///                用于日志记录或透传给调用方。
struct Error {
    Errc code;
    std::string message;
};

/// @brief 通用结果类型。
///
/// 零异常替代方案。函数返回值要么携带正常结果 T，
/// 要么携带 Error（含错误码 + 详情文本）。
///
/// 建议函数签名加 [[nodiscard]] 防止返回值被忽略。
///
/// @tparam T 成功时返回值的类型
template<typename T>
using Result = std::expected<T, Error>;

} // namespace agent
