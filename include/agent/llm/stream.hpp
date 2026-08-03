#pragma once

// 异步事件通道（L1 传输基础）：AsyncStream 薄包装。
// 不依赖 LLM 类型层——任意 T 可用（Agent 层、LLM 事件流共用）。

#include <agent/core/result.hpp>

#include <asio.hpp>
#include <asio/experimental/basic_concurrent_channel.hpp>
#include <asio/experimental/channel_traits.hpp>

#include <cstddef>
#include <optional>
#include <utility>

namespace agent {

// ─────────────────────────────────────────────────────────────
// AsyncStream — 异步事件通道（薄包装）
// ─────────────────────────────────────────────────────────────

/// @brief 基于 asio::experimental::basic_concurrent_channel 的薄包装。
///        ⚠️ 不用 concurrent_channel 别名——它硬编码 channel_traits<>（缺
///        receive_cancelled/closed_signature），编译报错（L0-spike 已验证）。
///        必须显式 traits：
///        basic_concurrent_channel<any_io_executor,
///        channel_traits<void(error_code, T)>,
///        void(error_code, T)>
///        - 有界容量（默认 64）：消费慢时生产协程 async_send 挂起（背压），不无限积压
///        - **可共享**：内部 channel 用 shared_ptr 包装，AsyncStream 可拷贝——
///          生产端（move 进 stream_async）与消费端（SyncStreamBridge / 调用方）
///          各自持一份副本共享同一 channel，close 语义对全体副本生效
///        - close 语义：生产协程结束（Done/Error 后）close；消费端 receive 收到
///        channel_closed 错误即流结束
///        - 消费侧同步取：try_receive(handler)——handler 收 (error_code, value)，
///        非 try_receive(ec, value)（L0-spike 已验证）
template<typename T>
class AsyncStream {
public:
    using channel_traits_t = asio::experimental::channel_traits<void(asio::error_code, T)>;  ///< 通道 traits（receive 签名）
    using channel_t = asio::experimental::basic_concurrent_channel<asio::any_io_executor, channel_traits_t, void(asio::error_code, T)>;  ///< 有界并发通道类型

    /// @param executor 生产/消费协程的 io_context executor
    /// @param capacity 有界容量（背压上限）
    AsyncStream(asio::any_io_executor executor, std::size_t capacity = 64)
        : ch_(std::make_shared<channel_t>(std::move(executor), capacity))
    {
    }

    AsyncStream(AsyncStream const&) noexcept = default;   ///< 拷贝：共享同一 channel
    AsyncStream& operator=(AsyncStream const&) noexcept = default;  ///< 拷贝赋值：共享同一 channel
    AsyncStream(AsyncStream&&) noexcept = default;        ///< 移动：通道本体是 shared_ptr，移动即共享
    AsyncStream& operator=(AsyncStream&&) noexcept = default;       ///< 移动赋值

    /// @brief 生产端发送一个事件。channel 满则挂起（背压）。
    /// @param value 事件载荷
    /// @return true = 已入队；false = channel 已关闭（消费端提前结束，应停止生产）
    asio::awaitable<bool> send(T value)
    {
        // channel 的 async_send(error_code ec, T value, token)：第一个 error_code
        // 是发送方预置错误码（正常消息传默认），token 收操作结果 void(error_code)。
        auto [ec] = co_await ch_->async_send(asio::error_code(), std::move(value), asio::as_tuple(asio::use_awaitable));
        co_return !ec;
    }

    /// @brief 生产端结束：发送完毕后关闭通道。消费端 receive 收到 channel_closed 错误。
    void close() { ch_->close(); }

    /// @brief 通道是否仍开放（未关闭）。
    bool is_open() const { return ch_->is_open(); }

    /// @brief 消费端异步取一个事件。
    /// @return 成功 → 事件值；channel 关闭 → Error（调用方以此判断流结束）
    asio::awaitable<Result<T>> receive()
    {
        auto [ec, value] = co_await ch_->async_receive(asio::as_tuple(asio::use_awaitable));
        if (!ec)
            co_return Result<T>{ std::move(value) };
        co_return Result<T>{ std::unexpect, Error{ Errc::NetworkError, ec.message() } };
    }

    /// @brief 消费端同步取（非阻塞）。队列空 → nullopt；否则返回事件或关闭错误。
    std::optional<Result<T>> try_receive()
    {
        std::optional<Result<T>> out;
        ch_->try_receive([&out](asio::error_code ec, T value) {
            if (!ec)
                out = Result<T>{ std::move(value) };
            else
                out = Result<T>{ std::unexpect, Error{ Errc::NetworkError, ec.message() } };
        });
        return out;
    }

private:
    std::shared_ptr<channel_t> ch_;
};

}  // namespace agent
