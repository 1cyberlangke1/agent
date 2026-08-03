#pragma once

// L1 交互原语壳（域无关，按交互模式复用）：
//   StreamFacade<Engine> —— 流式域四接口壳（chat / 未来图像 partial / TTS 共用）
//   SyncStreamBridge<Event> —— 同步 stream() 的事件桥（单线程 io_context）
//
// 关键泛化：壳对引擎的**参数包完全透明**（完美转发），只依赖两个关联类型
// （event_type / result_type）和 as_done / as_error，加一个终结契约。
// 桥接/收集/取消逻辑写一次，所有流式能力域共用。
//
// 终结契约：引擎保证流以 Done 或 Error 事件收尾后 close channel；
// channel 关闭而两者皆缺（生产协程内部故障）→ 壳兜底合成 Error
// （complete 返回 Result 错误、stream 补发一个 Error 事件），绝不静默成功。

#include <agent/core/result.hpp>
#include <agent/llm/options.hpp>
#include <agent/llm/stream.hpp>

#include <asio.hpp>

#include <generator>
#include <optional>
#include <utility>

namespace agent::detail {

/// 流式引擎 concept：能力域无关的最小契约。
template<typename E>
concept StreamEngine = requires(typename E::event_type const& ev) {
    typename E::event_type;
    typename E::result_type;
    { E::as_done(ev) } -> std::same_as<std::optional<typename E::result_type>>;
    { E::as_error(ev) } -> std::same_as<std::optional<Error>>;
    requires std::constructible_from<typename E::event_type, Error>;   // 兜底合成 Error 事件
};

/// 同步事件桥：单线程 io_context + 共享 channel。
///
/// 线程模型（防死锁）：generator、io_context、生产协程同线程，普通有界
/// channel 即可，**不用 mutex/condition_variable**（同线程无并发；cv.wait
/// 在唯一线程上等待自己 notify 是自死锁）。消费侧 try_receive 非阻塞，
/// 队列空则 io.run_one() 推进网络直到生产协程投递事件或流结束。
///
/// 提前终止语义：用户 break 丢弃 generator → SyncStreamBridge 析构触发
/// cancellation signal 并 run 至生产协程退出（防悬空回调持有已亡 bridge），
/// 绝不重发请求。
template<typename Event>
class SyncStreamBridge {
public:
    SyncStreamBridge()
        : stream_(io_.get_executor(), 64)
    {
    }

    SyncStreamBridge(SyncStreamBridge const&) = delete;
    SyncStreamBridge& operator=(SyncStreamBridge const&) = delete;

    /// 生产协程的写入端（传入 engine_.stream_async 尾参）。
    AsyncStream<Event>& sink() { return stream_; }

    asio::io_context& io() { return io_; }

    /// 生产协程的取消信号（co_spawn 的 cancel 通道）。
    asio::cancellation_signal& cancel_signal() { return signal_; }

    /// 取下一事件；channel 关闭（生产协程结束）返回 nullopt。
    std::optional<Event> pop_after_io()
    {
        while (true) {
            std::optional<Result<Event>> item = stream_.try_receive();
            if (item) {
                if (*item)
                    return std::move(**item);
                return std::nullopt;   // channel closed
            }
            if (io_.stopped())
                return std::nullopt;
            io_.run_one();             // 推进网络直到有事件或流结束
        }
    }

    ~SyncStreamBridge()
    {
        if (io_.stopped())
            return;
        signal_.emit(asio::cancellation_type::all);
        stream_.close();               // 让生产协程的 send 失败返回
        io_.run();                     // 跑到生产协程退出，回收全部挂起回调
    }

private:
    asio::io_context io_;
    AsyncStream<Event> stream_;
    asio::cancellation_signal signal_;
};

/// 四接口壳：使用者经 Provider（如 OpenAIProvider）调用，无需直接接触。
///
/// 四接口语义：
///   stream(model, ctx, opts)        同步流式：返回 std::generator<StreamEvent>，
///                                   for 循环逐事件消费；提前 break 会取消并关闭请求
///   complete(model, ctx, opts)      同步非流式：阻塞收集完整响应（Result<ChatResponse>）
///   stream_async(model, ctx, opts, sink)   异步流式：生产协程推入调用方提供的 AsyncStream
///   complete_async(model, ctx, opts)       异步非流式：co_await 返回完整响应
///
/// 终结契约：流必然以 Done 或 Error 事件收尾；两者皆缺（内部故障）→ 壳兜底合成
/// Error，绝不静默成功。Done/Error 后请求即关闭。
template<StreamEngine Engine>
class StreamFacade {
public:
    explicit StreamFacade(EndpointConfig config)
        : engine_(std::move(config))
    {
    }

    /// 同步流式。generator 每次 resume 驱动 io_context（见 SyncStreamBridge）。
    template<typename... Args>
    std::generator<typename Engine::event_type> stream(Args const&... args);

    /// 同步非流式：跑 complete_async 直到返回完整结果。
    template<typename... Args>
    Result<typename Engine::result_type> complete(Args const&... args) const;

    /// 异步流式：生产协程推入调用方提供的 sink。按值移动进入，生命周期由引擎保证。
    template<typename... Args>
    asio::awaitable<void> stream_async(Args&&... args)
    {
        return engine_.stream_async(std::forward<Args>(args)...);
    }

    /// 异步非流式：建内部 sink + co_spawn 核心 + 收集 Done/Error。
    template<typename... Args>
    asio::awaitable<Result<typename Engine::result_type>> complete_async(Args const&... args) const;

private:
    /// mutable：complete / complete_async 为 const（Agent 的 Behaviors 钩子拿 Provider const&，
    /// 压缩要在 const 上调用 provider）。引擎 stream_async 非 const，靠 mutable 放行。
    mutable Engine engine_;
};

// ───────────────────── 实现 ─────────────────────

template<StreamEngine Engine>
template<typename... Args>
std::generator<typename Engine::event_type> StreamFacade<Engine>::stream(Args const&... args)
{
    using Event = typename Engine::event_type;
    SyncStreamBridge<Event> bridge;
    // 复制 sink（共享 channel）：生产协程持副本，bridge 保留自己的副本消费
    asio::co_spawn(bridge.io(),
                   engine_.stream_async(args..., bridge.sink()),
                   asio::bind_cancellation_slot(bridge.cancel_signal().slot(),
                                                asio::detached));

    bool emitted_terminal = false;
    while (std::optional<Event> event = bridge.pop_after_io()) {
        if (Engine::as_done(*event).has_value() || Engine::as_error(*event).has_value())
            emitted_terminal = true;
        co_yield std::move(*event);
    }
    // 终结契约：channel 关闭而 Done/Error 皆缺 → 壳兜底合成 Error，绝不静默成功。
    if (!emitted_terminal)
        co_yield Event{ Error{ Errc::NetworkError, "stream ended without terminal event" } };
}

template<StreamEngine Engine>
template<typename... Args>
Result<typename Engine::result_type> StreamFacade<Engine>::complete(Args const&... args) const
{
    using ResultType = typename Engine::result_type;
    asio::io_context io;
    std::optional<Result<ResultType>> outcome;
    asio::co_spawn(io, complete_async(args...),
        [&outcome](std::exception_ptr, Result<ResultType> result) {
            outcome.emplace(std::move(result));
        });
    io.run();
    // co_spawn 完成回调必然执行（io.run() 跑完全部工作），此处 outcome 非空。
    return std::move(*outcome);
}

template<StreamEngine Engine>
template<typename... Args>
asio::awaitable<Result<typename Engine::result_type>> StreamFacade<Engine>::complete_async(
    Args const&... args) const
{
    using Event = typename Engine::event_type;
    using ResultType = typename Engine::result_type;

    asio::any_io_executor executor = co_await asio::this_coro::executor;
    AsyncStream<Event> sink(executor);
    // 生产协程引用 args...（栈上 const&，complete_async 返回前仍有效）；
    // 复制 sink（共享 channel）：生产协程持副本，本协程用原对象 receive。
    asio::co_spawn(executor, [&]() -> asio::awaitable<void> {
        co_await engine_.stream_async(args..., sink);
    }, asio::detached);

    std::optional<ResultType> result;
    std::optional<Error> error;
    while (true) {
        Result<Event> received = co_await sink.receive();
        if (!received)
            break;   // channel closed = 生产协程已结束
        if (std::optional<ResultType> done = Engine::as_done(*received)) {
            result = std::move(done);
            break;
        }
        if (std::optional<Error> err = Engine::as_error(*received)) {
            error = std::move(err);
            break;
        }
    }
    // 终结契约：生产协程结束而 Done/Error 皆缺 → 合成错误，绝不静默成功。
    if (result.has_value())
        co_return std::move(*result);
    if (error.has_value())
        co_return Result<ResultType>{ std::unexpect, std::move(*error) };
    co_return Result<ResultType>{ std::unexpect,
                                  Error{ Errc::NetworkError, "stream ended without terminal event" } };
}

}  // namespace agent::detail
