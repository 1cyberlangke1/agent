#pragma once

// 高层 Agent 封装（对齐原版 pi packages/agent 全部 agent 功能，零裁剪）：
//   Agent<Provider, Behaviors> —— 引擎循环 + 可覆写行为（钩子 + 压缩策略一体）。
//   DefaultBehaviors —— 默认钩子 + 压缩策略；override 用「非虚名字隐藏」编译期决议，零 vtable。
//
// 双层循环（对齐 pi 〇.4）：
//   内层 turn 循环：steering 注入 → 流式收集回复 → 提取工具调用 → 执行（before/after 钩子）
//     → 结果回传 → turn_end → prepareNextTurn → shouldStop → 取 steering，直到无工具调用；
//   外层：follow-up 队列非空 → 继续；否则 agent_end。
// 压缩：**run 内防爆**（每轮发请求前 should_compact / compact，超限才压）+ **外部主动压缩**
//   （agent.should_compact() / agent.compact()），共用同一可覆写策略。
//
// ⚠️ 覆盖约束：override 的钩子必须是模板方法（template<typename Provider>），签名与默认一致，
//   不一致会「隐藏同名但签名不同」静默编译错。不用 CRTP——Agent 持有具体类型，名字隐藏已保证编译期决议。
// ⚠️ Agent 不可拷贝 / 不可移动：async 协程（run_async / compact_async / wait_for_idle）引用内部
//   provider_ / behaviors_，Agent 生命周期必须覆盖调用期间。
// ⚠️ 并发：不支持并发 run（一次一个 run 在途）；abort() 可从其他线程调用（最佳努力，见 cancelled()）。

#include <agent/agent/agent_event.hpp>
#include <agent/agent/compaction.hpp>
#include <agent/core/result.hpp>
#include <agent/llm/content.hpp>
#include <agent/llm/model.hpp>
#include <agent/llm/options.hpp>
#include <agent/llm/stream.hpp>
#include <agent/llm/stream_facade.hpp>
#include <agent/llm/types.hpp>
#include <agent/tools/tools.hpp>

#include <asio.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <generator>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace agent {

/// 每轮结束可替换的运行时状态（对齐 pi AgentLoopTurnUpdate）。
struct NextTurnUpdate {
    std::optional<ModelView> model;                 ///< 换模型（ModelRegistry 取）
    std::optional<ThinkingLevel> reasoning;         ///< 换思考档
    std::optional<std::vector<Message>> context;    ///< 替换发给模型的上下文
};

/// run 开始前可改写的初始状态（对齐 pi BeforeAgentStartResult）。
struct RunStartPatch {
    std::vector<Message> messages;      ///< 改写后的初始消息
    std::string system_prompt;          ///< 改写后的系统提示
};

/// 队列模式（对齐 pi QueueMode）：steer / follow_up 注入时的取法。
enum class QueueMode { OneAtATime, All };

/// 上下文快照：Agent 打包的当前上下文信息，统一传给所有 Behaviors 方法
///（钩子 / 压缩判断都能拿到 usage / 上下文窗口 / 上次摘要 / 当前对话 / 模型）。
/// ⚠️ messages 是引用：压缩会丢弃旧段，要旧上下文请在压缩前自行读取。
struct ContextSnapshot {
    Usage usage;                          ///< 当前上下文用量（真实锚点 + 估算补尾，cost 已算）
    int context_window;                   ///< 模型上下文上限
    std::string_view system_prompt;       ///< 当前系统提示
    std::string_view previous_summary;    ///< 上次摘要文本（增量更新用）
    ModelView model;                      ///< 当前模型
    std::vector<Message> const& messages; ///< 当前对话
};

/// 默认行为：钩子 + 压缩策略统一在一个对象。
/// 未 override 的成员自动用默认行为——想扩展哪个就只写哪个。
class DefaultBehaviors {
public:
    CompactionSettings settings_;         ///< 压缩策略阈值

    // ── 压缩策略（同一对象，钩子可直接调 should_compact / compact）──
    /// 判断：当前上下文是否该压缩。
    bool should_compact(ContextSnapshot const& snap) const
    {
        return agent::should_compact(estimate_context_tokens(snap.messages),
                                     snap.context_window, settings_);
    }

    /// 执行压缩：判断超限 → 选切点 → 摘要生成一体（带 Provider 调 Agent 的模型）。
    /// Result<std::optional<std::vector<Message>>>：
    ///   Error = 压缩失败（Errc::CompactionFailed + 原因：调工具 / 流中断 / 空摘要 / 请求超限）；
    ///   值 nullopt = 不压；有值 = 新对话 = [摘要消息] + [尾部保留段]，旧段直接丢弃。
    /// 摘要请求：指令作尾部 user 消息、system 不变、Context.tools 留空、<previous-summary> 增量。
    template<typename Provider>
    Result<std::optional<std::vector<Message>>> compact(Provider const& provider,
                                                        ContextSnapshot const& snap) const
    {
        std::optional<SummaryRequest> request = build_summary_request(
            snap.messages, snap.model, settings_, snap.system_prompt, snap.previous_summary);
        if (!request)
            return std::nullopt;
        Result<ChatResponse> response = provider.complete(snap.model, request->ctx, request->opts);
        if (!response)
            return std::unexpected(Error{ Errc::CompactionFailed,
                                          "压缩摘要请求失败: " + response.error().message });
        Result<std::string> summary = validate_summary_response(*response);
        if (!summary)
            return std::unexpected(summary.error());
        return apply_summary(snap.messages, request->cut, *summary);
    }

    // ── 钩子：每次发请求前改写消息列表（对齐 pi transformContext）──
    /// 改写发给模型的消息列表（可注入检索结果 / 拼 skills / 删敏感消息 / 截断历史）。
    /// 返回的列表就是模型这次请求看到的内容；不覆盖则原样返回 snap.messages。
    template<typename Provider>
    Result<std::vector<Message>> transform_context(Provider const& provider,
                                                   ContextSnapshot const& snap) const
    { (void)provider; return snap.messages; }
    /// 每个工具执行前拦截（对齐 pi beforeToolCall）；false = 拒绝执行，模型收到「工具被拒」。
    template<typename Provider>
    Result<bool> before_tool_call(Provider const& provider, ToolCall const& tool_call,
                                  ContextSnapshot const& snap) const
    { (void)provider; (void)tool_call; (void)snap; return true; }
    /// 每个工具执行后改写结果（对齐 pi afterToolCall，字段级覆盖，模型信什么你说了算）。
    template<typename Provider>
    Result<ToolResult> after_tool_call(Provider const& provider, ToolCall const& tool_call,
                                       ToolResult result) const
    { (void)provider; (void)tool_call; return result; }
    /// 每轮结束换模型 / 改上下文（对齐 pi prepareNextTurn）；nullopt = 保持现状 —— 多模型编排。
    template<typename Provider>
    Result<std::optional<NextTurnUpdate>> prepare_next_turn(
        Provider const& provider, Message const& assistant_message,
        std::vector<ToolResult> const& tool_results) const
    { (void)provider; (void)assistant_message; (void)tool_results; return std::nullopt; }
    /// 每轮结束提前停止（对齐 pi shouldStopAfterTurn）—— 成本/上下文控制。
    template<typename Provider>
    Result<bool> should_stop(Provider const& provider, Message const& assistant_message) const
    { (void)provider; (void)assistant_message; return false; }
    /// 动态取 API key（对齐 pi getApiKey）—— 短时 OAuth token 场景；非空则覆盖本次请求 key。
    Result<std::optional<std::string>> get_api_key(std::string_view provider_name) const
    { (void)provider_name; return std::nullopt; }

    // ── 低层时机点（对齐 pi before_agent_start / before_provider_request）──
    /// run 开始前：改写初始消息 + 系统提示；nullopt = 不改。
    template<typename Provider>
    Result<std::optional<RunStartPatch>> run_start(
        Provider const& provider, std::vector<Message> const& user_messages,
        std::string const& system_prompt) const
    { (void)provider; (void)user_messages; (void)system_prompt; return std::nullopt; }
    /// 每次发请求前：改写 StreamOptions（max_tokens / 超时 / 头等）；nullopt = 不改。
    template<typename Provider>
    Result<std::optional<StreamOptions>> before_request(
        Provider const& provider, StreamOptions const& options) const
    { (void)provider; (void)options; return std::nullopt; }
    /// 每次发请求前：改写请求体 body（对齐 pi beforeProviderPayload）。
    /// 收的是引擎 build_params 生成的完整 body，在其上改写（其余字段保留）；nullopt = 不改。
    template<typename Provider>
    Result<std::optional<nlohmann::json>> before_payload(
        Provider const& provider, nlohmann::json const& body) const
    { (void)provider; (void)body; return std::nullopt; }
};

template<typename Provider, typename Behaviors = DefaultBehaviors>
class Agent {
public:
    /// 构造：连接配置 + 模型 + 行为（钩子 + 压缩策略一体）+ 系统提示。
    /// 默认不开放任何工具——模型看不到也不会执行任何工具，须 set_tools 显式指定
    ///（广告子集 + 执行门控双重生效，未开放的调用会被拒绝回传）。
    Agent(EndpointConfig config, ModelView model,
          Behaviors behaviors = {}, std::string system_prompt = {})
        : provider_name_(config.name)
        , provider_(std::move(config))
        , model_(model)
        , system_prompt_(std::move(system_prompt))
        , behaviors_(std::move(behaviors))
    {
    }

    /// 不可拷贝 / 不可移动：async 协程引用内部 provider_ / behaviors_，
    /// Agent 生命周期必须覆盖调用期间（co_spawn 前先确保 agent 存活）。
    Agent(Agent const&) = delete;
    Agent& operator=(Agent const&) = delete;
    Agent(Agent&&) = delete;
    Agent& operator=(Agent&&) = delete;

    // ── 运行 ──
    /// 同步 run（= prompt）：返回事件 generator，for 循环消费；不遍历 = 不执行。
    /// 提前 break → generator 析构 → SyncStreamBridge 析构触发取消（见文件头 ⚠️）。
    std::generator<AgentEvent> run(std::vector<Message> user_messages,
                                   StreamOptions const& opts = {})
    {
        detail::SyncStreamBridge<AgentEvent> bridge;
        aborted_.store(false);   // 每次 run 重置取消标志
        cancel_.emplace();       // 每次 run 新建取消信号
        // ⚠️ loop 不绑取消槽（cancellation_signal 单槽，槽留给生产协程独占）；
        //    取消靠 aborted_ 标志检查 + 生产协程关流（llm.close()）触发 receive 返回。
        asio::co_spawn(bridge.io(),
                       loop_async(std::move(user_messages), bridge.sink(), opts),
                       asio::detached);
        while (std::optional<AgentEvent> event = bridge.pop_after_io())
            co_yield std::move(*event);
    }

    /// 异步 run：事件推到调用方提供的 sink（共享通道可拷贝）。
    asio::awaitable<void> run_async(std::vector<Message> user_messages,
                                    AsyncStream<AgentEvent> sink,
                                    StreamOptions const& opts = {})
    {
        auto ex = co_await asio::this_coro::executor;
        aborted_.store(false);
        cancel_.emplace();
        auto [ec] = co_await asio::co_spawn(ex,
            loop_async(std::move(user_messages), std::move(sink), opts),
            asio::as_tuple(asio::use_awaitable));
        (void)ec;   // 取消以 Aborted 形式由 loop 内 fail_run 处理
    }

    /// 续跑（最后一条消息须是 user 或 toolResult）。
    std::generator<AgentEvent> continue_run(StreamOptions const& opts = {})
    {
        return run({}, opts);
    }

    /// 续跑（异步）。
    asio::awaitable<void> continue_async(AsyncStream<AgentEvent> sink,
                                         StreamOptions const& opts = {})
    {
        co_await run_async({}, std::move(sink), opts);
    }

    // ── 队列 steering（对齐 pi steer / followUp）：运行中注入消息 ──
    /// 插队：本轮 assistant 答完就注入（干预下一轮）。
    void steer(Message message)
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        steer_queue_.push_back(std::move(message));
    }
    /// 排队：agent 本来要停时才注入（等干完再追加）。
    void follow_up(Message message)
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        follow_up_queue_.push_back(std::move(message));
    }
    /// 清空 steer + follow_up 两条队列（未注入的消息丢弃）。
    void clear_queues()
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        steer_queue_.clear();
        follow_up_queue_.clear();
    }
    /// 任一条队列还有未注入的消息？
    bool has_queued_messages() const
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return !steer_queue_.empty() || !follow_up_queue_.empty();
    }
    void set_steering_mode(QueueMode mode) { steering_mode_ = mode; }
    void set_follow_up_mode(QueueMode mode) { follow_up_mode_ = mode; }

    // ── 状态 ──
    /// 当前对话（压缩会丢弃旧段；要旧上下文请在压缩前自行读取）。
    std::vector<Message> const& messages() const { return messages_; }
    std::string const& system_prompt() const { return system_prompt_; }
    /// 当前模型（prepare_next_turn / set_model 换过之后从这取）。
    ModelView model() const { return model_; }
    void set_model(ModelView model) { model_ = model; }
    /// 换思考档（下一轮生效）。
    void set_reasoning(ThinkingLevel level) { reasoning_ = level; }
    /// 当前是否在 run（async 调用期间）。
    bool is_streaming() const { return streaming_.load(); }
    /// 正在执行的工具 id（外部/UI 查询）。
    std::vector<std::string> pending_tools() const { return pending_tools_; }
    /// 最近失败/中止错误（无则 nullopt）。
    std::optional<Error> last_error() const { return last_error_; }
    /// 传给模型的工具子集（**默认空 = 不开放任何工具**；指定后广告 + 执行门控同时生效，
    /// 未开放的工具调用会被拒绝回传给模型）。
    void set_tools(std::vector<std::string> tools) { tools_ = std::move(tools); }
    std::vector<std::string> const& tools() const { return tools_; }

    // ── 生命周期 ──
    /// 取消当前 run（AgentError 带 Errc::Aborted）。可重置：下次 run 清标志 + 重建取消信号。
    void abort()
    {
        aborted_.store(true);
        if (cancel_)
            cancel_->emit(asio::cancellation_type::all);
    }
    /// 等当前 run 彻底结束。
    asio::awaitable<void> wait_for_idle()
    {
        auto ex = co_await asio::this_coro::executor;
        while (streaming_.load()) {
            asio::steady_timer timer(ex, std::chrono::milliseconds(5));
            // as_tuple：本项目生产零异常，裸 use_awaitable 在取消时会抛 operation_aborted
            co_await timer.async_wait(asio::as_tuple(asio::use_awaitable));
        }
    }
    /// 清空对话 + 队列 + 状态。
    void reset()
    {
        messages_.clear();
        steer_queue_.clear();
        follow_up_queue_.clear();
        pending_tools_.clear();
        last_error_.reset();
        last_usage_.reset();
        previous_summary_.clear();
        cumulative_cost_ = 0;
    }

    // ── 工具执行模式（对齐 pi toolExecution）：Agent 全局默认 ──
    void set_tool_execution_mode(ToolExecutionMode mode) { tool_execution_mode_ = mode; }

    // ── 压缩 ──
    /// 判断：该不该压（外部决定主动时机，无参）。
    bool should_compact() const { return behaviors_.should_compact(build_snapshot()); }
    /// 执行压缩（同步，摘要走 provider.complete）。
    /// true = 成功且 messages_ 已替换（旧段丢弃）；false = 没压（未触发）；
    /// Error = 压缩失败（含原因）。
    Result<bool> compact()
    {
        ContextSnapshot snap = build_snapshot();
        Result<std::optional<std::vector<Message>>> r = behaviors_.compact(provider_, snap);
        if (!r)
            return std::unexpected(r.error());
        if (!*r)
            return false;   // 未触发
        messages_ = std::move(**r);
        update_previous_summary();
        return true;
    }
    /// 执行压缩（异步，摘要走 provider.complete_async）。
    asio::awaitable<Result<bool>> compact_async()
    {
        std::optional<SummaryRequest> request = build_summary_request(
            messages_, model_, behaviors_.settings_, system_prompt_, previous_summary_);
        if (!request)
            co_return false;
        Result<ChatResponse> response =
            co_await provider_.complete_async(model_, request->ctx, request->opts);
        if (!response)
            co_return Result<bool>{ std::unexpect,
                Error{ Errc::CompactionFailed, "压缩摘要请求失败: " + response.error().message } };
        Result<std::string> summary = validate_summary_response(*response);
        if (!summary)
            co_return Result<bool>{ std::unexpect, summary.error() };
        messages_ = apply_summary(messages_, request->cut, *summary);
        update_previous_summary();
        co_return true;
    }
    // run 内防爆：每轮发请求前自动调 behaviors_.should_compact / compact（超限才压，内建底线），
    //   与外部主动压缩并存，共用同一可覆写策略。

    // ── 暴露信息（供外部判断要不要压 + 成本监控）──
    /// 当前上下文完整用量（真实锚点 + 估算补尾，cost 已算）。
    Usage context_usage() const
    {
        Usage usage;
        int total = estimate_context_tokens(messages_);
        usage.total_tokens = total;
        usage.input_tokens = total;   // 上下文整体按输入计（重新发送成本近似）
        if (last_usage_)
            usage.output_tokens = last_usage_->output_tokens;
        usage.cost = static_cast<double>(total) * (model_.price_input / 1e6)
                   + (last_usage_
                          ? static_cast<double>(last_usage_->output_tokens) * (model_.price_output / 1e6)
                          : 0.0);
        return usage;
    }
    /// 最近一次真实 usage（模型返回，cost 已算；无则 nullopt）。
    std::optional<Usage> last_usage() const { return last_usage_; }
    /// 累计花费（美元）。
    double context_cost() const { return cumulative_cost_; }
    /// = context_usage().total_tokens。
    int context_tokens() const { return context_usage().total_tokens; }
    /// 上次摘要文本（增量更新用）。
    std::string_view previous_summary() const { return previous_summary_; }

private:
    // ── 快照 / 用量 ──
    ContextSnapshot build_snapshot() const
    {
        return ContextSnapshot{
            .usage = context_usage(),
            .context_window = model_.context_window,
            .system_prompt = system_prompt_,
            .previous_summary = previous_summary_,
            .model = model_,
            .messages = messages_,
        };
    }

    /// usage × 模型单价 → 美元（单价为美元/百万 token）。
    static double compute_cost(Usage const& usage, ModelView const& model)
    {
        return (static_cast<double>(usage.input_tokens) * model.price_input
              + static_cast<double>(usage.output_tokens) * model.price_output
              + static_cast<double>(usage.cache_read_tokens) * model.price_cache_read
              + static_cast<double>(usage.cache_write_tokens) * model.price_cache_write) / 1e6;
    }

    /// 记录一轮真实 usage 并补算 cost（累计到 cumulative_cost_）。
    void record_usage(Usage const& usage)
    {
        Usage recorded = usage;
        recorded.cost = compute_cost(recorded, model_);
        last_usage_ = recorded;
        cumulative_cost_ += recorded.cost;
    }

    /// 是否已取消（abort() 置位；每次 run 清标志，见 run()/run_async()）。
    bool cancelled() const { return aborted_.load(); }

    /// 发送事件（返回 channel 是否仍开放；循环忽略返回值，靠自然结束 / 取消收尾）。
    asio::awaitable<bool> emit(AsyncStream<AgentEvent>& sink, AgentEvent event)
    {
        co_return co_await sink.send(std::move(event));
    }

    /// 失败收尾：记 last_error_ + 发 AgentError + AgentEnd + streaming_ 复位。
    asio::awaitable<void> fail_run(AsyncStream<AgentEvent>& sink, Error const& error)
    {
        last_error_ = error;
        streaming_.store(false);
        co_await emit(sink, AgentEvent{ AgentError{ error } });
        co_await emit(sink, AgentEvent{ AgentEnd{ messages_ } });
    }

    /// 追加正文增量（合并到末尾 Text 块）。
    static void append_text(Message& message, std::string const& text)
    {
        if (!message.content.empty()) {
            if (auto t = std::get_if<Text>(&message.content.back())) {
                t->text += text;
                return;
            }
        }
        message.content.push_back(Text{ text });
    }

    /// 追加思考增量（合并到末尾 Thinking 块）。
    static void append_thinking(Message& message, std::string const& text)
    {
        if (!message.content.empty()) {
            if (auto t = std::get_if<Thinking>(&message.content.back())) {
                t->text += text;
                return;
            }
        }
        message.content.push_back(Thinking{ text });
    }

    /// 从最近一次压缩后更新 previous_summary_（找最后一条摘要消息）。
    void update_previous_summary()
    {
        previous_summary_.clear();
        for (auto it = messages_.rbegin(); it != messages_.rend(); ++it) {
            if (is_compaction_summary(*it)) {
                previous_summary_ = extract_summary_text(*it);
                break;
            }
        }
    }

    bool has_queued_steer() const
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return !steer_queue_.empty();
    }
    bool has_queued_follow_up() const
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return !follow_up_queue_.empty();
    }

    // ── 队列注入（coroutine：要发 MessageStart/End 事件）──
    asio::awaitable<void> inject_steering(AsyncStream<AgentEvent>& sink)
    {
        std::vector<Message> to_inject;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (steer_queue_.empty())
                co_return;
            if (steering_mode_ == QueueMode::OneAtATime) {
                to_inject.push_back(std::move(steer_queue_.front()));
                steer_queue_.erase(steer_queue_.begin());
            } else {
                to_inject.swap(steer_queue_);
            }
        }
        for (Message const& m : to_inject) {
            messages_.push_back(m);
            co_await emit(sink, AgentEvent{ MessageStart{ m } });
            co_await emit(sink, AgentEvent{ MessageEnd{ m } });
        }
    }

    asio::awaitable<void> inject_follow_up(AsyncStream<AgentEvent>& sink)
    {
        std::vector<Message> to_inject;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (follow_up_queue_.empty())
                co_return;
            if (follow_up_mode_ == QueueMode::OneAtATime) {
                to_inject.push_back(std::move(follow_up_queue_.front()));
                follow_up_queue_.erase(follow_up_queue_.begin());
            } else {
                to_inject.swap(follow_up_queue_);
            }
        }
        for (Message const& m : to_inject) {
            messages_.push_back(m);
            co_await emit(sink, AgentEvent{ MessageStart{ m } });
            co_await emit(sink, AgentEvent{ MessageEnd{ m } });
        }
    }

    // ── 工具执行 ──
    static void apply_exec_result(ToolResult& result, ToolCall const& call,
                                  Result<std::string> const& r)
    {
        (void)call;
        if (r) {
            result.output = *r;
            result.is_error = false;
        } else {
            result.is_error = true;
            result.output = r.error().message;
        }
    }

    /// 工具是否在允许集（tools_）内——执行门控：不在集内的调用被拒绝回传。
    bool is_tool_allowed(std::string const& name) const
    {
        return std::find(tools_.begin(), tools_.end(), name) != tools_.end();
    }

    /// 工具有效执行模式：per-tool（注册时指定，!= Default）覆盖 Agent 全局默认。
    ToolExecutionMode effective_tool_mode(std::string const& name) const
    {
        Result<ToolExecutionMode> mode = Tools::mode(name);
        if (mode && *mode != ToolExecutionMode::Default)
            return *mode;
        return tool_execution_mode_;
    }

    /// 执行一批工具：**先门控**（tools_ 允许集 + before 钩子）→ 执行（Parallel 用 std::async）→
    /// after 钩子 → ToolExecStart/End 事件 → 回传 ToolResult 消息。同步工具无部分结果，
    /// 不发 ToolExecUpdate。
    asio::awaitable<std::vector<ToolResult>> execute_tools(std::vector<ToolCall> const& calls,
                                                           AsyncStream<AgentEvent>& sink)
    {
        struct Planned {
            ToolCall call;
            ToolResult result;
            bool rejected;
        };
        std::vector<Planned> planned;
        planned.reserve(calls.size());
        for (ToolCall const& tc : calls) {
            pending_tools_.push_back(tc.id);
            co_await emit(sink, AgentEvent{ ToolExecStart{ tc.id, tc.name, tc.arguments } });
            Result<bool> before = behaviors_.before_tool_call(provider_, tc, build_snapshot());
            ToolResult result;
            result.tool_call_id = tc.id;
            bool rejected = false;
            // 执行门控：不在 tools_ 允许集的工具直接拒绝（须 set_tools 显式指定），
            // 不真执行，把「未开放」原因回传给模型让它调整。
            if (!is_tool_allowed(tc.name)) {
                rejected = true;
                result.is_error = true;
                result.output = "工具「" + tc.name + "」未开放（须 set_tools 显式指定）";
            } else if (!before) {
                rejected = true;
                result.is_error = true;
                result.output = before.error().message;
            } else if (!*before) {
                rejected = true;
                result.is_error = true;
                result.output = "工具被拒绝执行: " + tc.name;
            }
            planned.push_back({ tc, std::move(result), rejected });
        }

        // 每个工具的有效执行模式：per-tool（!= Default）覆盖 Agent 全局默认。
        // 执行：Parallel 用 std::async 并发（先全部启动，后台跑），Sequential 按序内联，
        // 结果按 planned 原顺序收集（每个工具结果带自己的 tool_call_id）。
        std::vector<ToolExecutionMode> modes;
        modes.reserve(planned.size());
        for (auto const& p : planned)
            modes.push_back(effective_tool_mode(p.call.name));

        std::vector<std::future<Result<std::string>>> futures(planned.size());
        for (std::size_t i = 0; i < planned.size(); ++i)
            if (!planned[i].rejected && modes[i] == ToolExecutionMode::Parallel)
                futures[i] = std::async(std::launch::async, [&p = planned[i]]() {
                    return Tools::exec(p.call.name, p.call.arguments);
                });

        for (std::size_t i = 0; i < planned.size(); ++i) {
            Planned& p = planned[i];
            if (p.rejected)
                continue;
            Result<std::string> r = (modes[i] == ToolExecutionMode::Parallel)
                ? futures[i].get()
                : Tools::exec(p.call.name, p.call.arguments);
            apply_exec_result(p.result, p.call, r);
            Result<ToolResult> after = behaviors_.after_tool_call(provider_, p.call, p.result);
            if (after)
                p.result = *after;
            else {
                p.result.is_error = true;
                p.result.output = after.error().message;
            }
        }

        std::vector<ToolResult> results;
        results.reserve(planned.size());
        for (Planned& p : planned) {
            co_await emit(sink, AgentEvent{ ToolExecEnd{ p.call.id, p.call.name, p.result } });
            results.push_back(std::move(p.result));
            pending_tools_.pop_back();
        }
        for (ToolResult const& tr : results) {
            Message tool_message{ Role::ToolResult, { tr } };
            messages_.push_back(tool_message);
            co_await emit(sink, AgentEvent{ MessageStart{ tool_message } });
            co_await emit(sink, AgentEvent{ MessageEnd{ tool_message } });
        }
        co_return results;
    }

    // ── 核心循环（同步 run / 异步 run_async 共用）──
    asio::awaitable<void> loop_async(std::vector<Message> inject,
                                     AsyncStream<AgentEvent> sink,
                                     StreamOptions opts)
    {
        auto ex = co_await asio::this_coro::executor;
        streaming_.store(true);
        last_error_.reset();

        // 任何退出路径（正常 / fail_run / 取消 unwind）都关闭事件通道——
        // 否则 SyncStreamBridge 的 pop_after_io 永远等不到 channel_closed 而挂起。
        struct SinkGuard {
            AsyncStream<AgentEvent>& sink;
            ~SinkGuard() { sink.close(); }
        } sink_guard{ sink };

        // AgentStart 最先（对齐 pi agent_start = run 开始），随后才是 user 消息事件
        if (!co_await emit(sink, AgentEvent{ AgentStart{} }))
            co_return;   // 消费端已关闭（generator break 场景）
        if (cancelled()) {
            co_await fail_run(sink, Error{ Errc::Aborted, "run 被取消" });
            co_return;
        }

        // 注入初始 user 消息
        for (Message const& m : inject) {
            messages_.push_back(m);
            co_await emit(sink, AgentEvent{ MessageStart{ m } });
            co_await emit(sink, AgentEvent{ MessageEnd{ m } });
        }

        // run_start 钩子：改写初始消息 + 系统提示
        Result<std::optional<RunStartPatch>> patch = behaviors_.run_start(provider_, messages_, system_prompt_);
        if (!patch) {
            co_await fail_run(sink, patch.error());
            co_return;
        }
        if (*patch) {
            messages_ = std::move((**patch).messages);
            system_prompt_ = std::move((**patch).system_prompt);
        }

        while (true) {   // 外层：follow-up
            while (true) {   // 内层 turn 循环
                if (cancelled()) {
                    co_await fail_run(sink, Error{ Errc::Aborted, "run 被取消" });
                    co_return;
                }

                // run 内防爆：每轮发请求前自动压缩（超限才压，内建底线）
                {
                    ContextSnapshot snap = build_snapshot();
                    if (behaviors_.should_compact(snap)) {
                        Result<std::optional<std::vector<Message>>> cr = behaviors_.compact(provider_, snap);
                        if (!cr) {
                            co_await fail_run(sink, cr.error());
                            co_return;
                        }
                        if (*cr) {
                            messages_ = std::move(**cr);
                            update_previous_summary();
                        }
                    }
                }

                co_await emit(sink, AgentEvent{ TurnStart{} });
                co_await inject_steering(sink);
                if (cancelled()) {
                    co_await fail_run(sink, Error{ Errc::Aborted, "run 被取消" });
                    co_return;
                }

                // 构建请求上下文
                Context ctx;
                ctx.system_prompt = system_prompt_;
                ctx.messages = messages_;
                ctx.tools = tools_;
                Result<std::vector<Message>> tc = behaviors_.transform_context(provider_, build_snapshot());
                if (!tc) {
                    co_await fail_run(sink, tc.error());
                    co_return;
                }
                ctx.messages = std::move(*tc);

                StreamOptions effective = opts;
                if (!effective.reasoning && reasoning_)
                    effective.reasoning = reasoning_;
                Result<std::optional<StreamOptions>> br = behaviors_.before_request(provider_, effective);
                if (!br) {
                    co_await fail_run(sink, br.error());
                    co_return;
                }
                if (*br)
                    effective = **br;
                Result<std::optional<std::string>> key = behaviors_.get_api_key(provider_name_);
                if (!key) {
                    co_await fail_run(sink, key.error());
                    co_return;
                }
                if (*key)
                    effective.api_key = **key;

                // before_payload：引擎生成请求体 → 钩子改写 → 塞回 prebuilt_body 交还引擎。
                // 钩子 nullopt（不改）→ 也用构建原样，避免引擎二次 build_params。
                Result<nlohmann::json> built = provider_.build_params(model_, ctx, effective);
                if (!built) {
                    co_await fail_run(sink, built.error());
                    co_return;
                }
                Result<std::optional<nlohmann::json>> payload = behaviors_.before_payload(provider_, *built);
                if (!payload) {
                    co_await fail_run(sink, payload.error());
                    co_return;
                }
                effective.prebuilt_body = *payload ? **payload : *built;

                // LLM 流：生产协程绑定取消信号（abort() → 干净中断）。
                // ⚠️ lambda 按值捕获 ctx / effective：生产协程可能晚于 loop 局部析构
                //（取消时两者独立 unwind），引用捕获会悬空。
                // ⚠️ RAII CloseGuard：llm.close() 在正常/取消/异常任何退出路径都执行——
                //   消费端 receive 必得 channel_closed，不会挂起。析构清理不算异常用法。
                AsyncStream<StreamEvent> llm(ex);
                asio::co_spawn(ex,
                    [this, ctx, effective, llm]() mutable -> asio::awaitable<void> {
                        struct CloseGuard {
                            AsyncStream<StreamEvent>& llm;
                            ~CloseGuard() { llm.close(); }
                        } guard{ llm };
                        co_await provider_.stream_async(model_, ctx, effective, llm);
                    },
                    asio::bind_cancellation_slot(cancel_->slot(), asio::detached));

                // 消费流 → assistant 消息 + 工具调用
                Message assistant{ Role::Assistant, {} };
                bool assistant_started = false;
                std::vector<ToolCall> tool_calls;
                std::optional<Error> stream_error;
                bool got_done = false;
                while (true) {
                    Result<StreamEvent> ev = co_await llm.receive();
                    if (!ev)
                        break;   // 生产协程结束（channel 关闭）
                    switch (ev->type()) {
                        case StreamEvent::Type::TextDelta: {
                            std::string text = std::get<TextDelta>(ev->data).text;
                            if (!assistant_started) {
                                assistant_started = true;
                                co_await emit(sink, AgentEvent{ MessageStart{ assistant } });
                            }
                            append_text(assistant, text);
                            co_await emit(sink, AgentEvent{ MessageUpdate{ assistant, TextDelta{ text } } });
                            break;
                        }
                        case StreamEvent::Type::ThinkingDelta: {
                            std::string text = std::get<ThinkingDelta>(ev->data).text;
                            if (!assistant_started) {
                                assistant_started = true;
                                co_await emit(sink, AgentEvent{ MessageStart{ assistant } });
                            }
                            append_thinking(assistant, text);
                            co_await emit(sink, AgentEvent{ MessageUpdate{ assistant, ThinkingDelta{ text } } });
                            break;
                        }
                        case StreamEvent::Type::ToolCallDelta: {
                            co_await emit(sink, AgentEvent{ std::get<ToolCallDelta>(ev->data) });
                            break;
                        }
                        case StreamEvent::Type::ToolCallEnd: {
                            ToolCallEnd const& ce = std::get<ToolCallEnd>(ev->data);
                            ToolCall call{ ce.id, ce.name, ce.arguments };
                            tool_calls.push_back(call);
                            assistant.push_back(ContentBlock{ call });
                            co_await emit(sink, AgentEvent{ ce });
                            break;
                        }
                        case StreamEvent::Type::Usage: {
                            break;   // Done 的 usage 为准
                        }
                        case StreamEvent::Type::Done: {
                            ChatResponse& response = std::get<DoneEvent>(ev->data).response;
                            assistant = Message{ Role::Assistant, response.content };
                            record_usage(response.usage);
                            assistant.set_input_tokens(static_cast<std::size_t>(response.usage.input_tokens));
                            assistant.set_output_tokens(static_cast<std::size_t>(response.usage.output_tokens));
                            if (!assistant_started) {
                                assistant_started = true;
                                co_await emit(sink, AgentEvent{ MessageStart{ assistant } });
                            }
                            co_await emit(sink, AgentEvent{ MessageEnd{ assistant } });
                            got_done = true;
                            break;
                        }
                        case StreamEvent::Type::Error: {
                            stream_error = std::get<Error>(ev->data);
                            break;
                        }
                    }
                    if (got_done || stream_error)
                        break;
                }

                if (stream_error) {
                    co_await fail_run(sink, *stream_error);
                    co_return;
                }
                if (!got_done) {
                    // 流中断（channel 关闭而无终结事件）：取消 → Aborted；否则异常终断
                    if (cancelled())
                        co_await fail_run(sink, Error{ Errc::Aborted, "run 被取消" });
                    else
                        co_await fail_run(sink, Error{ Errc::NetworkError, "stream ended without terminal event" });
                    co_return;
                }

                Message assistant_copy = assistant;
                messages_.push_back(std::move(assistant));

                // 工具执行（回传 ToolResult 消息）
                std::vector<ToolResult> turn_results;
                if (!tool_calls.empty())
                    turn_results = co_await execute_tools(tool_calls, sink);
                if (cancelled()) {
                    co_await fail_run(sink, Error{ Errc::Aborted, "run 被取消" });
                    co_return;
                }

                // TurnEnd + 每轮结束钩子
                co_await emit(sink, AgentEvent{ TurnEnd{ assistant_copy, turn_results } });
                Result<bool> stop = behaviors_.should_stop(provider_, assistant_copy);
                if (!stop) {
                    co_await fail_run(sink, stop.error());
                    co_return;
                }
                Result<std::optional<NextTurnUpdate>> next = behaviors_.prepare_next_turn(
                    provider_, assistant_copy, turn_results);
                if (!next) {
                    co_await fail_run(sink, next.error());
                    co_return;
                }
                if (*next) {
                    if ((**next).model) model_ = *((**next).model);
                    if ((**next).reasoning) reasoning_ = *((**next).reasoning);
                    if ((**next).context) messages_ = std::move(*((**next).context));
                }

                if (*stop)
                    break;   // 提前停（应 stop）
                if (!has_queued_steer() && tool_calls.empty())
                    break;   // 无 steer 且无工具往返 → 内层结束
                // 否则继续内层（有 steer 插队或工具往返待续）
            }

            if (!has_queued_follow_up())
                break;   // 外层结束
            co_await inject_follow_up(sink);   // 将停时注入 → 新的一轮
        }

        co_await emit(sink, AgentEvent{ AgentEnd{ messages_ } });
        streaming_.store(false);
    }

    std::string provider_name_;
    Provider provider_;
    ModelView model_;
    std::string system_prompt_;
    std::vector<Message> messages_;                    ///< 当前对话（压缩丢弃旧段，只留摘要 + 尾部保留段）
    std::vector<std::string> tools_;                   ///< 传给模型的工具子集
    std::optional<Usage> last_usage_;                  ///< 最近一轮真实 usage（模型返回，Agent 补算 cost）
    double cumulative_cost_ = 0;                       ///< 累计花费（美元）
    std::string previous_summary_;                     ///< 上次摘要文本（压缩后更新）
    std::optional<ThinkingLevel> reasoning_;           ///< set_reasoning 设定的思考档（下一轮生效）
    std::optional<asio::cancellation_signal> cancel_;  ///< 每次 run 重建（可重置取消语义）
    std::atomic<bool> aborted_ = false;                ///< abort() 置位（每次 run 清标志）
    Behaviors behaviors_;                              ///< 钩子 + 压缩策略一体（编译期绑定）
    std::atomic<bool> streaming_ = false;                  ///< 是否正在 run
    std::vector<std::string> pending_tools_;           ///< 正在执行的工具 id
    std::optional<Error> last_error_;                  ///< 最近失败/中止错误
    mutable std::mutex queue_mutex_;                   ///< 保护 steer/follow_up 队列
    std::vector<Message> steer_queue_;
    std::vector<Message> follow_up_queue_;
    QueueMode steering_mode_ = QueueMode::OneAtATime;
    QueueMode follow_up_mode_ = QueueMode::OneAtATime;
    ToolExecutionMode tool_execution_mode_ = ToolExecutionMode::Parallel;
};

}  // namespace agent
