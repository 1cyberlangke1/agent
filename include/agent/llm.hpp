#pragma once

// LLM 调用层统一类型层（L1）。
// 包含：统一枚举 / Usage / 模型表三结构（BuiltinModel/RuntimeModel/ModelView）/
// ModelRegistry（纯静态类）/ EndpointConfig / Context / StreamOptions /
// ChatResponse / StreamEvent / AsyncStream / Provider 四接口声明 / 策略声明。

#include <agent/content.hpp>
#include <agent/result.hpp>
#include <agent/tools.hpp>

#include <asio.hpp>
#include <asio/experimental/basic_concurrent_channel.hpp>
#include <asio/experimental/channel_traits.hpp>

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <deque>
#include <expected>
#include <generator>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace agent {

// ─────────────────────────────────────────────────────────────
// 统一枚举
// ─────────────────────────────────────────────────────────────

/// @brief 统一思考等级（ThinkingLevel）。
///
/// 用户只传统一等级，各引擎内部映射到厂商原生格式：
/// - OpenAI / DeepSeek（effort 型）→ reasoning_effort（如 "high" / "max"）
/// - Anthropic → effort 档位，或 budget_tokens（预算 token 数，见 Model.thinking_level_map）
/// - Gemini → thinkingLevel / thinkingBudget
///
/// 枚举值必须按序 0..6（Off/Minimal/Low/Medium/High/XHigh/Max），
/// 用作数组下标访问 Model.thinking_level_map（固定数组，无 map 树、无查找）。
/// 等级由粗到细：Off 关闭思考，Minimal/Low 轻量，Medium 标准，
/// High/XHigh 深入，Max 最大化推理投入。
/// 模型不支持的等级由 clamp_thinking_level() 收敛到最近支持档。
enum class ThinkingLevel {
    Off,       ///< 关闭思考（引擎不发 thinking 参数即默认关闭）
    Minimal,   ///< 最轻量思考（快速响应优先）
    Low,       ///< 轻量思考
    Medium,    ///< 标准思考（默认档位）
    High,      ///< 深入思考
    XHigh,     ///< 很深入思考
    Max,       ///< 最大化推理投入（更慢、更彻底）
};

/// @brief 等级数。模型表三结构 thinking_level_map 的维度。
///        static_assert 与枚举同步（枚举按序 0..6，Max 下标即 6）：
///        static_assert(std::to_underlying(ThinkingLevel::Max) + 1 == thinking_level_count);
inline constexpr std::size_t thinking_level_count = 7;
static_assert(static_cast<std::size_t>(ThinkingLevel::Max) + 1 == thinking_level_count,
              "ThinkingLevel 枚举必须与 thinking_level_count 同步");

/// @brief 统一缓存保留策略（CacheRetention）。
///
/// 用户只传统一档位，各引擎映射到厂商缓存机制：
/// - OpenAI → prompt_cache_retention（"in_memory" / "24h"）
/// - Anthropic → cache_control（ephemeral / Long 时 ttl:"1h"）
/// - DeepSeek / Gemini → 自动缓存，档位被忽略
///
/// 语义是「缓存意图」（要不要缓存、尽量短/长）而非各家精确时长保证；
/// 想对特定厂商做精细控制请走 StreamOptions.extra 透传。
enum class CacheRetention {
    None,   ///< 不缓存
    Short,  ///< 短保留
    Long,   ///< 长保留（映射到各家的最长档）
};

/// @brief 停止原因：模型为什么结束生成（StopReason）。
///
/// 各厂商映射：OpenAI finish_reason / Anthropic stop_reason / Gemini finishReason。
/// 无 pending——流式中间态由 StreamEvent 表达，最终结果不会有 pending。
enum class StopReason {
    Stop,     ///< 自然停止点或命中了 stop 序列
    Length,   ///< 达到 max_tokens 上限
    ToolUse,  ///< 模型决定调用工具（需回传执行结果继续）
    Error,    ///< 内容过滤等异常终止
    Aborted,  ///< 用户/调用方取消
};

// ─────────────────────────────────────────────────────────────
// Usage — 统一用量
// ─────────────────────────────────────────────────────────────

/// @brief 一次调用的 token 用量。字段统一，各家引擎内部映射。
struct Usage {
    int input_tokens = 0;          // 输入 token
    int output_tokens = 0;         // 输出 token
    int cache_read_tokens = 0;     // 缓存命中读取
    int cache_write_tokens = 0;    // 缓存写入（Anthropic 有）
    int total_tokens = 0;          // 总计
};

// ─────────────────────────────────────────────────────────────
// 模型表三结构（双轨：静态内置表 + 运行时注册表）
// ─────────────────────────────────────────────────────────────

namespace detail {

/// @brief 内置静态表模型（生成器产物，真 constexpr，零分配）。
///        数据源：models.dev/api.json（LLM 过滤后按配置的 provider 生成）。
///        由 scripts/update_models.py 生成到 include/agent/models/generated.hpp，
///        放 detail 命名空间，非 API，重新生成不通知。
struct BuiltinModel {
    std::string_view id;                    // "deepseek-chat"
    int context_window = 0;                 // 总上下文窗口
    int max_output_tokens = 0;              // 最大输出 token
    bool reasoning = false;                 // 支持思维链
    bool supports_image_input = false;      // 视觉能力
    /// 统一 ThinkingLevel → 厂商原生值映射。值唯一语义，无混用：
    ///   nullopt → 模型无思考能力（reasoning=false）；
    ///   "off"   → Off 档：关闭思考（引擎不发 thinking 参数）；
    ///   "on"    → toggle 型（支持思考但无强度细分）的启用档；
    ///   effort 值（"low".."max"）→ effort 型档位强度；
    ///   budget 值（"1024".."32768"）→ budget 型档位预算。
    /// 固定数组按下标访问（枚举按序 0..6），无 map 树。
    /// effort 型非 Off 档已预填「向下收敛最近有效值」——引擎取 map[level]
    /// 非 Off 恒有值，免判空。
    /// 值统一为字符串形态，指向静态字面量。
    std::array<std::optional<std::string_view>, thinking_level_count> thinking_level_map;
    /// thinking 值写入的请求字段名（Gemini 3 → "thinkingLevel"，2.5 → "thinkingBudget"）；
    /// 空 = 用引擎/ThinkingPolicy 的默认字段。
    std::string_view thinking_field;
    /// 单价（美元/百万 token）。0 = 未知或免费。
    double price_input = 0;
    double price_output = 0;
    double price_cache_read = 0;
    double price_cache_write = 0;
};

/// @brief 生成器产物的汇总表项：provider 名 → 模型子表。
struct BuiltinProviderTable {
    std::string_view provider;
    std::span<const BuiltinModel> models;
};

}  // namespace detail

/// @brief 运行时注册的模型（全 std::string 自持）。
///        语义同 detail::BuiltinModel，字符串动态，由 ModelRegistry 拥有生命周期。
struct RuntimeModel {
    std::string id;
    int context_window = 0;
    int max_output_tokens = 0;
    bool reasoning = false;
    bool supports_image_input = false;
    std::array<std::optional<std::string>, thinking_level_count> thinking_level_map;
    std::string thinking_field;
    double price_input = 0;          // 美元/百万 token；0 = 未知或免费
    double price_output = 0;
    double price_cache_read = 0;
    double price_cache_write = 0;
};

/// @brief 统一模型视图：引擎/调用方唯一消费类型。
///        string_view 指向底层存储（静态 → 全局字面量；动态 → ModelRegistry 的 deque 元素，
///        全局存储、地址稳定、进程生命周期），返回后长期有效。
///        从 BuiltinModel / RuntimeModel 均可构造（ModelRegistry 内部转换）。
struct ModelView {
    std::string_view id;
    int context_window = 0;
    int max_output_tokens = 0;
    bool reasoning = false;
    bool supports_image_input = false;
    std::array<std::optional<std::string_view>, thinking_level_count> thinking_level_map;
    std::string_view thinking_field;
    double price_input = 0;          // 美元/百万 token；0 = 未知或免费
    double price_output = 0;
    double price_cache_read = 0;
    double price_cache_write = 0;
};

// ─────────────────────────────────────────────────────────────
// ModelRegistry — 全局模型注册表（纯静态类，多线程安全）
// ─────────────────────────────────────────────────────────────

/// @brief 全局模型注册表。**纯静态类**：禁实例化，全 static 成员，进程生命周期。
///        独立于 API 调用层——只关心「有哪些模型、元数据是什么」，零协议概念。
///        内置表（生成器产物）+ 运行时注册 + 合并索引；动态覆盖静态。
///        线程安全：shared_mutex（register 唯一锁 / find、for_each 共享锁）+
///        call_once 惰性加载内置表。
class ModelRegistry {
public:
    ModelRegistry() = delete;

    /// @brief 运行时注册/覆盖模型（同 id 覆盖内置或已有动态项）。线程安全。
    ///        覆盖实现约束：deque 只追加新条目、旧条目永不改写/删除——
    ///        原地改写会使已发出的 ModelView 的 string_view 悬空。
    ///        index_ 改指新条目；for_each 输出以 index_ 当前指向为准（旧重复条目跳过）。
    ///        注册支持思考的模型：须填 reasoning=true 且按新语义填 thinking_level_map
    ///        （"off"/"on"/effort 值/budget 值）；thinking_level_map 全 nullopt = 无思考能力，
    ///        thinking 静默失效——非推理模型才应留空。
    /// @param model 模型数据（拷贝进注册表）
    /// @return true = 新增 id，false = 覆盖已有 id
    static bool register_model(RuntimeModel model);

    /// @brief 统一查找（动态覆盖静态）。O(1) 哈希索引。线程安全。
    /// @param id 模型 id
    /// @return 匹配的 ModelView；未找到返回 nullopt
    static std::optional<ModelView> find_model(std::string_view id);

    /// @brief 遍历全部可用模型（稳定顺序：静态表顺序 → 动态首次注册序，动态覆盖不重复）。
    ///        内部先持共享锁快照再回调 → 回调内可安全再调 register_model，无重入死锁。
    /// @param f 回调，接收 const ModelView&
    template<typename F>
    static void for_each_model(F&& f)
    {
        std::shared_lock lock(mutex_);
        init();
        std::vector<ModelView> snapshot;
        snapshot.reserve(index_.size());
        std::unordered_set<std::string_view> emitted;
        for (std::string_view id : static_order_) {
            if (dynamic_ids_.contains(id)) continue;   // 被动态覆盖 → 跳过静态项
            auto it = index_.find(id);
            if (it != index_.end()) {
                snapshot.push_back(it->second);
                emitted.insert(id);
            }
        }
        for (RuntimeModel const& rtm : runtime_models_) {
            // 首次注册位置输出，内容取 index_ 当前指向（最新注册）
            if (emitted.insert(rtm.id).second)
                snapshot.push_back(index_.at(rtm.id));
        }
        lock.unlock();
        for (ModelView const& mv : snapshot)
            f(mv);
    }

private:
    static void init();                                   // call_once 惰性加载内置表
    static ModelView to_view(detail::BuiltinModel const& m);
    static ModelView to_view(RuntimeModel const& m);

    inline static std::once_flag init_flag_;
    inline static std::shared_mutex mutex_;               // 多线程安全
    inline static std::deque<RuntimeModel> runtime_models_;   // 动态表（地址稳定，只追加）
    inline static std::unordered_map<std::string_view, ModelView> index_;  // 合并索引
    inline static std::unordered_set<std::string_view> dynamic_ids_;       // 动态覆盖的 id
    inline static std::vector<std::string_view> static_order_;             // 静态表遍历顺序
};

/// @brief 把用户请求的思考等级收敛到模型支持的范围内（从低向高找最近支持档）。
/// @param model  目标模型
/// @param level  用户请求的等级
/// @return 模型支持的最高等级（若全部不支持则返回 Off）
ThinkingLevel clamp_thinking_level(ModelView const& model, ThinkingLevel level);

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

// ─────────────────────────────────────────────────────────────
// ChatResponse — 完整响应
// ─────────────────────────────────────────────────────────────

/// @brief 一次完整（非流式或流式收尾）的模型响应。
struct ChatResponse {
    std::vector<ContentBlock> content;
    StopReason stop_reason = StopReason::Stop;
    Usage usage;
    std::string response_id;
    /// 原始上游响应 JSON。仅 StreamOptions::capture_raw_response 开启时填充
    /// （流式 = 最终累积的原始对象；非流式 = 完整 body），否则为空对象。
    nlohmann::json raw;
};

// ─────────────────────────────────────────────────────────────
// StreamEvent — enum Type + variant（稳定、明确）
// ─────────────────────────────────────────────────────────────

/// @brief 流式文本增量。
///
/// 模型输出文本的逐段增量，Agent 层自行累积为完整文本。
struct TextDelta {
    std::string text;   ///< 本次增量的文本片段
};

/// @brief 流式思考内容增量。
///
/// 模型推理过程的内部思考逐段增量（OpenAI reasoning / Anthropic thinking /
/// DeepSeek reasoning_content），与正式输出 TextDelta 分开。
struct ThinkingDelta {
    std::string text;   ///< 本次增量的思考片段
};

/// @brief 工具参数增量。delta 阶段只给 JSON 字符串增量，不 parse——
///        由 Agent 层自行决定何时累积/解析。
///
/// 一个工具调用会发出多次 ToolCallDelta（arguments_delta 逐段累加），
/// 结束时由 ToolCallEnd 给出完整解析后的 arguments。
struct ToolCallDelta {
    std::string id;                ///< 工具调用 id（首次出现时非空，后续可能为空）
    std::string name;              ///< 工具名（首次出现时非空，后续可能为空）
    std::string arguments_delta;   ///< 参数 JSON 的字符串增量（逐段累加）
};

/// @brief 工具调用完成：参数为完整解析后的 JSON 对象。
///
/// 流中该工具调用的最后事件；此后应把 ToolCall 加入 assistant 消息，
/// 多轮 tool calling 由 Agent 层驱动。
struct ToolCallEnd {
    std::string id;                 ///< 工具调用 id
    std::string name;               ///< 工具名
    nlohmann::json arguments;       ///< 完整解析后的参数对象
};

/// @brief token 用量事件。
///
/// 流中最多一次（OpenAI include_usage / Anthropic message_delta /
/// Gemini usageMetadata）；也可能缺失（断流时官方不保证）。
struct UsageEvent {
    Usage usage;   ///< 本次请求累计用量
};

/// @brief 流结束事件，携带完整响应。
///
/// 终结事件之一（另一个是 Error）。DoneEvent 之后流必被 close；
/// Done/Error 皆缺时壳层兜底合成 Error（见 StreamFacade 终结契约）。
struct DoneEvent {
    ChatResponse response;   ///< 聚合好的完整响应
};

/// @brief 统一流事件。每个事件 = 类型判别 + 变体载荷，不用「一堆 optional 字段」。
///
/// 流式语义：模型输出由若干 TextDelta / ThinkingDelta / ToolCallDelta 增量
/// 组成，工具调用以 ToolCallEnd 收尾，Token 用量由 Usage 给出，流以
/// Done（完整响应）或 Error 终结。
///
/// 消费方式：用 std::visit 访问载荷（与 ContentBlock 的 variant 风格一致），
/// 或用 type() 快速判别后 std::get 取具体载荷。
struct StreamEvent {
    /// 事件类型枚举。顺序必须与 data variant 的载荷声明顺序严格一致
    /// （type() 直接强转 variant index，顺序错则判别错位）。
    enum class Type {
        TextDelta,
        ThinkingDelta,
        ToolCallDelta,
        ToolCallEnd,
        Usage,
        Done,
        Error,
    };
    std::variant<TextDelta, ThinkingDelta, ToolCallDelta, ToolCallEnd, UsageEvent, DoneEvent, Error> data;
    /// @brief 类型判别从 variant 下标推导——单一真相源，不存冗余字段
    ///        （存独立 type 字段会出现 type 与 data 失同步的双源真相）。
    Type type() const { return static_cast<Type>(data.index()); }
};
static_assert(std::variant_size_v<decltype(StreamEvent::data)> == 7,
              "StreamEvent 载荷类型数与 Type 枚举必须一致");

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
///        - 可移动：按值移动传入 stream_async，生产端持有
///        - close 语义：生产协程结束（Done/Error 后）close；消费端 receive 收到
///        channel_closed 错误即流结束
///        - 消费侧同步取：try_receive(handler)——handler 收 (error_code, value)，
///        非 try_receive(ec, value)（L0-spike 已验证）
template<typename T>
class AsyncStream {
public:
    using channel_traits_t = asio::experimental::channel_traits<void(asio::error_code, T)>;
    using channel_t = asio::experimental::basic_concurrent_channel<asio::any_io_executor, channel_traits_t, void(asio::error_code, T)>;

    /// @param executor 生产/消费协程的 io_context executor
    /// @param capacity 有界容量（背压上限）
    AsyncStream(asio::any_io_executor executor, std::size_t capacity = 64)
        : ch_(std::move(executor), capacity)
    {
    }

    AsyncStream(AsyncStream&&) noexcept = default;
    AsyncStream& operator=(AsyncStream&&) noexcept = default;
    AsyncStream(AsyncStream const&) = delete;
    AsyncStream& operator=(AsyncStream const&) = delete;

    /// @brief 生产端发送一个事件。channel 满则挂起（背压）。
    /// @param value 事件载荷
    /// @return true = 已入队；false = channel 已关闭（消费端提前结束，应停止生产）
    asio::awaitable<bool> send(T value)
    {
        // channel 的 async_send(error_code ec, T value, token)：第一个 error_code
        // 是发送方预置错误码（正常消息传默认），token 收操作结果 void(error_code)。
        auto [ec] = co_await ch_.async_send(asio::error_code(), std::move(value), asio::as_tuple(asio::use_awaitable));
        co_return !ec;
    }

    /// @brief 生产端结束：发送完毕后关闭通道。消费端 receive 收到 channel_closed 错误。
    void close() { ch_.close(); }

    /// @brief 通道是否仍开放（未关闭）。
    bool is_open() const { return ch_.is_open(); }

    /// @brief 消费端异步取一个事件。
    /// @return 成功 → 事件值；channel 关闭 → Error（调用方以此判断流结束）
    asio::awaitable<Result<T>> receive()
    {
        auto [ec, value] = co_await ch_.async_receive(asio::as_tuple(asio::use_awaitable));
        if (!ec)
            co_return Result<T>{ std::move(value) };
        co_return Result<T>{ std::unexpect, Error{ Errc::NetworkError, ec.message() } };
    }

    /// @brief 消费端同步取（非阻塞）。队列空 → nullopt；否则返回事件或关闭错误。
    std::optional<Result<T>> try_receive()
    {
        std::optional<Result<T>> out;
        ch_.try_receive([&out](asio::error_code ec, T value) {
            if (!ec)
                out = Result<T>{ std::move(value) };
            else
                out = Result<T>{ std::unexpect, Error{ Errc::NetworkError, ec.message() } };
        });
        return out;
    }

private:
    channel_t ch_;
};

// ─────────────────────────────────────────────────────────────
// 策略声明（L4-L6 实现；数据开关 + 行为差异，编译期绑定）
// ─────────────────────────────────────────────────────────────

namespace detail {

/// @brief OpenAI 协议能力位（数据开关，constexpr static 类型策略）。
///        DeepSeek 能力位同 OpenAI → 直接复用；Moonshot/Together 等非标准厂商
///        才需要自己的 Compat。
struct OpenAICompat {
    static constexpr std::string_view max_tokens_field = "max_completion_tokens";
    static constexpr bool supports_developer_role = true;   // OpenAI 用 developer
    static constexpr bool supports_strict_mode = true;      // tool 带 strict
};

/// @brief OpenAI 思考行为（模板策略，静态成员函数，编译期绑定）。
struct OpenAIThinking {
    static constexpr std::string_view reasoning_field = "reasoning";
    static void add_params(nlohmann::json& params, StreamOptions const& opts, ModelView const& model);
    static std::optional<std::string> extract_delta(nlohmann::json const& delta);
    static void finalize_assistant(nlohmann::json& msg, Message const& message) { (void)msg; (void)message; }
};

/// @brief DeepSeek 思考行为：thinking 内容走 reasoning_content，须补空字段。
struct DeepSeekThinking {
    static constexpr std::string_view reasoning_field = "reasoning_content";
    static void add_params(nlohmann::json& params, StreamOptions const& opts, ModelView const& model);
    static std::optional<std::string> extract_delta(nlohmann::json const& delta);
    static void finalize_assistant(nlohmann::json& msg, Message const& message);
};

/// @brief OpenAI 协议引擎：L4 实现。
template<typename ThinkingPolicy, typename Compat>
class OpenAICompletionsEngine {
public:
    using event_type = StreamEvent;
    using result_type = ChatResponse;
    static std::optional<ChatResponse> as_done(StreamEvent const& ev);
    static std::optional<Error> as_error(StreamEvent const& ev);

    explicit OpenAICompletionsEngine(EndpointConfig config);

    asio::awaitable<void> stream_async(ModelView const& model, Context const& ctx, StreamOptions const& opts,
                                       AsyncStream<StreamEvent> sink);

private:
    nlohmann::json build_params(ModelView const& model, Context const& ctx, StreamOptions const& opts) const;
    EndpointConfig config_;
};

/// @brief Anthropic 协议引擎：L5 实现。
class AnthropicMessagesEngine {
public:
    using event_type = StreamEvent;
    using result_type = ChatResponse;
    static std::optional<ChatResponse> as_done(StreamEvent const& ev);
    static std::optional<Error> as_error(StreamEvent const& ev);

    explicit AnthropicMessagesEngine(EndpointConfig config);

    asio::awaitable<void> stream_async(ModelView const& model, Context const& ctx, StreamOptions const& opts,
                                       AsyncStream<StreamEvent> sink);

private:
    nlohmann::json build_params(ModelView const& model, Context const& ctx, StreamOptions const& opts) const;
    EndpointConfig config_;
};

/// @brief Gemini 协议引擎：L6 实现。
class GeminiGenerateContentEngine {
public:
    using event_type = StreamEvent;
    using result_type = ChatResponse;
    static std::optional<ChatResponse> as_done(StreamEvent const& ev);
    static std::optional<Error> as_error(StreamEvent const& ev);

    explicit GeminiGenerateContentEngine(EndpointConfig config);

    asio::awaitable<void> stream_async(ModelView const& model, Context const& ctx, StreamOptions const& opts,
                                       AsyncStream<StreamEvent> sink);

private:
    nlohmann::json build_params(ModelView const& model, Context const& ctx, StreamOptions const& opts) const;
    EndpointConfig config_;
};

}  // namespace detail

// ─────────────────────────────────────────────────────────────
// Provider 四接口（统一签名，零虚函数）
// 组合式：Provider 拥有引擎，引擎拥有策略；表在 ModelRegistry，不在此。
// ─────────────────────────────────────────────────────────────

/// @brief OpenAI 协议 Provider。实现见 src/provider_openai.cpp。
class OpenAIProvider {
public:
    explicit OpenAIProvider(EndpointConfig config);

    std::generator<StreamEvent> stream(ModelView const& model, Context const& ctx, StreamOptions const& opts = {});
    Result<ChatResponse> complete(ModelView const& model, Context const& ctx, StreamOptions const& opts = {});
    /// @brief AsyncStream 按值移动进入（channel 可移动），生产协程生命周期由引擎保证，
    ///        调用方不持有悬空引用。
    asio::awaitable<void> stream_async(ModelView const& model, Context const& ctx, StreamOptions const& opts, AsyncStream<StreamEvent> sink);
    asio::awaitable<Result<ChatResponse>> complete_async(ModelView const& model, Context const& ctx, StreamOptions const& opts);

private:
    detail::OpenAICompletionsEngine<detail::OpenAIThinking, detail::OpenAICompat> engine_;
};

/// @brief DeepSeek Provider：复用 OpenAI 协议引擎 + DeepSeek 思考策略。
class DeepSeekProvider {
public:
    explicit DeepSeekProvider(EndpointConfig config);

    std::generator<StreamEvent> stream(ModelView const& model, Context const& ctx, StreamOptions const& opts = {});
    Result<ChatResponse> complete(ModelView const& model, Context const& ctx, StreamOptions const& opts = {});
    asio::awaitable<void> stream_async(ModelView const& model, Context const& ctx, StreamOptions const& opts, AsyncStream<StreamEvent> sink);
    asio::awaitable<Result<ChatResponse>> complete_async(ModelView const& model, Context const& ctx, StreamOptions const& opts);

private:
    detail::OpenAICompletionsEngine<detail::DeepSeekThinking, detail::OpenAICompat> engine_;
};

/// @brief Anthropic Messages 协议 Provider。实现见 src/provider_anthropic.cpp。
class AnthropicMessagesProvider {
public:
    explicit AnthropicMessagesProvider(EndpointConfig config);

    std::generator<StreamEvent> stream(ModelView const& model, Context const& ctx, StreamOptions const& opts = {});
    Result<ChatResponse> complete(ModelView const& model, Context const& ctx, StreamOptions const& opts = {});
    asio::awaitable<void> stream_async(ModelView const& model, Context const& ctx, StreamOptions const& opts, AsyncStream<StreamEvent> sink);
    asio::awaitable<Result<ChatResponse>> complete_async(ModelView const& model, Context const& ctx, StreamOptions const& opts);
};

/// @brief Gemini GenerateContent 协议 Provider。实现见 src/provider_gemini.cpp。
class GeminiGenerateContentProvider {
public:
    explicit GeminiGenerateContentProvider(EndpointConfig config);

    std::generator<StreamEvent> stream(ModelView const& model, Context const& ctx, StreamOptions const& opts = {});
    Result<ChatResponse> complete(ModelView const& model, Context const& ctx, StreamOptions const& opts = {});
    asio::awaitable<void> stream_async(ModelView const& model, Context const& ctx, StreamOptions const& opts, AsyncStream<StreamEvent> sink);
    asio::awaitable<Result<ChatResponse>> complete_async(ModelView const& model, Context const& ctx, StreamOptions const& opts);
};

}  // namespace agent
