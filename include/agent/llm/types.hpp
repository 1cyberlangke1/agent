#pragma once

// LLM 调用层统一类型层（L1）：枚举 / Usage / ChatResponse / StreamEvent。
// 不包含模型表（model.hpp）、请求配置（llm_options.hpp）、异步通道（async_stream.hpp）、
// 策略（llm_policy.hpp）、引擎/Provider（provider.hpp）。

#include <agent/llm/content.hpp>
#include <agent/core/result.hpp>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

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
    int input_tokens = 0;          ///< 输入 token
    int output_tokens = 0;         ///< 输出 token
    int cache_read_tokens = 0;     ///< 缓存命中读取
    int cache_write_tokens = 0;    ///< 缓存写入（Anthropic 有）
    int total_tokens = 0;          ///< 总计
    double cost = 0;               ///< 美元；引擎 parse_usage 不填（默认 0），Agent 算好后填；向后兼容
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
/// Done/Error 皆缺时壳层兜底合成 Error（见终结契约）。
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
class StreamEvent {
public:
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

}  // namespace agent
