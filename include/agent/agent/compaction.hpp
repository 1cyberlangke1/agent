#pragma once

// 压缩默认实现的纯函数载体 + 摘要请求构造。
//
// DefaultBehaviors（agent.hpp）的 should_compact / compact 默认成员内部调用这些函数；
// override 时可选复用。摘要请求构造是「前缀缓存」的关键（对齐 opencode）：
//   - 摘要指令必须作**尾部 user 消息** append 进 messages，**绝不放进 system prompt**——
//     system prompt 一旦变，prompt 前缀全变 → 缓存全废（pi 把摘要指令塞
//     SUMMARIZATION_SYSTEM_PROMPT 是败笔）。
//   - 摘要请求 StreamOptions：max_tokens = min(model.max_output_tokens, 4096)、
//     thinking 关闭、temperature 0、Context.tools 留空——摘要响应出现 ToolCall
//     → 压缩失败（CompactionFailed），不做任何工具处理。
//
// 已知简化（诚实记录，见 PLAN_3.md 五.5.4）：
//   - find_cut_point 轮内切点是「整条消息」保留而非字符级切片：超预算单条整体保留
//     （可能略超 keep_recent_tokens），避免切断 ToolCall/ToolResult 配对。

#include <agent/core/result.hpp>
#include <agent/llm/content.hpp>
#include <agent/llm/model.hpp>
#include <agent/llm/options.hpp>
#include <agent/llm/types.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace agent {

/// 压缩策略阈值。
struct CompactionSettings {
    int reserve_tokens = 16384;       ///< 触发阈值余量：estimate > context_window - reserve 才压
    int keep_recent_tokens = 8000;    ///< 保留尾部 token 预算（对齐 opencode keep.tokens）
    int tail_turns = 2;               ///< 保留尾部轮数（对齐 opencode tail_turns）
};

/// 摘要消息前缀（按此前缀可识别摘要消息）。
inline constexpr std::string_view kCompactionSummaryPrefix = "[会话摘要]\n";

/// 判断是否为摘要消息（Role::User 且首个 Text 块带前缀）。
bool is_compaction_summary(Message const& message);

/// 提取摘要消息的正文（去前缀）；非摘要消息返回空。
std::string extract_summary_text(Message const& message);

/// 单条消息 token 估算：字符数/4，image 按 4800 折算（对齐 pi estimateTokens）。
int estimate_tokens_default(Message const& message);

/// 整段对话 token 估算（usage 锚点真实优先、估算兜底）：
/// 找最近一条带真实 token 计数的 assistant 消息（其 input+output = 该请求的真实累计）
/// 当锚点 + 其后消息逐条估算；没找到 → 全量逐条估算（chars/4 兜底）。
int estimate_context_tokens(std::vector<Message> const& messages);

/// 触发决策（默认）：context_tokens > context_window - reserve。
bool should_compact(int context_tokens, int context_window, CompactionSettings const& settings);

/// 切点决策（默认，对齐 opencode select + splitTurn 的近似）：
///   1) 把消息切成 turn（每个 user 消息 = 一个 turn 起点）
///   2) 取最近 tail_turns 个 turn 作为候选保留区
///   3) 从后往前累加每个 turn 的 token：累计 <= keep_recent_tokens → 整轮保留；
///      超出 → 在该 turn 内部从后往前保留消息（超预算单条整体保留，见文件头简化）。
///   返回保留段起点 index（保留 = [cut, end)，摘要 = [0, cut)）。
std::size_t find_cut_point(std::vector<Message> const& messages, CompactionSettings const& settings);

/// 摘要指令文本（含 <previous-summary> 增量段）。
std::string build_summary_instruction(std::string_view previous_summary);

/// 摘要请求构造（纯函数）。返回 nullopt = 未触发（估算未超限 / 无可压内容 / 切点在起点）。
struct SummaryRequest {
    Context ctx;            ///< system 不变、messages = [摘要段] + [尾部 user 指令]、tools 留空
    StreamOptions opts;     ///< max_tokens 钳制 / thinking 关 / temperature 0
    std::size_t cut = 0;    ///< 保留段起点（摘要段 = [0, cut)）
};
std::optional<SummaryRequest> build_summary_request(
    std::vector<Message> const& messages, ModelView const& model,
    CompactionSettings const& settings, std::string_view system_prompt,
    std::string_view previous_summary);

/// 校验摘要响应：出现 ToolCall / 空文本 → CompactionFailed（message 写明原因）；否则返回摘要文本。
Result<std::string> validate_summary_response(ChatResponse const& response);

/// 组装压缩结果：新对话 = [摘要消息] + [保留段]，旧段丢弃。
std::vector<Message> apply_summary(std::vector<Message> const& messages, std::size_t cut,
                                   std::string const& summary);

// ───────────────────── 实现 ─────────────────────

inline bool is_compaction_summary(Message const& message)
{
    if (message.role != Role::User)
        return false;
    for (auto const& block : message.content)
        if (auto text = std::get_if<Text>(&block))
            return text->text.starts_with(kCompactionSummaryPrefix);
    return false;
}

inline std::string extract_summary_text(Message const& message)
{
    if (message.role != Role::User)
        return {};
    for (auto const& block : message.content)
        if (auto text = std::get_if<Text>(&block))
            if (text->text.starts_with(kCompactionSummaryPrefix))
                return text->text.substr(kCompactionSummaryPrefix.size());
    return {};
}

inline int estimate_tokens_default(Message const& message)
{
    int tokens = 0;
    for (auto const& block : message.content) {
        tokens += std::visit([](auto const& b) -> int {
            using T = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<T, Text>)
                return static_cast<int>(b.text.size() / 4);
            else if constexpr (std::is_same_v<T, Thinking>)
                return static_cast<int>(b.text.size() / 4);
            else if constexpr (std::is_same_v<T, Image>)
                return 4800;
            else if constexpr (std::is_same_v<T, ToolCall>) {
                int n = static_cast<int>(b.name.size() / 4);
                std::string const& args = b.arguments.is_string()
                    ? b.arguments.template get_ref<std::string const&>()
                    : b.arguments.dump();
                return n + static_cast<int>(args.size() / 4);
            }
            else if constexpr (std::is_same_v<T, ToolResult>)
                return static_cast<int>((b.output.size() + 8) / 4);
            else
                return 0;
        }, block);
    }
    return tokens;
}

inline int estimate_context_tokens(std::vector<Message> const& messages)
{
    // 找最近一条带真实 token 计数的 assistant 消息（usage 锚点）：
    // 锚点 = 该请求的真实 input+output（含此前全部上下文），其后消息逐条估算。
    std::optional<int> anchor;
    std::size_t anchor_tail = 0;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        Message const& m = messages[i];
        if (m.role == Role::Assistant && m.input_tokens() && m.output_tokens()) {
            anchor = static_cast<int>(*m.input_tokens() + *m.output_tokens());
            anchor_tail = i + 1;
        }
    }
    int total = anchor.value_or(0);
    for (std::size_t i = anchor_tail; i < messages.size(); ++i)
        total += estimate_tokens_default(messages[i]);
    return total;   // 无锚点 → 全量估算（anchor_tail = 0，total 从 0 累加）
}

inline bool should_compact(int context_tokens, int context_window, CompactionSettings const& settings)
{
    return context_tokens > context_window - settings.reserve_tokens;
}

inline std::size_t find_cut_point(std::vector<Message> const& messages, CompactionSettings const& settings)
{
    if (messages.empty())
        return 0;

    // 1) 切 turn：每个 User 消息 = turn 起点
    struct Turn {
        std::size_t begin;
        std::size_t end;
        int tokens;
    };
    std::vector<Turn> turns;
    std::size_t i = 0;
    while (i < messages.size()) {
        if (messages[i].role != Role::User) {
            ++i;
            continue;
        }
        std::size_t begin = i;
        ++i;
        while (i < messages.size() && messages[i].role != Role::User)
            ++i;
        int tokens = 0;
        for (std::size_t k = begin; k < i; ++k)
            tokens += estimate_tokens_default(messages[k]);
        turns.push_back({ begin, i, tokens });
    }
    if (turns.empty())
        return 0;   // 无 user 消息 → 不压

    // 2) 候选保留区 = 最后 tail_turns 个 turn
    std::size_t tail = settings.tail_turns > 0 ? static_cast<std::size_t>(settings.tail_turns) : 1;
    std::size_t first_candidate = turns.size() > tail ? turns.size() - tail : 0;

    // 3) 从后往前累加 turn token：<= keep_recent_tokens 整轮保留，否则轮内切
    std::size_t cut = turns[first_candidate].begin;   // 兜底：整个候选区保留
    int budget = settings.keep_recent_tokens;
    for (std::size_t ti = turns.size(); ti > first_candidate; --ti) {
        Turn const& turn = turns[ti - 1];
        if (turn.tokens <= budget) {
            cut = turn.begin;
            budget -= turn.tokens;
            continue;
        }
        // 该 turn 超出剩余预算 → 轮内从后往前保留消息；超预算单条整体保留
        std::size_t keep_from = turn.end;
        int remain = budget;
        for (std::size_t k = turn.end; k > turn.begin; --k) {
            int message_tokens = estimate_tokens_default(messages[k - 1]);
            if (message_tokens > remain) {
                keep_from = (remain > 0) ? (k - 1) : k;   // 有预算 → 该条整体保留；无预算 → 整条进摘要
                break;
            }
            remain -= message_tokens;
            keep_from = k - 1;
        }
        cut = keep_from;
        break;
    }
    return cut;
}

inline std::string build_summary_instruction(std::string_view previous_summary)
{
    std::string instruction =
        "请把上面的对话压缩成一个简洁的会话摘要，必须保留：\n"
        "- 用户的目标与已完成的事项\n"
        "- 关键决策与结论\n"
        "- 待办 / 未完成事项\n"
        "- 重要的事实与上下文\n";
    if (!previous_summary.empty()) {
        instruction += "\n这是之前的摘要，请在此基础上合并更新（不要重复已有内容）：\n";
        instruction += previous_summary;
        instruction += "\n";
    }
    return instruction;
}

inline std::optional<SummaryRequest> build_summary_request(
    std::vector<Message> const& messages, ModelView const& model,
    CompactionSettings const& settings, std::string_view system_prompt,
    std::string_view previous_summary)
{
    if (!agent::should_compact(estimate_context_tokens(messages), model.context_window, settings))
        return std::nullopt;
    std::size_t cut = find_cut_point(messages, settings);
    if (cut == 0 || cut >= messages.size())
        return std::nullopt;   // 无可压（全对话都在保留预算内）或无保留内容

    SummaryRequest request;
    request.cut = cut;
    request.ctx.system_prompt = std::string(system_prompt);   // system 不变
    request.ctx.messages.assign(messages.begin(), messages.begin() + static_cast<std::ptrdiff_t>(cut));
    request.ctx.messages.push_back(
        Message{ Role::User, { Text{ build_summary_instruction(previous_summary) } } });
    // ctx.tools 默认空 = 留空
    int max_out = model.max_output_tokens > 0 ? model.max_output_tokens : 4096;
    request.opts.max_tokens = std::min(max_out, 4096);
    request.opts.reasoning = ThinkingLevel::Off;
    request.opts.temperature = 0;
    return request;
}

inline Result<std::string> validate_summary_response(ChatResponse const& response)
{
    for (auto const& block : response.content)
        if (std::get_if<ToolCall>(&block))
            return std::unexpected(Error{ Errc::CompactionFailed, "压缩摘要响应出现工具调用" });
    std::string summary;
    for (auto const& block : response.content)
        if (auto text = std::get_if<Text>(&block))
            summary += text->text;
    if (summary.empty())
        return std::unexpected(Error{ Errc::CompactionFailed, "压缩摘要为空" });
    return summary;
}

inline std::vector<Message> apply_summary(std::vector<Message> const& messages, std::size_t cut,
                                          std::string const& summary)
{
    std::vector<Message> result;
    result.reserve(1 + (messages.size() - cut));
    result.push_back(
        Message{ Role::User, { Text{ std::string(kCompactionSummaryPrefix) + summary } } });
    result.insert(result.end(), messages.begin() + static_cast<std::ptrdiff_t>(cut), messages.end());
    return result;
}

}  // namespace agent
