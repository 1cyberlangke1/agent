#pragma once

// Agent 层事件（对齐 pi AgentEvent，13 种）。
//
// ⚠️ ToolCallDelta / ToolCallEnd 复用 types.hpp 的同名类型（同形状）——
//    不在此重新定义，避免与 LLM 流事件撞名；MessageUpdate.delta 也用
//    types.hpp 的 TextDelta / ThinkingDelta。AgentEvent.data 的 variant
//    index 与 Type 枚举顺序严格一致（type() 直接强转 index）。

#include <agent/core/result.hpp>
#include <agent/llm/content.hpp>
#include <agent/llm/types.hpp>

#include <nlohmann/json.hpp>

#include <string>
#include <variant>
#include <vector>

namespace agent {

/// run 开始（纯信号）
struct AgentStart {};

/// run 结束，携带完整对话
struct AgentEnd {
    std::vector<Message> messages;  ///< 完整对话消息（user + assistant + toolResult）
};

/// 一轮开始（纯信号；一轮 = 一次回复 + 其工具往返）
struct TurnStart {};

/// 一轮结束
struct TurnEnd {
    Message assistant_message;              ///< 本轮 assistant 回复
    std::vector<ToolResult> tool_results;   ///< 本轮工具结果
};

/// 一条消息出现（user / assistant / toolResult 都发）
struct MessageStart {
    Message message;  ///< 出现的消息
};

/// assistant 流式增量
struct MessageUpdate {
    Message message;                              ///< 当前 partial 消息
    std::variant<TextDelta, ThinkingDelta> delta;  ///< 本次增量（正文/思考）
};

/// 一条消息完成
struct MessageEnd {
    Message message;  ///< 完成的消息
};

/// 工具开跑（ToolCallDelta / ToolCallEnd 复用 types.hpp，见文件头注释）
struct ToolExecStart {
    std::string id;             ///< 工具调用 id（对齐 ToolCallDelta.id）
    std::string name;           ///< 工具名
    nlohmann::json arguments;   ///< 工具参数（JSON 对象）
};

/// 工具部分结果（当前同步工具执行不产生流式部分结果，此事件类型保留供流式工具使用）
struct ToolExecUpdate {
    std::string id;             ///< 工具调用 id
    std::string name;           ///< 工具名
    nlohmann::json arguments;   ///< 工具参数（JSON 对象）
    std::string partial_result; ///< 已产生的部分结果文本
};

/// 工具跑完
struct ToolExecEnd {
    std::string id;         ///< 工具调用 id
    std::string name;       ///< 工具名
    ToolResult result;      ///< 最终工具结果
};

/// 失败不裸抛（对齐 pi failure message）
struct AgentError {
    Error error;  ///< 错误详情（码 + 详情文本）
};

/// 统一事件（variant 风格，对齐 StreamEvent）。聚合初始化：
/// `AgentEvent{ MessageStart{ msg } }` 直接初始化 data。
class AgentEvent {
public:
    /// 事件种类，与 data 的 variant 下标严格一一对应（type() 直接强转 index）。
    enum class Type {
        AgentStart, AgentEnd, TurnStart, TurnEnd,
        MessageStart, MessageUpdate, MessageEnd,
        ToolCallDelta, ToolCallEnd,
        ToolExecStart, ToolExecUpdate, ToolExecEnd, AgentError,
    };
    /// 事件载荷（variant 风格，聚合初始化如 `AgentEvent{ MessageStart{ msg } }`）。
    std::variant<AgentStart, AgentEnd, TurnStart, TurnEnd, MessageStart, MessageUpdate,
                 MessageEnd, ToolCallDelta, ToolCallEnd,
                 ToolExecStart, ToolExecUpdate, ToolExecEnd, AgentError> data;

    /// 类型判别从 variant 下标推导——单一真相源，不存冗余 type 字段。
    Type type() const { return static_cast<Type>(data.index()); }
};
static_assert(std::variant_size_v<decltype(AgentEvent::data)> == 13,
              "AgentEvent 载荷类型数与 Type 枚举必须一致");

}  // namespace agent
