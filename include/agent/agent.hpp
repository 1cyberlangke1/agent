#pragma once

// 高层 Agent 封装聚合头：一次 include 引入事件 / Agent / 压缩。
//
// 快速开始（开篇示例，见 PLAN_3.md）：
//   #include <agent/agent.hpp>
//   #include <agent/llm.hpp>
//   using namespace agent;
//
//   Agent<DeepSeekProvider> agent{
//       { .name = "deepseek", .api_key = KEY, .base_url = "https://api.deepseek.com" },
//       *ModelRegistry::find_model("deepseek-v4-flash"),
//   };
//   for (AgentEvent const& ev : agent.run({ { Role::User, { Text{ "你好" } } } }))
//       ...
//
// 头文件导航（按需 include 减少编译量）：
//   <agent/agent/agent_event.hpp>   13 种 AgentEvent
//   <agent/agent/agent.hpp>         Agent + DefaultBehaviors + NextTurnUpdate / ContextSnapshot 等
//   <agent/agent/compaction.hpp>    CompactionSettings + 估算 / 切点 / 摘要请求纯函数

#include <agent/agent/agent_event.hpp>
#include <agent/agent/agent.hpp>
#include <agent/agent/compaction.hpp>
