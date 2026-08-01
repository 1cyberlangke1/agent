#pragma once

// LLM 调用层聚合头：一次 include 引入全部类型与 Provider。
//
// 快速开始：
//   // 1. 查模型表拿 ModelView（内置表或 register_model 注册）
//   auto model = *ModelRegistry::find_model("deepseek-v4-flash");
//   // 2. 构造 Provider（按厂商选对应头，见各 Provider 头顶部使用说明）
//   DeepSeekProvider deepseek({.name="deepseek", .api_key=KEY, .base_url="https://api.deepseek.com"});
//   // 3. 调用四接口：
//   auto resp = deepseek.complete(model, ctx);                              // 同步非流式
//   for (auto& ev : deepseek.stream(model, ctx, {.reasoning=ThinkingLevel::High})) ...  // 同步流式
//   //    stream_async / complete_async：异步版本（跑在自己的 io_context 上）
//
// 头文件导航（按需 include 减少编译量）：
//   <agent/llm/types.hpp>        类型层（枚举/Usage/ChatResponse/StreamEvent）
//   <agent/llm/content.hpp>      Message / ContentBlock（对话消息内容）
//   <agent/llm/options.hpp>      EndpointConfig / Context / StreamOptions
//   <agent/llm/model.hpp>        模型表三结构 + ModelRegistry（register/find/for_each）
//   <agent/llm/providers/*.hpp>  各厂商 Provider（OpenAI/DeepSeek/兼容端点/Gemini/Anthropic）
//   <agent/llm/stream.hpp>       AsyncStream 异步通道
//   <agent/core/result.hpp>      错误处理（Result / Error / Errc）
//
// 厂商注意事项见各 Provider 头顶部：base_url 路径差异（OpenAI/NVIDIA 带 /v1，DeepSeek 不带）、
// 认证方式（Bearer vs x-goog-api-key）、思考行为（gemma 自动 / DeepSeek 移采样参数）等。

#include <agent/core/http_client.hpp>
#include <agent/core/result.hpp>
#include <agent/llm/content.hpp>
#include <agent/llm/engine/openai_completions.hpp>
#include <agent/llm/model.hpp>
#include <agent/llm/options.hpp>
#include <agent/llm/providers/anthropic.hpp>
#include <agent/llm/providers/compatible.hpp>
#include <agent/llm/providers/deepseek.hpp>
#include <agent/llm/providers/gemini.hpp>
#include <agent/llm/providers/openai.hpp>
#include <agent/llm/stream.hpp>
#include <agent/llm/types.hpp>
#include <agent/tools/tools.hpp>
