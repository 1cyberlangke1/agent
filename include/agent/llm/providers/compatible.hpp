#pragma once

// 第三方 OpenAI 兼容端点统一 Provider（vLLM / Ollama / Moonshot / NVIDIA NIM / 自建网关等）。
// 使用方式与注意事项见 OpenAICompatibleProvider 的 Doxygen 注释。

#include <agent/llm/engine/openai_completions.hpp>
#include <agent/llm/providers/openai.hpp>
#include <agent/llm/stream_facade.hpp>

namespace agent {

/// @brief OpenAI 兼容端点统一 Provider（第三方 OpenAI 兼容 API）。
///
/// 四接口：stream（同步流式）/ complete（同步非流式）/
/// stream_async（异步流式）/ complete_async（异步非流式）。
///
/// @par 用法
/// ```
/// ModelRegistry::register_model(RuntimeModel{.id="my-model", ...});   // 第三方模型注册
/// auto model = *ModelRegistry::find_model("my-model");
/// OpenAICompatibleProvider provider({.name="vllm", .api_key=KEY, .base_url="http://127.0.0.1:8000/v1"});
/// auto resp = provider.complete(model, ctx);
/// ```
///
/// @note 注意事项：
/// - base_url 需与端点兼容路径匹配：NVIDIA NIM 是 https://integrate.api.nvidia.com/v1（带 /v1）；
///   Ollama/vLLM 本地端点通常是 http://host:port/v1
/// - max_tokens 请求字段用 max_tokens（第三方端点多数只实现旧字段，非 max_completion_tokens）
/// - 思考：reasoning 传统一 ThinkingLevel → reasoning_effort（端点支持才有效，默认不发）
/// - 认证：Authorization: Bearer \c &lt;api_key&gt;（大多数第三方端点同 OpenAI）
using OpenAICompatibleProvider =
    detail::StreamFacade<detail::OpenAICompletionsEngine<detail::OpenAIThinking, detail::OpenAICompatibleCompat>>;

}  // namespace agent
