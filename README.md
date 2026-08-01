# agent — C++26 统一 LLM 调用层

一个 C++26 编写的统一多 provider LLM SDK：四种接口（同步/异步 × 流式/非流式），一套统一抽象，底层各厂商协议差异由引擎内部消化。**零虚函数**——所有抽象编译期确定（模板策略 + 组合复用）。

## 特性

- **统一抽象**：消息 / 工具 / 思考等级（`ThinkingLevel`）/ 缓存档位（`CacheRetention`）/ 停止原因（`StopReason`）/ token 用量（`Usage`），用户只看一套语义
- **四接口全覆盖**：`stream`（同步流式）/ `complete`（同步非流式）/ `stream_async`（异步流式）/ `complete_async`（异步非流式），底层同一份异步 SSE 协程
- **组合式引擎**：共享协议引擎模板（如 `OpenAICompletionsEngine<ThinkingPolicy, Compat>`），厂商差异用编译期策略注入——DeepSeek 复用 OpenAI 协议不写两套
- **思考链归一化**：`reasoning_effort`（OpenAI）/ `thinking + reasoning_effort`（DeepSeek）/ `thinking`（Anthropic adaptive/budget）/ `thinkingBudget`（Gemini）全部从统一 `ThinkingLevel` 映射
- **缓存归一化**：`prompt_cache_key`（OpenAI）/ `cache_control`（Anthropic 挂 system/工具/最后 user 消息）/ 自动缓存（DeepSeek/Gemini）
- **工具调用闭环**：反射注册工具（P2996 静态反射）→ 工具 schema → 模型 tool call → 真实执行 → 多轮结果回传
- **多模态图片输入**：统一 `Image` 内容块 → OpenAI `image_url` / Gemini `inlineData` / Anthropic `image` block；
  模型能力由模型表 `supports_image_input` 承载（引擎零模型特判），不支持时自动降级为占位符文本
- **契约级验证**：T0 纯函数 + T1 MockServer 回放官方/真实 fixture + T2 官方 SDK 镜像契约（openai / anthropic SDK 当裁判打破自证循环）
- **传输层**：libcurl `multi_socket` 嫁接 asio 单事件循环，SSE 流式、分层超时、429/5xx 重试、取消、gzip/br 解压、schannel TLS

## 支持的 Provider

| Provider | 引擎 | 验证程度 |
|---|---|---|
| OpenAI 官方 | OpenAI Completions | T0 + T1 官方快照 + T2 官方 SDK 契约 |
| DeepSeek | OpenAI Completions（DeepSeekThinking） | ✅ 真实 API 端到端（文本/思考/工具） |
| 第三方 OpenAI 兼容（vLLM / Ollama / NVIDIA NIM / 网关） | OpenAI Completions（Compatible） | ✅ NVIDIA NIM 真实 API 端到端 |
| Anthropic Messages | Anthropic Messages | T0 + T1 官方格式 + T2 官方 anthropic SDK 契约 + DeepSeek anthropic 兼容端点真实验证 |
| Gemini | Gemini GenerateContent | ✅ 真实 API（gemma）端到端（文本/思考/工具） |

## 快速开始

```cpp
#include <agent/llm.hpp>

using namespace agent;

// 1. 查模型表（内置表 + 运行时注册，纯静态类多线程安全）
ModelView model = *ModelRegistry::find_model("deepseek-chat");

// 2. 建 Provider
DeepSeekProvider deepseek({ .name = "deepseek",
                            .api_key = KEY,
                            .base_url = "https://api.deepseek.com" });

// 3. 对话上下文（系统提示 + 消息 + 工具）
Context ctx{ "你是助手", { Message{ Role::User, { Text{ "北京天气怎么样" } } } } };
ctx.tools.push_back(ToolInfo{ .name = "get_weather", /* ... */ });

// 4. 四接口任意选：同步流式
for (auto& ev : deepseek.stream(model, ctx, { .reasoning = ThinkingLevel::High })) {
    if (ev.type() == StreamEvent::Type::TextDelta)
        std::cout << std::get<TextDelta>(ev.data).text;
}
```

其他 Provider 同构：

```cpp
AnthropicMessagesProvider anthropic({ .name = "anthropic", .api_key = KEY,
                                      .base_url = "https://api.deepseek.com/anthropic" });  // 兼容端点
GeminiGenerateContentProvider gemini({ .name = "gemini", .api_key = KEY,
                                       .base_url = "https://generativelanguage.googleapis.com" });
```

### 思考等级 / 缓存

```cpp
StreamOptions opts;
opts.reasoning = ThinkingLevel::High;            // 统一等级 → 各家原生格式
opts.cache_retention = CacheRetention::Long;     // 统一缓存档位 → 各家机制
opts.session_id = "conv-001";                    // 会话关联
opts.max_tokens = 1024;                          // 不传则不上传（Anthropic 用模型表必填值）
opts.temperature = 0.5;                          // 不传则不上传
```

## 构建

依赖：CMake ≥ 3.26、Ninja、gcc ≥ 16（`-freflection` 静态反射）、libcurl、asio、nlohmann_json。

```powershell
cmake --preset default
cmake --build --preset default
.\build\src\agent.exe
```

`CMakeUserPresets.json`（gcc 路径）不入库；他人 `cmake -B build -G Ninja .` 用自己的编译器。

## 测试

```powershell
cmake --preset default -DAGENT_BUILD_TESTS=ON
cmake --build --preset default --target test_agent
.\build\tests\test_agent.exe

# 全部（含 T2 契约镜像，需 .venv + 官方 SDK）
ctest --test-dir build --output-on-failure
```

验证金字塔：**T0** 协议纯函数（请求体构建 / 流解析）→ **T1** MockServer 回放 fixture（四接口 + 工具调用闭环 + 难样例/猴子测试）→ **T2** 官方 SDK 镜像契约（响应侧 SDK 消费 fixture、请求侧 golden 对比）。

## 目录结构

```
include/agent/
  core/      域无关基础设施（Result / HttpStreamReader / SseParser）
  tools/     工具系统（注册中心 + P2996 反射引擎）
  llm/
    llm.hpp  聚合头（快速开始 + 导航）
    types.hpp       统一枚举 / Usage / ChatResponse / StreamEvent
    content.hpp     Message / ContentBlock
    options.hpp     EndpointConfig / Context / StreamOptions
    stream.hpp      AsyncStream（共享通道）
    stream_facade.hpp  四接口壳 + 同步桥
    model.hpp       模型表三结构 / ModelRegistry
    engine/         共享协议引擎模板（openai_completions）
    providers/      每厂商独立（openai / deepseek / compatible / anthropic / gemini）
src/          镜像实现
tests/        T0/T1 测试 + fixtures + contract（T2）
```

## 致谢

本项目使用以下第三方库：

| 库 | 用途 | 许可证 |
|---|---|---|
| [asio](https://github.com/chriskohlhoff/asio) | 异步事件循环 / 协程 | Boost Software License 1.0 |
| [libcurl](https://github.com/curl/curl) | HTTP 传输（multi_socket 模式） | curl License（MIT 风格） |
| [nlohmann/json](https://github.com/nlohmann/json) | JSON 解析 / 序列化 | MIT |
| [OpenSSL](https://github.com/openssl/openssl) | TLS（非 Windows 平台） | Apache-2.0 / OpenSSL License |
| [zlib](https://github.com/madler/zlib) | gzip 解压 | zlib License |
| [brotli](https://github.com/google/brotli) | br 解压 | MIT |
| [doctest](https://github.com/doctest/doctest) | 测试框架（仅测试） | MIT |

这些库均为宽松许可（permissive），不产生 copyleft 传染：本项目以 MIT 发布，可自由使用 / 链接 / 静态包含上述库，分发二进制时保留各库自身的版权与许可声明即可（`3rdparty/` 子模块已自带）。

## License

[MIT](LICENSE)
