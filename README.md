# agent — C++26 实验性 LLM / Agent 库

> ## ⚠️ 实验性质声明
>
> **这是一个实验性（experimental）项目，不是生产就绪库。**
>
> - **API 不稳定**：公开接口仍在演进，未来提交可能随时破坏性变更，不保证向后兼容。
> - **功能覆盖不完整**：部分能力（如真实并发工具执行、跨平台 TLS、极端网络场景）验证有限，
>   仅在特定环境（Windows + schannel / gcc 16 + asio）实测过。
> - **生产慎用**：如果你要用它做线上产品，请自行充分测试、评估风险；实验性质意味着我们
>   不为稳定性 / 安全性 / 性能做任何承诺。
> - **欢迎实验**：这正是它的用途——快速尝试 C++26 协程 + 静态反射 + LLM/Agent 的组合。
>
> 继续阅读即表示你理解并接受上述风险。

---

## 简介

一个 C++26 编写的统一多 provider LLM SDK **外加**高层 Agent 封装：从「裸调用各家 LLM API」到
「一个会自己调工具的 Agent」，四层能力由浅入深：

```
LLM 调用层   统一抽象 6 家 provider，四接口（同步/异步 × 流式/非流式），零虚函数
工具系统     反射注册（P2996 静态反射）或运行时注册，per-tool 执行模式
HTTP 传输层  完整 libcurl 封装（方法/body 形态/认证/TLS/代理/cookie/连接复用）+ HttpClient
高层 Agent   Agent<Provider, Behaviors>：自动工具往返、钩子、队列、压缩、执行门控
```

**零虚函数**——所有抽象编译期确定（模板策略 + 组合复用），无 vtable、无 `std::function`
（测试除外）。

## 特性

### LLM 调用层
- **统一抽象**：消息 / 工具 / 思考等级（`ThinkingLevel`）/ 缓存档位（`CacheRetention`）/
  停止原因（`StopReason`）/ token 用量（`Usage` + cost），用户只看一套语义
- **四接口全覆盖**：`stream` / `complete` / `stream_async` / `complete_async`
- **组合式引擎**：共享协议引擎模板，厂商差异用编译期策略注入（DeepSeek 复用 OpenAI 协议）
- **思考链 / 缓存 / 多模态图片**统一归一化；模型能力由模型表承载（引擎零模型特判）
- **契约级验证**：T0 纯函数 + T1 MockServer 回放 + T2 官方 SDK 镜像契约

### 工具系统
- **反射注册**（P2996 静态反射）：`struct X : ToolBase<X> { struct params_type {...}; static Result invoke(...); }`
  → 工具名 / 描述 / JSON Schema 全自动生成，跨 TU 静态注册
- **运行时注册**：`Tools::reg(ToolInfo{...}, fn)`，支持复杂嵌套参数 schema、异步工具
- **per-tool 执行模式**：每个工具可声明 Parallel / Sequential，覆盖 Agent 全局

### HTTP 传输层（libcurl 完整封装）
- `HttpRequest` 全映射：任意方法、body 三形态（字符串 / multipart / 大文件流式上传）、
  认证（Basic/Bearer/Digest/Ntlm/Negotiate）、TLS（verify/CA/客户端证书）、重定向、代理（含 SOCKS）、cookie
- 便捷层：`http_get_json` / `http_post_json` / `http_post_form` / `http_upload_file`
  （非 2xx 自动归一化为错误）
- `HttpClient`：常驻连接池 keep-alive 复用 + cookie jar，纯协程零线程
- libcurl `multi_socket` 嫁接 asio 单事件循环：SSE 流式、分层超时、429/5xx 重试、取消、gzip/br 解压

### 高层 Agent
- **自动工具往返**：模型调用工具 → 真实执行 → 结果回传 → 继续，直到自然结束
- **钩子系统**（`DefaultBehaviors`，非虚名字隐藏零 vtable）：`transform_context` /
  `before_tool_call` / `after_tool_call` / `prepare_next_turn`（多模型编排）/ `should_stop` /
  `get_api_key` / `run_start` / `before_request` / `before_payload`
- **执行门控**：`set_tools` 显式指定允许的工具，未开放的调用被拒绝回传（安全边界）
- **steer / follow_up 队列**：运行中插队 / 将停时追加
- **压缩**（compaction）：run 内防爆 + 手动触发，尾部保留 + 增量摘要 + 旧段丢弃
- **成本监控**：usage 锚点 + 单价计算 `context_cost()`

## 支持的 Provider

| Provider | 引擎 | 验证程度 |
|---|---|---|
| OpenAI 官方 | OpenAI Completions | T0 + T1 官方快照 + T2 官方 SDK 契约 |
| DeepSeek | OpenAI Completions（DeepSeekThinking） | ✅ 真实 API 端到端（文本/思考/工具） |
| 第三方 OpenAI 兼容（vLLM / Ollama / NVIDIA NIM / 网关） | OpenAI Completions（Compatible） | ✅ NVIDIA NIM 真实 API 端到端 |
| Anthropic Messages | Anthropic Messages | T0 + T1 官方格式 + T2 官方 anthropic SDK 契约 + DeepSeek anthropic 兼容端点真实验证 |
| Gemini | Gemini GenerateContent | ✅ 真实 API（gemma）端到端（文本/思考/工具） |
| Agnes（sglang） | OpenAI Completions（AgnesThinking） | ✅ 真实 API 端到端（文本/思考/多模态看图） |

## 快速开始

### 用 Agent（最高层，推荐先试这个）

```cpp
#include <agent/agent.hpp>
#include <agent/llm.hpp>
#include <iostream>
using namespace agent;

// 1. 查模型表
ModelView model = *ModelRegistry::find_model("deepseek-v4-flash");

// 2. 建 Agent（自动工具往返 + 压缩内建）
Agent<DeepSeekProvider> agent(
    { .name = "deepseek", .api_key = KEY, .base_url = "https://api.deepseek.com" },
    model,
    {},                    // 默认 Behaviors（钩子 + 压缩策略一体）
    "你是一个会使用工具的助手。");
agent.set_tools({ "weather_now" });   // ⚠️ 默认不开放任何工具，须显式指定

// 3. 消费事件流（正文增量在 MessageUpdate.delta 里）
for (AgentEvent const& ev : agent.run({ { Role::User, { Text{ "北京今天天气" } } } })) {
    if (ev.type() == AgentEvent::Type::MessageUpdate) {
        auto const& update = std::get<MessageUpdate>(ev.data);
        if (auto* td = std::get_if<TextDelta>(&update.delta))
            std::cout << td->text << std::flush;
    }
}
```

### 直接调 LLM（跳过 Agent）

```cpp
#include <agent/llm.hpp>
using namespace agent;

Context ctx{ "你是助手", { Message{ Role::User, { Text{ "你好" } } } } };
for (auto& ev : deepseek.stream(model, ctx, { .reasoning = ThinkingLevel::High })) {
    if (ev.type() == StreamEvent::Type::TextDelta)
        std::cout << std::get<TextDelta>(ev.data).text;
}
```

其他 Provider 同构：`AnthropicMessagesProvider` / `GeminiGenerateContentProvider` / `AgnesProvider` /
`OpenAICompatibleProvider`（vLLM / Ollama / 网关）。

## 核心概念速览

| 概念 | 一句话 |
|---|---|
| `Agent<Provider, Behaviors>` | 会自己调工具的对话引擎，消费事件流驱动 |
| `DefaultBehaviors` | 钩子 + 压缩策略一体；override 用模板方法（非虚名字隐藏） |
| `AgentEvent` | 13 种事件（AgentStart/End、Turn、Message、ToolExec、AgentError…） |
| `ToolBase<T>` / `Tools::reg` | 反射 / 运行时注册工具 |
| `Tools` 注册表 | 全局、COW 快照读路径零锁、per-tool 执行模式 |
| `HttpRequest` / `HttpClient` | 完整 curl 能力 / 连接复用 |
| `ModelRegistry` | 全局模型表（内置 + 运行时注册，多线程安全） |

## 构建

依赖：CMake ≥ 3.26、Ninja、gcc ≥ 16（`-freflection` 静态反射）、asio、nlohmann_json、libcurl。

```powershell
cmake --preset default
cmake --build --preset default
.\build\src\agent.exe
```

`CMakeUserPresets.json`（gcc 路径）不入库；他人 `cmake -B build -G Ninja .` 用自己的编译器。

## 示例（真实 API 跑通）

| 示例 | 演示 |
|---|---|
| `agent_chat` | Agent 自动工具往返 + set_tools 门控 + 手动压缩 + 真实查天气 API |
| `deepseek_chat` | LLM 层四接口 + 思考/缓存 + 反射工具闭环 |
| `agnes_chat` | Agnes provider 文本/思考/多模态看图（需 agnes key） |

```powershell
cmake --build --preset default --target agent_chat
.\build\examples\agent_chat.exe --key <your_key>
```

## 测试

```powershell
cmake --preset default -DAGENT_BUILD_TESTS=ON
cmake --build --preset default --target test_agent
.\build\tests\test_agent.exe

# 全部（含 T2 契约镜像，需 .venv + 官方 SDK）
ctest --test-dir build --output-on-failure
```

验证金字塔：**T0** 协议纯函数 → **T1** MockServer 回放（四接口 + 工具闭环 + 难样例）→ **T2** 官方 SDK 镜像契约。

## 目录结构

```
include/agent/
  core/      域无关基础设施（Result / HttpRequest·HttpClient / SseParser）
  tools/     工具系统（注册中心 + P2996 反射引擎 + per-tool 模式）
  llm/       统一 LLM 抽象（types / content / options / model / engine / providers）
  agent/     高层 Agent（agent_event / agent / compaction）
examples/    可运行示例（agent_chat / deepseek_chat / agnes_chat）
tests/       T0/T1 测试 + fixtures + contract（T2）
```

## 已知限制（实验性质的部分诚实交代）

- **平台**：主要在 Windows + schannel + gcc 16 上开发测试；Linux/macOS 未经完整验证。
- **工具执行**：Parallel 用 `std::async` 真并发，但同步工具无流式部分结果（`ToolExecUpdate` 事件保留不发）。
- **压缩**：压缩后旧上下文直接丢弃（如需存档请自行读取）。
- **Agent 并发**：同一 Agent 不支持并发 run；`abort()` 跨线程为最佳努力。
- **Provider 覆盖**：OpenAI / DeepSeek / Agnes / 兼容端点验证最充分；Anthropic 无官方 key，
  Gemini 走 gemma 端点，覆盖相对弱。

## 致谢 / License

第三方库：asio（BSL-1.0）、libcurl（curl License）、nlohmann/json（MIT）、OpenSSL、zlib、brotli、
doctest（测试）。均为宽松许可，本项目以 [MIT](LICENSE) 发布。
