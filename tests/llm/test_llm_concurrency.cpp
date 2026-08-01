// LLM Provider 层并发测试：同一 Provider 上多线程并发调用安全（无共享可变状态）。
// 验证文档承诺：Provider 每次调用独立协程栈与连接，并发多请求安全。

#include <agent/llm/model.hpp>
#include <agent/llm/providers/openai.hpp>
#include <agent/llm/types.hpp>

#include "../core/mock_server.hpp"

#include <doctest/doctest.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

using namespace agent;
using namespace agent::detail;

namespace {

std::vector<agent::test::MockServer::Chunk> sse_response(std::string const& body)
{
    std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        "Connection: close\r\n\r\n";
    return { { { head + body } } };
}

/// 最小文本流（content 增量 + finish=stop）。
std::string text_sse()
{
    return
        "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\","
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n"
        "data: {\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\","
        "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{"
        "\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n"
        "data: [DONE]\n\n";
}

void ensure_plain_model()
{
    static bool done = [] {
        ModelRegistry::register_model(RuntimeModel{
            .id = "t-plain", .context_window = 128000, .max_output_tokens = 4096,
            .reasoning = false,
        });
        return true;
    }();
    (void)done;
}

Context simple_ctx()
{
    Context ctx;
    ctx.messages.push_back(Message{ Role::User, { Text{ "hi" } } });
    return ctx;
}

}  // namespace

TEST_CASE("LLM 并发：8 线程并发 provider.complete 全部成功")
{
    ensure_plain_model();
    auto model = ModelRegistry::find_model("t-plain");
    REQUIRE(model.has_value());

    constexpr int kThreads = 8;
    constexpr int kPerThread = 4;
    constexpr int kTotal = kThreads * kPerThread;

    agent::test::MockServer server;
    for (int i = 0; i < kTotal; ++i)
        server.enqueue({ {}, sse_response(text_sse()) });

    OpenAIProvider provider({ .name = "openai", .api_key = "k", .base_url = server.base_url() });

    std::atomic<int> succeeded{0};
    std::atomic<int> failed{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < kPerThread; ++i) {
                Result<ChatResponse> result = provider.complete(*model, simple_ctx(), StreamOptions{});
                if (result && result->stop_reason == StopReason::Stop)
                    ++succeeded;
                else
                    ++failed;
            }
        });
    }
    for (auto& th : threads)
        th.join();

    CHECK(succeeded == kTotal);
    CHECK(failed == 0);
    CHECK(server.request_count() == kTotal);
    CHECK(server.errors().empty());
}
