// Tools 注册表并发安全测试：多线程混合 exec / list / get / reg。
// 只依赖 tools.hpp（不含 <meta>）——顺带验证轻量头文件独立可编译。

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include <agent/tools.hpp>
#include <doctest/doctest.h>

using agent::Result;
using agent::ToolInfo;
using agent::Tools;

namespace {
// 探针计数器：static 存储——注册表持有工具闭包直到进程结束，不能引用栈上变量
std::atomic<int> probe_counter{0};
}

TEST_CASE("Tools concurrent exec/list/get")
{
    auto registered = Tools::reg(
        ToolInfo{"concurrency_probe", "并发探针", R"({"type":"object"})"_json},
        [](nlohmann::json) -> Result<std::string> {
            probe_counter.fetch_add(1, std::memory_order_relaxed);
            return "ok";
        });
    REQUIRE(registered.has_value());

    constexpr int thread_count = 8;
    constexpr int iterations = 500;
    std::atomic<int> failures{0};
    std::vector<std::thread> pool;
    pool.reserve(thread_count);

    for (int t = 0; t < thread_count; ++t) {
        pool.emplace_back([&failures] {
            for (int i = 0; i < iterations; ++i) {
                Result<std::string> run =
                    Tools::exec("concurrency_probe", nlohmann::json::object());
                if (!run || *run != "ok")
                    failures.fetch_add(1, std::memory_order_relaxed);
                if (Tools::list().empty())
                    failures.fetch_add(1, std::memory_order_relaxed);
                if (!Tools::get("concurrency_probe"))
                    failures.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (std::thread& th : pool)
        th.join();

    CHECK(failures.load() == 0);
    CHECK(probe_counter.load() == thread_count * iterations);
}

TEST_CASE("Tools concurrent registration of distinct names")
{
    constexpr int thread_count = 8;
    std::atomic<int> success_count{0};
    std::vector<std::thread> pool;
    pool.reserve(thread_count);

    for (int t = 0; t < thread_count; ++t) {
        pool.emplace_back([t, &success_count] {
            ToolInfo info{
                "concurrent_reg_" + std::to_string(t),
                "并发注册",
                R"({"type":"object"})"_json
            };
            Result<void> r = Tools::reg(std::move(info),
                [](nlohmann::json) -> Result<std::string> { return "y"; });
            if (r)
                success_count.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (std::thread& th : pool)
        th.join();

    CHECK(success_count.load() == thread_count);
    for (int t = 0; t < thread_count; ++t)
        CHECK(Tools::get("concurrent_reg_" + std::to_string(t)).has_value());
}

TEST_CASE("Tools concurrent duplicate registration resolves to single winner")
{
    constexpr int thread_count = 8;
    std::atomic<int> success_count{0};
    std::vector<std::thread> pool;
    pool.reserve(thread_count);

    for (int t = 0; t < thread_count; ++t) {
        pool.emplace_back([&success_count] {
            Result<void> r = Tools::reg(
                ToolInfo{"duplicate_race", "并发抢注", R"({"type":"object"})"_json},
                [](nlohmann::json) -> Result<std::string> { return "winner"; });
            if (r)
                success_count.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (std::thread& th : pool)
        th.join();

    // 同名并发注册恰好一个成功，其余 Duplicate
    CHECK(success_count.load() == 1);
    CHECK(Tools::get("duplicate_race").has_value());
}
