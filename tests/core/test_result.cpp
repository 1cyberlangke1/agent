/// @brief Result 类型单元测试
///
/// 覆盖：
/// - 成功值构造与访问
/// - 错误值构造与访问
/// - bool 转换
/// - value_or 默认值
/// - Error 结构体字段
/// - Errc 枚举完整性
/// - 函数返回 Result
/// - Result<void> 特化
///
/// @input  直接构造或函数返回的 Result 对象
/// @output CHECK 断言成功/失败
/// @behavior 所有断言通过则测试通过，失败则 doctest 报告具体行号

#include <doctest/doctest.h>
#include <agent/core/result.hpp>
#include <string>

using namespace agent;

/// @brief Result 持有成功值时 has_value() 返回 true，* 可取值
TEST_CASE("Result with value") {
    Result<std::string> r("hello");
    CHECK(r.has_value() == true);
    CHECK(*r == "hello");
}

/// @brief Result 持有错误时 has_value() 返回 false，error() 返回 Error
TEST_CASE("Result with error") {
    Result<int> r = std::unexpected(Error{Errc::NotFound, "not found"});
    CHECK(r.has_value() == false);
    CHECK(r.error().code == Errc::NotFound);
    CHECK(r.error().message == "not found");
}

/// @brief if (r) 等价于 has_value()
TEST_CASE("Result bool conversion") {
    Result<double> ok(3.14);
    CHECK(static_cast<bool>(ok) == true);

    Result<double> err = std::unexpected(Error{Errc::ExecutionFailed, "fail"});
    CHECK(static_cast<bool>(err) == false);
}

/// @brief 错误时返回默认值，不抛异常
TEST_CASE("Result value_or") {
    Result<std::string> ok("value");
    CHECK(ok.value_or("default") == "value");

    Result<std::string> err = std::unexpected(Error{Errc::InvalidArgs, "bad"});
    CHECK(err.value_or("default") == "default");
}

/// @brief Error 的 code 和 message 字段独立比较
TEST_CASE("Error equality") {
    Error a{Errc::NotFound, "msg"};
    Error b{Errc::NotFound, "msg"};
    Error c{Errc::ExecutionFailed, "msg"};

    CHECK(a.code == b.code);
    CHECK(a.message == b.message);
    CHECK(a.code != c.code);
}

/// @brief 验证 Errc 枚举所有值编译期存在（没有 break 漏写）
TEST_CASE("All Errc values exist") {
    auto check = [](Errc e) { return static_cast<int>(e) >= 0; };
    CHECK(check(Errc::NotFound));
    CHECK(check(Errc::Duplicate));
    CHECK(check(Errc::InvalidArgs));
    CHECK(check(Errc::ExecutionFailed));
    CHECK(check(Errc::NetworkError));
    CHECK(check(Errc::RateLimited));
    CHECK(check(Errc::AuthError));
    CHECK(check(Errc::ProviderError));
    CHECK(check(Errc::ServerNotConnected));
}

/// @brief 函数返回 Result 成功时，直接 return 值即可隐式转换
TEST_CASE("Result function returning value") {
    auto fn = []() -> Result<std::string> {
        return "success";
    };
    auto r = fn();
    CHECK(r.has_value());
    CHECK(*r == "success");
}

/// @brief 函数返回 Result 错误时，用 std::unexpected 包装
TEST_CASE("Result function returning error") {
    auto fn = []() -> Result<int> {
        return std::unexpected(Error{Errc::RateLimited, "too many requests"});
    };
    auto r = fn();
    CHECK(!r.has_value());
    CHECK(r.error().code == Errc::RateLimited);
}

/// @brief Result<void> 特化：成功 / 错误两种状态
TEST_CASE("Result<void>") {
    Result<void> ok{};
    CHECK(ok.has_value());

    Result<void> err = std::unexpected(Error{Errc::ExecutionFailed, "fail"});
    CHECK(!err.has_value());
    CHECK(err.error().code == Errc::ExecutionFailed);
}
