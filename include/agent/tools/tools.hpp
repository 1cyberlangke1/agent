#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <agent/core/result.hpp>
#include <asio.hpp>
#include <nlohmann/json.hpp>

namespace agent {

/// @brief 注解值载体。字符串通过 define_static_string 转为 static-storage 指针后传入。
///        必须为 structural type 才能在 NTTP 中作为注解值。
///        构造函数 Desc() 在 tools_reflection.hpp 中（依赖 \c &lt;meta&gt; 头）。
struct DescArg
{
    const char* msg;  ///< 注解描述文本（static-storage，进程生命周期有效）
};

/// @brief 工具注册信息。包含名称、描述、参数 JSON Schema。
struct ToolInfo
{
    std::string             name;        ///< 工具名（模型按此调用，唯一）
    std::string             description; ///< 工具描述（模型决定何时调用）
    nlohmann::json          parameters;  ///< 参数 JSON Schema（arguments 校验 + 广告）
};

/// @brief exec() 调用工具函数前的参数预校验策略。
enum class ArgsCheck
{
    /// exec 按 parameters schema 预校验参数（运行时注册的默认值）。
    /// 手写 fn 无需自己做类型检查，垃圾输入在进入 fn 前被拦截。
    Schema,
    /// 跳过预校验，工具函数自行完成校验。
    /// 反射注册（ToolBase）自动使用此项——assign_from_json 已做全量校验
    /// （类型、required、数值范围、枚举值），再跑一遍 schema 校验纯属浪费。
    Tool,
};

/// @brief 工具执行模式（对齐 pi ToolExecutionMode）。
///        工具定义自带（per-tool），覆盖 Agent 全局默认：
///        Default = 跟随 Agent 全局（set_tool_execution_mode）；Sequential / Parallel = 强制该工具。
enum class ToolExecutionMode { Default, Sequential, Parallel };

/// @brief 内部实现。不视为 API，用户不应直接使用。
namespace detail {

/// @brief 透明哈希：unordered_map 支持 string_view 直接查找，免临时 std::string 分配。
class StringHash
{
public:
    using is_transparent = void;
    std::size_t operator()(std::string_view s) const noexcept
    {
        return std::hash<std::string_view>{}(s);
    }
};

/// @brief 注册表条目。注册后不可变（const 共享），exec 复制 shared_ptr 出锁后调用，
///        避免持锁期间执行用户代码（防死锁、防 rehash 失效迭代器）。
struct RegisteredTool
{
    ToolInfo                                            info;
    std::function<Result<std::string>(nlohmann::json)>  fn;
    ArgsCheck                                           check;
    ToolExecutionMode                                   mode = ToolExecutionMode::Default;
};

/// @brief 注册表不可变快照（COW 写一次读多次）：读路径只 atomic load 快照，零锁。
///        reg 重建快照原子替换；旧快照由 shared_ptr 保活，并发读看到一致（可能略旧）视图。
struct ToolsSnapshot
{
    using RegistryMap = std::unordered_map<std::string,
        std::shared_ptr<RegisteredTool const>, StringHash, std::equal_to<>>;
    RegistryMap map;
};

/// @brief 判断 JSON 值是否可按 integer 接受。
///
/// LLM 偶尔把整数写成 3.0（浮点形态），按 JSON Schema 语义数值为整数即算 integer。
/// 限制在 ±2^53（double 精确整数范围）内，超出精度不可靠直接拒绝。
inline bool is_integer_value(nlohmann::json const& j)
{
    if (j.is_number_integer())
        return true;
    if (j.is_number_float()) {
        double v = j.template get<double>();
        return std::trunc(v) == v && std::abs(v) <= 9007199254740992.0;
    }
    return false;
}

/// @brief 执行前参数校验：value 对照 schema 逐节点检查，错误累积到 errors。
///
/// 防御式实现：schema 任何节点畸形（type 非字符串、properties 非对象等）时
/// 跳过该约束而不是崩溃或抛异常——注册期已做 schema 良构校验，此处为双保险。
///
/// 语义与 assign_from_json 对齐：
/// - integer 接受数值为整数的浮点（is_integer_value）
/// - required 仅要求键存在；null 是否合法交给属性自身的 type 判定
/// - schema 未声明的多余字段忽略（宽容输入）
///
/// @param value  待校验的 JSON 参数
/// @param schema JSON Schema 节点
/// @param path   当前路径（错误信息前缀）
/// @param errors 错误累积器，空表示校验通过
inline void validate_args(
    nlohmann::json const& value,
    nlohmann::json const& schema,
    std::string const& path,
    std::vector<std::string>& errors)
{
    if (!schema.is_object())
        return;

    // ── type 检查（支持 "string" 与 ["string","null"] 两种形式）── //
    auto type_it = schema.find("type");
    if (type_it != schema.end()) {
        bool matched = false;
        bool checkable = false;
        auto match_one = [&](nlohmann::json const& t) {
            if (!t.is_string())
                return;
            checkable = true;
            std::string_view type_name = t.template get_ref<std::string const&>();
            if (type_name == "null")         matched = matched || value.is_null();
            else if (type_name == "object")  matched = matched || value.is_object();
            else if (type_name == "array")   matched = matched || value.is_array();
            else if (type_name == "string")  matched = matched || value.is_string();
            else if (type_name == "boolean") matched = matched || value.is_boolean();
            else if (type_name == "number")  matched = matched || value.is_number();
            else if (type_name == "integer") matched = matched || is_integer_value(value);
        };
        if (type_it->is_string()) {
            match_one(*type_it);
        } else if (type_it->is_array()) {
            for (auto const& t : *type_it)
                match_one(t);
        }
        if (checkable && !matched) {
            errors.push_back(path.empty()
                ? std::string("arguments type mismatch")
                : path + " type mismatch");
            return;  // 类型不对，后续约束无意义，避免级联误报
        }
        if (value.is_null())
            return;  // null 已被 ["x","null"] 接受，无更多约束可查
    }

    // ── object：required 存在性 + properties 递归 ── //
    if (value.is_object()) {
        auto required_it = schema.find("required");
        if (required_it != schema.end() && required_it->is_array()) {
            for (auto const& r : *required_it) {
                if (!r.is_string())
                    continue;
                std::string_view key = r.template get_ref<std::string const&>();
                if (!value.contains(key))
                    errors.push_back(path + "." + std::string(key) + " is required");
            }
        }
        auto properties_it = schema.find("properties");
        if (properties_it != schema.end() && properties_it->is_object()) {
            for (auto it = value.begin(); it != value.end(); ++it) {
                auto prop_it = properties_it->find(it.key());
                if (prop_it != properties_it->end())
                    validate_args(it.value(), *prop_it, path + "." + it.key(), errors);
            }
        }
    }

    // ── array：长度约束 + items 递归 ── //
    if (value.is_array()) {
        auto take_count = [&](char const* key) -> std::int64_t {
            auto it = schema.find(key);
            if (it != schema.end() && it->is_number_integer())
                return it->template get<std::int64_t>();
            return -1;  // 缺失或畸形 → 不检查
        };
        if (std::int64_t min_items = take_count("minItems");
            min_items >= 0 && std::cmp_less(value.size(), min_items))
            errors.push_back(path + " too few items");
        if (std::int64_t max_items = take_count("maxItems");
            max_items >= 0 && std::cmp_greater(value.size(), max_items))
            errors.push_back(path + " too many items");

        auto items_it = schema.find("items");
        if (items_it != schema.end() && items_it->is_object()) {
            for (std::size_t i = 0; i < value.size(); ++i)
                validate_args(value[i], *items_it,
                    path + "[" + std::to_string(i) + "]", errors);
        }
    }

    // ── enum：值必须在枚举列表内 ── //
    auto enum_it = schema.find("enum");
    if (enum_it != schema.end() && enum_it->is_array()) {
        bool found = false;
        for (auto const& e : *enum_it) {
            if (e == value) {
                found = true;
                break;
            }
        }
        if (!found)
            errors.push_back(path + " not in enum");
    }
}

} // namespace detail

/// @brief 工具注册中心。纯静态类，不可实例化。
///
/// 两种注册方式：
///   编译期: struct X : ToolBase<X> { ... }; + template struct agent::ToolBase<X>;
///           （见 tools_reflection.hpp，schema 自动从反射生成）
///   运行时: Tools::reg(ToolInfo{...}, lambda);
///
/// 线程安全：**COW 不可变快照**——写路径（reg）持 write_mutex_ 重建快照原子替换；
/// 读路径（exec/list/get/resolve/names）只 atomic load 快照，**零锁**（单线程热路径零开销）。
/// exec 复制 shared_ptr<RegisteredTool> 后调用工具函数，不在持锁状态执行用户代码。
class Tools
{
    Tools() = delete;

public:
    /// @brief 执行工具。
    ///
    /// 流程：查找 → （ArgsCheck::Schema 时）按 schema 预校验参数 → 调用工具函数。
    /// 工具函数遵循零异常约定：错误全部通过 Result 返回，不抛异常。
    ///
    /// @param name 工具名
    /// @param args JSON 参数对象
    /// @return 成功返回工具输出文本；失败返回错误
    ///         （NotFound / InvalidArgs / 工具自身返回的错误）
    [[nodiscard]] static Result<std::string> exec(std::string_view name,
        nlohmann::json args);

    /// @brief 运行时注册工具。
    ///
    /// 校验链：名字非空 → fn 非空 → parameters 是良构 JSON Schema
    /// （根节点必须 type:"object"，各节点结构合法）→ 名字未被占用。
    /// 任何一步失败返回 InvalidArgs / Duplicate，注册表不变。
    ///
    /// @param info  工具信息（name/description/parameters）
    /// @param fn    工具函数，接收 JSON 参数返回 Result
    /// @param check exec 前参数预校验策略，默认按 schema 校验
    /// @param mode  该工具的**执行模式**（per-tool，覆盖 Agent 全局；Default = 跟随全局）
    static Result<void> reg(ToolInfo info,
        std::function<Result<std::string>(nlohmann::json)> fn,
        ArgsCheck check = ArgsCheck::Schema,
        ToolExecutionMode mode = ToolExecutionMode::Default);

    /// @brief 注册异步工具：fn 是协程（内部可 co_await async_http_get 等）。
    ///        exec 仍同步返回 Result——库在 exec 时用 io_context 桥接等协程完成。
    /// @param info  工具信息
    /// @param fn    异步工具函数，返回 asio 协程
    /// @param check exec 前参数预校验策略
    /// @param mode  该工具的执行模式（per-tool，覆盖 Agent 全局）
    static Result<void> reg(ToolInfo info,
        std::function<asio::awaitable<Result<std::string>>(nlohmann::json)> fn,
        ArgsCheck check = ArgsCheck::Schema,
        ToolExecutionMode mode = ToolExecutionMode::Default);

    /// @brief 列出所有已注册工具（副本）。
    ///
    /// Tools 只是全量工具仓库，实际传给 LLM 的工具子集由 Content 层决定。
    /// 返回按名字典序排序：输出确定性，不随 unordered_map rehash 变化。
    static std::vector<ToolInfo> list();

    /// @brief 按名称查询工具信息。不存在返回 Errc::NotFound。
    static Result<ToolInfo> get(std::string_view name);

    /// @brief 查询工具的执行模式（per-tool，Default = 跟随 Agent 全局）。不存在返回 Errc::NotFound。
    static Result<ToolExecutionMode> mode(std::string_view name);

    /// @brief 批量按名解析工具定义（一次 shared_lock，避免逐个 get 反复加锁）。
    ///        按传入名字顺序返回；遇到未注册的名字 → Result 错误（NotFound）。
    ///        供引擎 build_params 用：Context.tools 存工具名，这里取定义转各家 schema。
    static Result<std::vector<ToolInfo>> resolve(std::vector<std::string> const& names);

    /// @brief 全部已注册工具名（供调用方「本次要全部工具」时填 Context.tools）。
    static std::vector<std::string> names();

private:
    struct State
    {
        std::mutex write_mutex;                                       ///< 写锁（reg，罕见）
        std::atomic<std::shared_ptr<detail::ToolsSnapshot>> snapshot;  ///< 不可变快照
    };
    /// function-local static：C++11 保证初始化线程安全，静态注册阶段即可用
    static State& state();
    /// 取当前快照；首次调用建空快照。读路径入口，无锁。
    static std::shared_ptr<detail::ToolsSnapshot> load_snapshot();
};

} // namespace agent
