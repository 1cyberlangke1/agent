#include <algorithm>
#include <mutex>
#include <optional>

#include <agent/tools/tools.hpp>

namespace agent {

namespace {

/// JSON Schema type 名白名单（三家 provider 通用集合 + null）
bool valid_type_name(std::string_view type_name)
{
    return type_name == "string" || type_name == "integer" || type_name == "number"
        || type_name == "boolean" || type_name == "array" || type_name == "object"
        || type_name == "null";
}

/// @brief 递归检查 schema 节点结构良构。
///
/// 只检查会导致执行期校验器出错的结构问题，未知关键字放行
/// （JSON Schema 语义：不认识的关键字是注解，合法）：
/// - type 若存在须是白名单字符串或其非空数组
/// - properties 若存在须是 object，且每个属性递归良构
/// - items 若存在须递归良构
/// - required 若存在须是字符串数组，且引用的字段在 properties 中声明
///   （引用未声明字段几乎必然是工具作者的拼写错误，注册期拒绝以便早发现）
/// - enum 若存在须是非空数组
/// - minItems/maxItems 若存在须是非负整数
/// - 嵌套深度限制 64 层，防御恶意/失控的深 schema
bool schema_node_ok(nlohmann::json const& node, int depth)
{
    if (depth > 64 || !node.is_object())
        return false;

    if (auto it = node.find("type"); it != node.end()) {
        if (it->is_string()) {
            if (!valid_type_name(it->template get_ref<std::string const&>()))
                return false;
        } else if (it->is_array()) {
            if (it->empty())
                return false;
            for (auto const& t : *it) {
                if (!t.is_string()
                    || !valid_type_name(t.template get_ref<std::string const&>()))
                    return false;
            }
        } else {
            return false;
        }
    }

    if (auto it = node.find("properties"); it != node.end()) {
        if (!it->is_object())
            return false;
        for (auto const& [key, sub] : it->items()) {
            if (!schema_node_ok(sub, depth + 1))
                return false;
        }
    }

    if (auto it = node.find("items"); it != node.end()) {
        if (!schema_node_ok(*it, depth + 1))
            return false;
    }

    if (auto it = node.find("required"); it != node.end()) {
        if (!it->is_array())
            return false;
        auto properties_it = node.find("properties");
        for (auto const& r : *it) {
            if (!r.is_string())
                return false;
            if (properties_it == node.end()
                || !properties_it->contains(r.template get_ref<std::string const&>()))
                return false;
        }
    }

    if (auto it = node.find("enum"); it != node.end()) {
        if (!it->is_array() || it->empty())
            return false;
    }

    for (char const* key : {"minItems", "maxItems"}) {
        if (auto it = node.find(key); it != node.end()) {
            if (!it->is_number_integer() || it->template get<std::int64_t>() < 0)
                return false;
        }
    }

    return true;
}

/// 工具参数 schema 的根节点要求：必须是 type == "object" 的良构节点
/// （三家 provider 都要求工具参数根是 object）
bool tool_schema_ok(nlohmann::json const& schema)
{
    if (!schema_node_ok(schema, 0))
        return false;
    auto it = schema.find("type");
    return it != schema.end() && it->is_string()
        && it->template get_ref<std::string const&>() == "object";
}

} // namespace

Tools::State& Tools::state()
{
    static State s;
    return s;
}

std::shared_ptr<detail::ToolsSnapshot> Tools::load_snapshot()
{
    State& st = state();
    if (std::shared_ptr<detail::ToolsSnapshot> snap = st.snapshot.load())
        return snap;
    std::lock_guard lock(st.write_mutex);   // 首次建空快照
    if (std::shared_ptr<detail::ToolsSnapshot> snap = st.snapshot.load())
        return snap;
    auto snap = std::make_shared<detail::ToolsSnapshot>();
    st.snapshot.store(std::move(snap));
    return st.snapshot.load();
}

Result<void> Tools::reg(ToolInfo info,
    std::function<Result<std::string>(nlohmann::json)> fn,
    ArgsCheck check, ToolExecutionMode mode)
{
    // 校验链全部在锁外完成，持锁窗口只覆盖查重 + COW 替换
    if (info.name.empty())
        return std::unexpected(Error{Errc::InvalidArgs, "tool name is empty"});
    if (!fn)
        return std::unexpected(Error{Errc::InvalidArgs,
            "tool '" + info.name + "' function is empty"});
    if (!tool_schema_ok(info.parameters))
        return std::unexpected(Error{Errc::InvalidArgs,
            "tool '" + info.name + "' has invalid parameters schema "
            "(root must be a well-formed JSON Schema with type \"object\")"});

    // 先构造完成条目再插入：key 从条目内部引用，规避 std::move(info) 的评估顺序陷阱
    auto tool = std::make_shared<detail::RegisteredTool const>(
        detail::RegisteredTool{std::move(info), std::move(fn), check, mode});

    State& st = state();
    std::lock_guard lock(st.write_mutex);
    std::shared_ptr<detail::ToolsSnapshot> current = st.snapshot.load();
    if (!current)
        current = std::make_shared<detail::ToolsSnapshot>();
    if (current->map.contains(tool->info.name))
        return std::unexpected(Error{Errc::Duplicate,
            "tool '" + tool->info.name + "' already registered"});
    // COW：复制旧快照 → 插入 → 原子替换
    std::shared_ptr<detail::ToolsSnapshot> snap = std::make_shared<detail::ToolsSnapshot>();
    snap->map = current->map;
    snap->map.emplace(tool->info.name, std::move(tool));
    st.snapshot.store(std::move(snap));
    return {};
}

Result<void> Tools::reg(ToolInfo info,
    std::function<asio::awaitable<Result<std::string>>(nlohmann::json)> async_fn,
    ArgsCheck check, ToolExecutionMode mode)
{
    // 异步工具 → 同步包装：exec 时 io_context 桥接等协程完成（同步壳包异步核心）
    std::function<Result<std::string>(nlohmann::json)> sync_fn;
    if (async_fn) {
        sync_fn = [async_fn = std::move(async_fn)](nlohmann::json args) -> Result<std::string> {
            asio::io_context io;
            std::optional<Result<std::string>> outcome;
            asio::co_spawn(io, [&]() -> asio::awaitable<void> {
                outcome = co_await async_fn(std::move(args));
            }, asio::detached);
            io.run();
            if (!outcome)
                return std::unexpected(Error{Errc::NetworkError, "async tool: no outcome"});
            return std::move(*outcome);
        };
    }
    // 复用同步注册（校验链一致）
    return reg(std::move(info), std::move(sync_fn), check, mode);
}

Result<std::string> Tools::exec(std::string_view name, nlohmann::json args)
{
    std::shared_ptr<detail::ToolsSnapshot> snap = load_snapshot();
    // 透明哈希异构查找：string_view 直接查，无临时 string 分配。
    // 快照不可变，it->second 在 fn 调用期间稳定（无需出锁复制——根本没锁）。
    auto it = snap->map.find(name);
    if (it == snap->map.end())
        return std::unexpected(Error{Errc::NotFound,
            "tool '" + std::string(name) + "' not found"});
    std::shared_ptr<detail::RegisteredTool const> tool = it->second;

    // ArgsCheck::Tool（反射注册）时跳过：assign_from_json 会做全量校验，
    // 这里再跑一遍 schema 校验是纯冗余
    if (tool->check == ArgsCheck::Schema) {
        std::vector<std::string> errors;
        detail::validate_args(args, tool->info.parameters, "", errors);
        if (!errors.empty()) {
            std::string message;
            for (std::string const& e : errors) {
                if (!message.empty())
                    message += "; ";
                message += e;
            }
            return std::unexpected(Error{Errc::InvalidArgs, std::move(message)});
        }
    }

    // 工具函数遵循零异常约定（错误全部通过 Result 返回），直接调用
    return tool->fn(std::move(args));
}

std::vector<ToolInfo> Tools::list()
{
    std::shared_ptr<detail::ToolsSnapshot> snap = load_snapshot();
    std::vector<ToolInfo> result;
    result.reserve(snap->map.size());
    for (auto const& [name, tool] : snap->map)
        result.push_back(tool->info);
    // 按名排序：输出确定性（不随 rehash 变化），利于 LLM 端 prompt cache
    std::sort(result.begin(), result.end(),
        [](ToolInfo const& a, ToolInfo const& b) { return a.name < b.name; });
    return result;
}

Result<ToolInfo> Tools::get(std::string_view name)
{
    std::shared_ptr<detail::ToolsSnapshot> snap = load_snapshot();
    auto it = snap->map.find(name);
    if (it == snap->map.end())
        return std::unexpected(Error{Errc::NotFound,
            "tool '" + std::string(name) + "' not found"});
    return it->second->info;
}

Result<ToolExecutionMode> Tools::mode(std::string_view name)
{
    std::shared_ptr<detail::ToolsSnapshot> snap = load_snapshot();
    auto it = snap->map.find(name);
    if (it == snap->map.end())
        return std::unexpected(Error{Errc::NotFound,
            "tool '" + std::string(name) + "' not found"});
    return it->second->mode;
}

Result<std::vector<ToolInfo>> Tools::resolve(std::vector<std::string> const& names)
{
    std::shared_ptr<detail::ToolsSnapshot> snap = load_snapshot();
    std::vector<ToolInfo> result;
    result.reserve(names.size());
    for (auto const& name : names) {
        auto it = snap->map.find(name);
        if (it == snap->map.end())
            return std::unexpected(Error{Errc::NotFound,
                "tool '" + name + "' not registered"});
        result.push_back(it->second->info);
    }
    return result;
}

std::vector<std::string> Tools::names()
{
    std::shared_ptr<detail::ToolsSnapshot> snap = load_snapshot();
    std::vector<std::string> result;
    result.reserve(snap->map.size());
    for (auto const& [name, tool] : snap->map)
        result.push_back(name);
    return result;
}

} // namespace agent
