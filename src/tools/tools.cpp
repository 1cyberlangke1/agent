#include <algorithm>
#include <mutex>

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

Result<void> Tools::reg(ToolInfo info,
    std::function<Result<std::string>(nlohmann::json)> fn,
    ArgsCheck check)
{
    // 校验链全部在锁外完成，持锁窗口只覆盖查重 + 插入
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
        detail::RegisteredTool{std::move(info), std::move(fn), check});

    State& st = state();
    std::unique_lock lock(st.mtx);
    if (st.registry.contains(tool->info.name))
        return std::unexpected(Error{Errc::Duplicate,
            "tool '" + tool->info.name + "' already registered"});
    st.registry.emplace(tool->info.name, std::move(tool));
    return {};
}

Result<std::string> Tools::exec(std::string_view name, nlohmann::json args)
{
    State& st = state();
    std::shared_ptr<detail::RegisteredTool const> tool;

    {
        std::shared_lock lock(st.mtx);
        // 透明哈希异构查找：string_view 直接查，无临时 string 分配
        auto it = st.registry.find(name);
        if (it == st.registry.end())
            return std::unexpected(Error{Errc::NotFound,
                "tool '" + std::string(name) + "' not found"});
        tool = it->second;
    }

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
    State& st = state();
    std::vector<ToolInfo> result;

    {
        std::shared_lock lock(st.mtx);
        result.reserve(st.registry.size());
        for (auto const& [name, tool] : st.registry)
            result.push_back(tool->info);
    }

    // 锁外按名排序：输出确定性（不随 rehash 变化），利于 LLM 端 prompt cache
    std::sort(result.begin(), result.end(),
        [](ToolInfo const& a, ToolInfo const& b) { return a.name < b.name; });
    return result;
}

Result<ToolInfo> Tools::get(std::string_view name)
{
    State& st = state();
    std::shared_lock lock(st.mtx);
    auto it = st.registry.find(name);
    if (it == st.registry.end())
        return std::unexpected(Error{Errc::NotFound,
            "tool '" + std::string(name) + "' not found"});
    return it->second->info;
}

Result<std::vector<ToolInfo>> Tools::resolve(std::vector<std::string> const& names)
{
    State& st = state();
    std::shared_lock lock(st.mtx);
    std::vector<ToolInfo> result;
    result.reserve(names.size());
    for (auto const& name : names) {
        auto it = st.registry.find(name);
        if (it == st.registry.end())
            return std::unexpected(Error{Errc::NotFound,
                "tool '" + name + "' not registered"});
        result.push_back(it->second->info);
    }
    return result;
}

std::vector<std::string> Tools::names()
{
    State& st = state();
    std::shared_lock lock(st.mtx);
    std::vector<std::string> result;
    result.reserve(st.registry.size());
    for (auto const& [name, tool] : st.registry)
        result.push_back(name);
    return result;
}

} // namespace agent
