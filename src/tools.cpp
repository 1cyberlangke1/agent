#include <agent/tools.hpp>
#include <mutex>

namespace agent {

Tools::State& Tools::state()
{
    static State s;
    return s;
}

Result<void> Tools::reg(ToolInfo info,
    std::function<Result<std::string>(nlohmann::json)> fn)
{
    auto& st = state();
    std::unique_lock lock(st.mtx);

    // 复制 name，避免评估顺序问题（std::move(info) 可能在 operator[] 之前执行）
    std::string name = info.name;

    if (st.registry.contains(name))
        return std::unexpected(Error{Errc::Duplicate,
            "tool '" + name + "' already registered"});

    st.registry[name] = std::make_shared<detail::RegisteredTool>(
        detail::RegisteredTool{std::move(info), std::move(fn)});
    return {};
}

Result<std::string> Tools::exec(std::string_view name,
    nlohmann::json args)
{
    auto& st = state();
    std::shared_ptr<detail::RegisteredTool> tool;

    {
        std::shared_lock lock(st.mtx);
        auto it = st.registry.find(std::string(name));
        if (it == st.registry.end())
            return std::unexpected(Error{Errc::NotFound,
                "tool '" + std::string(name) + "' not found"});
        tool = it->second;
    }

    // 参数校验
    std::vector<std::string> errors;
    detail::validate_schema(args, tool->info.parameters, "", errors);
    if (!errors.empty()) {
        std::string msg;
        for (auto const& e : errors) {
            if (!msg.empty()) msg += "; ";
            msg += e;
        }
        return std::unexpected(Error{Errc::InvalidArgs, std::move(msg)});
    }

    return tool->fn(std::move(args));
}

std::vector<ToolInfo> Tools::list()
{
    auto& st = state();
    std::shared_lock lock(st.mtx);
    std::vector<ToolInfo> result;
    result.reserve(st.registry.size());
    for (auto const& [_, tool] : st.registry)
        result.push_back(tool->info);
    return result;
}

Result<ToolInfo> Tools::get(std::string_view name)
{
    auto& st = state();
    std::shared_lock lock(st.mtx);
    auto it = st.registry.find(std::string(name));
    if (it == st.registry.end())
        return std::unexpected(Error{Errc::NotFound,
            "tool '" + std::string(name) + "' not found"});
    return it->second->info;
}

} // namespace agent
