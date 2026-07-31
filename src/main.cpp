#include <cstdio>

#include <agent/tools_reflection.hpp>

using agent::ArgsCheck;
using agent::Desc;
using agent::Result;
using agent::ToolBase;
using agent::ToolInfo;
using agent::Tools;

// ——— 编译期反射注册：名称/描述/schema 全部自动生成 ——— //

struct GreetInput
{
    [[= Desc("用户名")]] std::string name;
    [[= Desc("语言")]]   std::string lang = "zh";
};

struct [[= Desc("打招呼")]]
Greet : ToolBase<Greet>
{
    using params_type = GreetInput;
    static Result<std::string> invoke(GreetInput const& p)
    {
        return "你好 " + p.name;
    }
};
template struct agent::ToolBase<Greet>;

// ——— 运行时动态注册：schema 手写，exec 按 schema 预校验 ——— //

struct RuntimeReg
{
    RuntimeReg()
    {
        nlohmann::json schema = {
            {"type", "object"},
            {"properties", {{"format", {{"type", "string"}}}}}
        };
        Result<void> registered = Tools::reg(
            ToolInfo{ "now", "当前时间", std::move(schema) },
            [](nlohmann::json) -> Result<std::string> { return "12:00"; }
        );
        if (!registered)
            std::fprintf(stderr, "register 'now' failed: %s\n",
                registered.error().message.c_str());
    }
} runtime_reg;

// ——— 主函数：列出工具并各执行一次 ——— //

int main()
{
    std::vector<ToolInfo> tools = Tools::list();
    std::printf("registered %zu tools:\n", tools.size());
    for (ToolInfo const& t : tools)
        std::printf("  [%s] %s\n", t.name.c_str(), t.description.c_str());

    Result<std::string> greet = Tools::exec("Greet", {{"name", "杭州"}});
    if (greet)
        std::printf("Greet: %s\n", greet->c_str());
    else
        std::printf("Greet failed: %s\n", greet.error().message.c_str());

    Result<std::string> now = Tools::exec("now", nlohmann::json::object());
    if (now)
        std::printf("now: %s\n", now->c_str());
    else
        std::printf("now failed: %s\n", now.error().message.c_str());

    std::printf("ok\n");
    return 0;
}
