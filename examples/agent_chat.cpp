// 用高层 Agent 类（Agent<Provider, Behaviors>）的简单交互示例。
//
// 演示三层内容：
//   1. Agent 类：run() 事件流 + 自动工具往返 + run 内防爆压缩 + 手动压缩（/compact）
//   2. 编译时工具：weather_now（ToolBase 反射注册，同步真实 API，Open-Meteo 无 key）
//   3. 运行时工具 ×2：send_report / plan_trip（Tools::reg 注册，复杂嵌套参数
//      ——嵌套对象 / 数组 of 对象 / 枚举 / 默认值 / 数值范围）
//
// 编译（需 -freflection）：
//   cmake --build --preset default --target agent_chat
//   ./build/examples/agent_chat.exe --key <your_api_key>
//
// 玩法：
//   - 查天气：例如「北京今天天气怎么样」「杭州多少度」
//   - 复杂参数工具：例如「给张三(a@x.com)发一份 report-001，抄送李四，高优先级」
//                   「规划去成都玩 5 天，高铁北京→成都，再大巴去峨眉，住 4 星酒店」
//   - /compact：手动压缩上下文；exit / quit：退出
//
// ⚠️ 正确的事件路径：正文增量在 AgentEvent::MessageUpdate.delta 里（TextDelta/ThinkingDelta），
//    不是独立事件（PLAN_3.md 开篇示例的 AgentEvent::Type::TextDelta 是图纸笔误）。

#include <agent/agent.hpp>
#include <agent/llm.hpp>
#include <agent/tools/tools_reflection.hpp>

#include <asio.hpp>

#include <cctype>
#include <cstdio>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace agent;

// ───────────────────── 终端颜色（管道重定向时去色）─────────────────────

bool stdout_is_terminal()
{
#ifdef _WIN32
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(STDOUT_FILENO) != 0;
#endif
}

// ───────────────────── 编译时工具：weather_now（反射注册，同步真实 API）─────────────────────

/// WMO weathercode → 中文描述（Open-Meteo current_weather.weathercode）。
std::string wmo_description(int code)
{
    switch (code) {
        case 0: return "晴";
        case 1: case 2: case 3: return "多云";
        case 45: case 48: return "有雾";
        case 51: case 53: case 55: case 56: case 57: return "毛毛雨";
        case 61: case 63: case 65: case 66: case 67: return "雨";
        case 71: case 73: case 75: case 77: return "雪";
        case 80: case 81: case 82: return "阵雨";
        case 85: case 86: return "阵雪";
        case 95: return "雷暴";
        case 96: case 99: return "雷暴伴冰雹";
        default: return "未知";
    }
}

class [[= Desc("查询城市当前天气（Open-Meteo 免费 API，无 key）")]]
weather_now : public ToolBase<weather_now>
{
public:
    struct params_type {
        [[= Desc("城市名，如「北京」「杭州」")]] std::string city;
        [[= Desc("温度单位：celsius/fahrenheit")]] std::string unit = "celsius";
    };
    static Result<std::string> invoke(params_type const& p);
};
// 显式实例化触发静态注册（必须在本 TU；且必须限定名）。
template struct agent::ToolBase<weather_now>;

Result<std::string> weather_now::invoke(params_type const& p)
{
    // 便捷层 http_get_json：传输/状态码/JSON 解析三层错误合一，一次 check 搞定
    bool fahrenheit = (p.unit == "fahrenheit");
    Result<nlohmann::json> geo = http_get_json(
        "https://geocoding-api.open-meteo.com/v1/search?name=" + url_encode(p.city) + "&count=1", {});
    if (!geo)
        return std::unexpected(geo.error());
    if (!geo->contains("results") || (*geo)["results"].empty())
        return std::unexpected(Error{ Errc::InvalidArgs, "未找到城市：「" + p.city + "」" });
    double lat = (*geo)["results"][0].value("latitude", 0.0);
    double lon = (*geo)["results"][0].value("longitude", 0.0);

    Result<nlohmann::json> forecast = http_get_json(
        "https://api.open-meteo.com/v1/forecast?latitude=" + std::to_string(lat)
        + "&longitude=" + std::to_string(lon) + "&current_weather=true"
        + (fahrenheit ? "&temperature_unit=fahrenheit" : ""), {});
    if (!forecast)
        return std::unexpected(forecast.error());
    if (!forecast->contains("current_weather") || !(*forecast)["current_weather"].is_object())
        return std::unexpected(Error{ Errc::NetworkError, "天气响应解析失败" });

    auto const& cw = (*forecast)["current_weather"];
    return "「" + p.city + "」当前天气：" + wmo_description(cw.value("weathercode", 0))
        + "，" + std::to_string(cw.value("temperature", 0.0)) + (fahrenheit ? "°F" : "°C")
        + "，风速 " + std::to_string(cw.value("windspeed", 0.0)) + " km/h";
}

// ───────────────────── 运行时工具：send_report（嵌套对象 + 数组 + 枚举 + 默认值）─────────────────────

/// 运行时注册：Tools::reg(ToolInfo{name, description, JSON Schema}, fn)。
/// Schema 全手写嵌套；ArgsCheck::Schema（默认）让 exec 在进 fn 前按 schema 校验
/// 类型/required/枚举/数值范围——fn 只负责读嵌套字段拼结果。
Result<void> register_send_report()
{
    return Tools::reg(
        ToolInfo{
            "send_report",
            "发送报告给一组人（演示运行时注册：嵌套对象 + 数组 of 对象 + 枚举 + 默认值）",
            {
                { "type", "object" },
                { "properties", {
                    { "report_id", {{ "type", "string" }, { "description", "报告 ID" }} },
                    { "recipient", {{ "type", "object" },
                        { "properties", {
                            { "name",  {{ "type", "string" }, { "description", "收件人姓名" }} },
                            { "email", {{ "type", "string" }, { "description", "收件人邮箱" }} },
                        }},
                        { "required", { "name", "email" } }} },
                    { "cc", {{ "type", "array" }, { "description", "抄送列表" },
                        { "items", {{ "type", "object" },
                            { "properties", {
                                { "name",  {{ "type", "string" }} },
                                { "email", {{ "type", "string" }} },
                            }},
                            { "required", { "email" } }} }} },
                    { "urgency", {{ "type", "string" },
                        { "enum", { "low", "normal", "high" } }, { "default", "normal" } } },
                    { "attach_snapshot", {{ "type", "boolean" }, { "default", true } } },
                }},
                { "required", { "report_id", "recipient" } },
            }
        },
        [](nlohmann::json args) -> Result<std::string> {
            std::string out = "已发送报告「" + args.value("report_id", "?")
                + "」给 " + args["recipient"].value("name", "?")
                + " <" + args["recipient"].value("email", "?") + ">"
                + "（优先级 " + args.value("urgency", "normal") + "）";
            if (args.contains("cc") && args["cc"].is_array()) {
                out += "，抄送 ";
                bool first = true;
                for (auto const& cc : args["cc"]) {
                    if (!first) out += "、";
                    first = false;
                    out += cc.value("name", cc.value("email", "?"));
                }
            }
            out += args.value("attach_snapshot", true) ? "，附快照" : "，不附快照";
            return out;
        });
}

// ───────────────────── 运行时工具：plan_trip（数组 of 嵌套对象 + 两级枚举 + 数值范围）─────────────────────

Result<void> register_plan_trip()
{
    return Tools::reg(
        ToolInfo{
            "plan_trip",
            "规划多段行程（演示运行时注册：数组 of 嵌套对象 + 两级枚举 + 数值范围）",
            {
                { "type", "object" },
                { "properties", {
                    { "destination", {{ "type", "string" }, { "description", "目的地" }} },
                    { "days", {{ "type", "integer" }, { "minimum", 1 }, { "maximum", 30 },
                                { "description", "行程天数" }} },
                    { "legs", {{ "type", "array" }, { "description", "行程段" },
                        { "items", {{ "type", "object" },
                            { "properties", {
                                { "from", {{ "type", "string" }} },
                                { "to",   {{ "type", "string" }} },
                                { "mode", {{ "type", "string" },
                                    { "enum", { "train", "flight", "bus", "car" } },
                                    { "default", "train" } } },
                            }},
                            { "required", { "from", "to" } }} }} },
                    { "hotel", {{ "type", "object" },
                        { "properties", {
                            { "name", {{ "type", "string" }, { "description", "酒店名" }} },
                            { "stars", {{ "type", "integer" }, { "minimum", 1 }, { "maximum", 5 },
                                         { "description", "星级 1-5" } } },
                        }},
                        { "required", { "stars" } }} },
                }},
                { "required", { "destination", "days" } },
            }
        },
        [](nlohmann::json args) -> Result<std::string> {
            std::string out = "规划「" + args.value("destination", "?")
                + "」" + std::to_string(args.value("days", 0)) + " 天行程";
            if (args.contains("legs") && args["legs"].is_array()) {
                out += "：";
                bool first = true;
                for (auto const& leg : args["legs"]) {
                    if (!first) out += " → ";
                    first = false;
                    out += leg.value("from", "?") + "→" + leg.value("to", "?")
                        + "（" + leg.value("mode", "train") + "）";
                }
            }
            if (args.contains("hotel") && args["hotel"].is_object())
                out += "，住 " + args["hotel"].value("name", "?")
                    + "（" + std::to_string(args["hotel"].value("stars", 0)) + " 星）";
            return out;
        });
}

// ───────────────────── 入口：Agent 交互循环 ─────────────────────

int main(int argc, char* argv[])
{
    std::string api_key;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argc > i ? argv[i] : "") == "--key" && i + 1 < argc)
            api_key = argv[++i];
    }
    if (api_key.empty()) {
        std::cout << "请输入 DeepSeek API key（或运行加 --key <key>）：";
        std::getline(std::cin, api_key);
    }
    if (api_key.empty()) {
        std::cerr << "未提供 API key\n";
        return 1;
    }

    std::optional<ModelView> model_opt = ModelRegistry::find_model("deepseek-v4-flash");
    if (!model_opt) {
        std::cerr << "未找到模型 deepseek-v4-flash\n";
        return 1;
    }

    // 先注册运行时工具（编译时 weather_now 静态注册），再构造 Agent。
    // Agent 默认不开放任何工具——须 set_tools 显式指定允许集（广告 + 执行门控双重生效）。
    Result<void> reg_report = register_send_report();
    Result<void> reg_trip = register_plan_trip();
    if (!reg_report)
        std::cerr << "[警告] send_report 注册失败: " << reg_report.error().message << "\n";
    if (!reg_trip)
        std::cerr << "[警告] plan_trip 注册失败: " << reg_trip.error().message << "\n";

    Agent<DeepSeekProvider> agent(
        { .name = "deepseek", .api_key = api_key, .base_url = "https://api.deepseek.com" },
        *model_opt,
        {},
        "你是一个会使用工具的助手。请根据用户请求调用合适的工具满足需求，用中文回答。");
    agent.set_tools({ "weather_now", "send_report", "plan_trip" });   // 手动指定允许的工具

    std::cout << "已连接 DeepSeek（deepseek-v4-flash），Agent 自动工具往返。\n";
    std::cout << "已开放工具：weather_now（查天气·真实 API）、send_report（发报告·嵌套参数）、"
                 "plan_trip（规划行程·嵌套参数）。\n";
    std::cout << "命令：/compact（手动压缩）、/help、exit（退出）\n\n";

    bool const color = stdout_is_terminal();
    std::string const kBlue = "\033[94m";
    std::string const kGreen = "\033[32m";
    std::string const kRed = "\033[31m";
    std::string const kReset = "\033[0m";

    while (true) {
        std::string line;
        std::cout << "> ";
        std::getline(std::cin, line);
        if (std::cin.eof() || line == "exit" || line == "quit")
            break;
        if (line.empty())
            continue;
        if (line == "/compact") {
            Result<bool> r = agent.compact();
            if (!r)
                std::cout << "[压缩失败] " << r.error().message << "\n";
            else if (*r)
                std::cout << "[已压缩] 上下文已替换为摘要 + 保留段\n";
            else
                std::cout << "[未触发] 上下文未超限\n";
            continue;
        }
        if (line == "/help") {
            std::cout << "命令：/compact（手动压缩）、/help、exit/quit（退出）\n";
            continue;
        }

        StreamOptions opts;
        bool thinking_open = false;
        auto close_thinking = [&]() {
            if (thinking_open) {
                std::cout << (color ? kReset : "") << std::endl;
                thinking_open = false;
            }
        };

        // Agent.run：自动工具往返 / 内防爆压缩都内置，这里只消费事件渲染
        for (AgentEvent const& ev : agent.run({ Message{ Role::User, { Text{ line } } } }, opts)) {
            switch (ev.type()) {
                case AgentEvent::Type::MessageUpdate: {
                    auto const& update = std::get<MessageUpdate>(ev.data);
                    if (auto* td = std::get_if<TextDelta>(&update.delta)) {
                        close_thinking();
                        std::cout << td->text << std::flush;   // 正文打字机
                    } else if (auto* th = std::get_if<ThinkingDelta>(&update.delta)) {
                        if (!thinking_open) {
                            std::cout << "\n" << (color ? kBlue : "") << "[思考] ";
                            thinking_open = true;
                        }
                        std::cout << th->text << std::flush;
                    }
                    break;
                }
                case AgentEvent::Type::ToolCallEnd: {
                    close_thinking();
                    auto const& call = std::get<ToolCallEnd>(ev.data);
                    std::string line2 = "\n→ 调用工具 " + call.name + "(" + call.arguments.dump() + ")";
                    std::cout << (color ? kGreen + line2 + kReset : line2) << std::endl;
                    break;
                }
                case AgentEvent::Type::ToolExecEnd: {
                    close_thinking();
                    auto const& end = std::get<ToolExecEnd>(ev.data);
                    if (end.result.is_error)
                        std::cout << (color ? kRed + "  [工具失败] " + kReset : "  [工具失败] ")
                                  << end.result.output << "\n";
                    else
                        std::cout << "  [结果] " << end.result.output << "\n";
                    break;
                }
                case AgentEvent::Type::AgentError: {
                    close_thinking();
                    std::cout << (color ? kRed : "") << "\n[错误] "
                              << std::get<AgentError>(ev.data).error.message
                              << (color ? kReset : "") << std::endl;
                    break;
                }
                case AgentEvent::Type::AgentEnd: {
                    close_thinking();
                    std::cout << "\n[完成] 对话消息 " << agent.messages().size() << " 条";
                    if (auto usage = agent.last_usage())
                        std::cout << " · 输入 " << usage->input_tokens
                                  << " · 输出 " << usage->output_tokens;
                    std::cout << " · 累计花费 $" << agent.context_cost() << std::endl;
                    break;
                }
                default:
                    break;   // AgentStart/TurnStart/MessageStart/ToolCallDelta 等不渲染
            }
        }
        close_thinking();
    }
    return 0;
}
