// DeepSeek 交互聊天示例：静态反射注册两个工具（add / weather），key 用户输入。
//
// 编译（需 -freflection）：
//   cmake --build --preset default --target deepseek_chat
//   ./build/examples/deepseek_chat.exe [--key <your_api_key>] [--reasoning [level]]
//
// 玩法：
//   - 直接提问：例如「1+1 等于几」
//   - 触发工具：例如「帮我算 123456789 + 987654321」「杭州今天天气怎么样」
//   - 思考：启动时加 --reasoning 开思考（默认 high），或运行时输入
//     `/think`（切换开/关）/ `/think low|medium|high|max`（指定档位）
//   - 输入 exit / quit 退出

#include <agent/llm.hpp>
#include <agent/tools/tools_reflection.hpp>

#include <asio.hpp>

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
using namespace agent::detail;

// ───────────────────── 终端颜色（浅蓝思考 / 绿色工具；管道重定向时去色）─────────────────────

/// stdout 是否为终端：管道/重定向时返回 false，渲染时去掉 ANSI 转义码。
bool stdout_is_terminal()
{
#ifdef _WIN32
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(STDOUT_FILENO) != 0;
#endif
}

/// UTF-8 百分号编码（Open-Meteo geocoding 的 name 参数支持中文城市名）。
std::string url_encode(std::string const& s)
{
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += static_cast<char>(c);
        } else {
            char buf[4];
            std::snprintf(buf, sizeof(buf), "%%%02X", c);
            out += buf;
        }
    }
    return out;
}

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

/// 打印 token 用量（输入 / 输出 / 缓存命中读 / 缓存写入 / 总计）。
void print_usage(Usage const& u)
{
    std::cout << "[用量] 输入 " << u.input_tokens
              << " · 输出 " << u.output_tokens
              << " · 缓存读 " << u.cache_read_tokens
              << " · 缓存写 " << u.cache_write_tokens
              << " · 总计 " << u.total_tokens << std::endl;
}

/// StopReason → 人类可读（与库统一语义）。
std::string stop_reason_str(StopReason reason)
{
    switch (reason) {
        case StopReason::Stop: return "自然停止";
        case StopReason::Length: return "达到长度上限";
        case StopReason::ToolUse: return "工具调用";
        case StopReason::Error: return "错误";
        case StopReason::Aborted: return "已取消";
    }
    return "未知";
}

// ───────────────────── 工具一：add（同步工具，纯计算）─────────────────────
// 同步工具：定义 invoke，ToolBase 同步注册。

struct [[= Desc("计算两个整数的和")]]
add : ToolBase<add>
{
    struct params_type {
        [[= Desc("第一个加数")]] long long a;
        [[= Desc("第二个加数")]] long long b;
    };
    static Result<std::string> invoke(params_type const& p)
    {
        return std::to_string(p.a + p.b);
    }
};
// 显式实例化触发静态注册（必须在本 TU，工具定义不能进 agent_core 库；
// 且必须限定名——using namespace 引入的名字不能用于显式实例化）
template struct agent::ToolBase<add>;

// ───────────────────── 工具二：get_weather（异步工具，真实免费 API，含网络错误处理）─────────────────────
// 异步工具：定义 invoke_async（协程，内部 co_await async_http_get），ToolBase 自动异步注册，
// exec 时库内部桥接。与同步工具 add 并存演示两种风格。

struct [[= Desc("查询城市当前天气（Open-Meteo 免费 API，无 key）")]]
get_weather : ToolBase<get_weather>
{
    struct params_type {
        [[= Desc("城市名，如「北京」「杭州」")]] std::string city;
    };
    static asio::awaitable<Result<std::string>> invoke_async(params_type const& p);
};
template struct agent::ToolBase<get_weather>;

asio::awaitable<Result<std::string>> get_weather::invoke_async(params_type const& p)
{
    // 1. 地理编码：城市名 → 经纬度（异步 GET）
    Result<HttpResponse> geo = co_await agent::async_http_get(
        co_await asio::this_coro::executor,
        "https://geocoding-api.open-meteo.com/v1/search?name=" + url_encode(p.city) + "&count=1",
        HttpRequestOptions{});
    if (!geo)
        co_return std::unexpected(Error{ Errc::NetworkError,
                                         "地理编码请求失败（网络错误？）：" + geo.error().message });
    if (geo->status / 100 != 2)
        co_return std::unexpected(Error{ Errc::NetworkError,
                                         "地理编码 HTTP " + std::to_string(geo->status) });
    nlohmann::json geo_json = nlohmann::json::parse(geo->body, nullptr, false);
    if (geo_json.is_discarded() || !geo_json.contains("results") || geo_json["results"].empty())
        co_return std::unexpected(Error{ Errc::InvalidArgs, "未找到城市：「" + p.city + "」" });
    double lat = geo_json["results"][0].value("latitude", 0.0);
    double lon = geo_json["results"][0].value("longitude", 0.0);

    // 2. 天气查询（异步 GET）
    std::string forecast_url = "https://api.open-meteo.com/v1/forecast?latitude="
        + std::to_string(lat) + "&longitude=" + std::to_string(lon) + "&current_weather=true";
    Result<HttpResponse> forecast = co_await agent::async_http_get(
        co_await asio::this_coro::executor, forecast_url, HttpRequestOptions{});
    if (!forecast)
        co_return std::unexpected(Error{ Errc::NetworkError,
                                         "天气查询失败（网络错误？）：" + forecast.error().message });
    if (forecast->status / 100 != 2)
        co_return std::unexpected(Error{ Errc::NetworkError,
                                         "天气查询 HTTP " + std::to_string(forecast->status) });
    nlohmann::json fc_json = nlohmann::json::parse(forecast->body, nullptr, false);
    if (fc_json.is_discarded() || !fc_json.contains("current_weather")
        || !fc_json["current_weather"].is_object())
        co_return std::unexpected(Error{ Errc::NetworkError, "天气响应解析失败" });

    auto const& cw = fc_json["current_weather"];
    double temp = cw.value("temperature", 0.0);
    int code = cw.value("weathercode", 0);
    double wind = cw.value("windspeed", 0.0);
    co_return "「" + p.city + "」当前天气：" + wmo_description(code)
        + "，" + std::to_string(temp) + "°C，风速 " + std::to_string(wind) + " km/h";
}

// ───────────────────── 工具三：create_event（同步工具，复杂嵌套参数）─────────────────────
// 演示嵌套参数：嵌套 struct（when）+ 数组 of struct（attendees）+ bool 默认值。
// 注意1：嵌套类型定义在工具类外层（类内定义带注解嵌套 struct 会触发 gcc 反射问题）。
// 注意2：工具名不能用 CreateEvent——gcc 反射会把该名字变成 CreateEventA（annotated）
//       导致 CRTP 模板参数不匹配（实测改名即正常，记入 AGENTS.md 已知坑）。
// 注意3：嵌套类型 When/Attendee 不需要类型级注解——build_schema 只读成员注解，
//       类型级 [[=Desc]] 是死代码（冗余）。

struct When
{
    [[= Desc("开始时间 HH:MM")]] std::string start;
    [[= Desc("结束时间 HH:MM")]] std::string end;
    [[= Desc("是否全天")]] bool all_day = false;
};

struct Attendee
{
    [[= Desc("参会人姓名")]] std::string name;
    [[= Desc("是否必须出席")]] bool required = true;
};

struct [[= Desc("创建日程安排（演示复杂嵌套参数：嵌套对象、数组、默认值）")]]
create_event : ToolBase<create_event>
{
    struct params_type {
        [[= Desc("日程标题")]] std::string title;
        [[= Desc("优先级：high/medium/low")]] std::string priority = "medium";
        [[= Desc("时间段")]] When when;
        [[= Desc("参会人列表")]] std::vector<Attendee> attendees;
    };
    static Result<std::string> invoke(params_type const& p);
};
template struct agent::ToolBase<create_event>;

Result<std::string> create_event::invoke(params_type const& p)
{
    std::string out = "已创建日程「" + p.title + "」（优先级 " + p.priority
        + "），时间 " + p.when.start + "–" + p.when.end
        + (p.when.all_day ? "（全天）" : "")
        + "，参会 " + std::to_string(p.attendees.size()) + " 人：";
    for (std::size_t i = 0; i < p.attendees.size(); ++i) {
        if (i) out += "、";
        out += p.attendees[i].name + (p.attendees[i].required ? "" : "（可选）");
    }
    return out;
}

// ───────────────────── key / 思考 解析（--key / --reasoning）─────────────────────

struct ChatConfig {
    std::string api_key;
    bool thinking = false;                    // 思考开关（/think 可切换）
    ThinkingLevel thinking_level = ThinkingLevel::High;
};

ChatConfig parse_config(int argc, char* argv[])
{
    ChatConfig config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--key" && i + 1 < argc) {
            config.api_key = argv[++i];
        } else if (arg == "--reasoning") {
            config.thinking = true;
            // 可选：--reasoning high / --reasoning low 等
            if (i + 1 < argc) {
                std::string level = argv[i + 1];
                if (level == "low") { config.thinking_level = ThinkingLevel::Low; ++i; }
                else if (level == "medium") { config.thinking_level = ThinkingLevel::Medium; ++i; }
                else if (level == "high") { config.thinking_level = ThinkingLevel::High; ++i; }
                else if (level == "max") { config.thinking_level = ThinkingLevel::Max; ++i; }
            }
        }
    }
    if (config.api_key.empty()) {
        std::cout << "请输入 DeepSeek API key（或重新运行加 --key <key>）：";
        std::getline(std::cin, config.api_key);
    }
    return config;
}

/// 解析交互命令（/think 等）。返回 true 表示输入被命令消费（不是普通提问）。
bool handle_command(std::string const& input, ChatConfig& config)
{
    if (input == "/think") {
        config.thinking = !config.thinking;
        std::cout << "思考已" << (config.thinking ? "开启" : "关闭") << "\n";
        return true;
    }
    if (input.rfind("/think ", 0) == 0) {
        std::string level = input.substr(7);
        if (level == "low") config.thinking_level = ThinkingLevel::Low;
        else if (level == "medium") config.thinking_level = ThinkingLevel::Medium;
        else if (level == "high") config.thinking_level = ThinkingLevel::High;
        else if (level == "max") config.thinking_level = ThinkingLevel::Max;
        else { std::cout << "档位需为 low/medium/high/max\n"; return true; }
        config.thinking = true;
        std::cout << "思考已开启（" << level << "）\n";
        return true;
    }
    if (input == "/help") {
        std::cout << "命令：/think（切换思考开/关）、/think <low|medium|high|max>、"
                     "/schema <工具名>（打印工具 JSON schema）、exit/quit（退出）\n";
        return true;
    }
    if (input.rfind("/schema ", 0) == 0) {
        std::string name = input.substr(8);
        Result<ToolInfo> tool = Tools::get(name);
        if (!tool) {
            std::cout << "未找到工具：" << name << "\n";
        } else {
            std::cout << tool->parameters.dump(2) << std::endl;
        }
        return true;
    }
    return false;
}

// ───────────────────── 交互聊天循环 ─────────────────────

int main(int argc, char* argv[])
{
    ChatConfig config = parse_config(argc, argv);
    if (config.api_key.empty()) {
        std::cerr << "未提供 API key\n";
        return 1;
    }

    // deepseek-v4-flash：reasoning 模型（effort 型 thinking_level_map），工具调用用
    ModelView model = *ModelRegistry::find_model("deepseek-v4-flash");
    DeepSeekProvider deepseek({ .name = "deepseek",
                                .api_key = config.api_key,
                                .base_url = "https://api.deepseek.com" });

    std::vector<Message> messages;
    std::cout << "已连接 DeepSeek（deepseek-v4-flash）。输入提问（/help 看命令）。\n";
    std::cout << "工具：add（求和）、get_weather（查天气，Open-Meteo 免费 API）、"
                 "create_event（创建日程，演示嵌套参数）。"
                 "思考：/think 切换，当前"
              << (config.thinking ? "开" : "关") << "\n\n";

    while (true) {
        std::string input;
        std::cout << "> ";
        std::getline(std::cin, input);
        if (std::cin.eof() || input == "exit" || input == "quit")
            break;
        if (input.empty())
            continue;
        if (handle_command(input, config))
            continue;

        messages.push_back(Message{ Role::User, { Text{ input } } });

        // 内层循环：模型可能连续调用多个工具（每轮 exec 后回传再问）
        while (true) {
            Context ctx;
            ctx.messages = messages;
            ctx.tools = Tools::names();   // Add + Weather 已自动注册（工具名 → 引擎按名查注册表）

            StreamOptions opts;
            if (config.thinking)
                opts.reasoning = config.thinking_level;   // 统一 ThinkingLevel → 引擎映射

            std::string text;
            std::vector<ToolCall> tool_calls;
            Usage final_usage;
            StopReason final_stop = StopReason::Stop;
            bool const color = stdout_is_terminal();
            std::string const kBlue = "\033[94m";    // 浅蓝：思考链
            std::string const kGreen = "\033[32m";   // 绿：工具调用
            std::string const kReset = "\033[0m";
            bool thinking_open = false;
            auto close_thinking = [&]() {
                if (thinking_open) {
                    std::cout << (color ? kReset : "") << std::endl;
                    thinking_open = false;
                }
            };
            for (StreamEvent const& ev : deepseek.stream(model, ctx, opts)) {
                switch (ev.type()) {
                    case StreamEvent::Type::TextDelta:   // 正文：打字机流式
                        close_thinking();
                        text += std::get<TextDelta>(ev.data).text;
                        std::cout << std::get<TextDelta>(ev.data).text << std::flush;
                        break;
                    case StreamEvent::Type::ThinkingDelta:   // 思考：浅蓝一段流式
                        if (!thinking_open) {
                            std::cout << "\n" << (color ? kBlue : "") << "[思考] ";
                            thinking_open = true;
                        }
                        std::cout << std::get<ThinkingDelta>(ev.data).text << std::flush;
                        break;
                    case StreamEvent::Type::ToolCallDelta:   // 参数增量太碎，不显示
                        close_thinking();
                        break;
                    case StreamEvent::Type::ToolCallEnd: {   // 工具调用：绿色一行
                        close_thinking();
                        auto const& end = std::get<ToolCallEnd>(ev.data);
                        tool_calls.push_back(ToolCall{ end.id, end.name, end.arguments });
                        std::string line = "\n→ 调用工具 " + end.name + "(" + end.arguments.dump() + ")";
                        std::cout << (color ? kGreen + line + kReset : line) << std::endl;
                        break;
                    }
                    case StreamEvent::Type::Usage:   // token 用量（流中）
                        close_thinking();
                        final_usage = std::get<UsageEvent>(ev.data).usage;
                        std::cout << "\n";
                        print_usage(final_usage);
                        break;
                    case StreamEvent::Type::Done:   // 流结束汇总
                        close_thinking();
                        final_usage = std::get<DoneEvent>(ev.data).response.usage;
                        final_stop = std::get<DoneEvent>(ev.data).response.stop_reason;
                        break;
                    case StreamEvent::Type::Error:   // 错误
                        close_thinking();
                        std::cout << "\n[错误] " << std::get<Error>(ev.data).message << std::endl;
                        break;
                }
            }
            close_thinking();
            if (!text.empty())
                std::cout << std::endl;
            // 最终汇总：停止原因 + 全部用量（含缓存命中）合并一行
            std::cout << "[完成] " << stop_reason_str(final_stop);
            if (final_usage.total_tokens > 0) {
                std::cout << " · 输入 " << final_usage.input_tokens
                          << " · 输出 " << final_usage.output_tokens
                          << " · 缓存读 " << final_usage.cache_read_tokens
                          << " · 缓存写 " << final_usage.cache_write_tokens
                          << " · 总计 " << final_usage.total_tokens;
            }
            std::cout << std::endl;

            if (tool_calls.empty())
                break;   // 无工具调用 → 本轮对话结束

            // 执行工具并把结果回传
            std::vector<ContentBlock> assistant_blocks;
            for (ToolCall const& tc : tool_calls)
                assistant_blocks.push_back(tc);
            messages.push_back(Message{ Role::Assistant, std::move(assistant_blocks) });

            for (ToolCall const& tc : tool_calls) {
                std::cout << "  [调用工具 " << tc.name << " " << tc.arguments.dump() << "]\n";
                Result<std::string> output = Tools::exec(tc.name, tc.arguments);
                bool failed = !output.has_value();
                std::string out_text = failed ? output.error().message : *output;
                messages.push_back(Message{ Role::ToolResult,
                                            { ToolResult{ tc.id, out_text, failed } } });
            }
        }
    }
    return 0;
}
