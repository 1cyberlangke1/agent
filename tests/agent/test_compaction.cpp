// 压缩纯函数 T0 测试：估算 / 触发决策 / 切点 / 摘要请求构造 / 校验 / 组装。

#include <agent/agent/compaction.hpp>

#include <doctest/doctest.h>

#include <string>

using namespace agent;

namespace {

Message user(std::string text)
{
    return Message{ Role::User, { Text{ std::move(text) } } };
}
Message assistant(std::string text)
{
    return Message{ Role::Assistant, { Text{ std::move(text) } } };
}
Message assistant_with_usage(std::string text, size_t input, size_t output)
{
    Message m = assistant(std::move(text));
    m.set_input_tokens(input);
    m.set_output_tokens(output);
    return m;
}

std::string message_text(Message const& message)
{
    for (auto const& block : message.content)
        if (auto text = std::get_if<Text>(&block))
            return text->text;
    return {};
}

}  // namespace

TEST_CASE("estimate_tokens_default：字符数/4，image 4800")
{
    CHECK(estimate_tokens_default(user("")) == 0);
    CHECK(estimate_tokens_default(user("aaaa")) == 1);           // 4 字符 → 1 token
    CHECK(estimate_tokens_default(user(std::string(40, 'x'))) == 10);
    Message image_message{ Role::User, { Image{ "AAAA", "image/png" } } };
    CHECK(estimate_tokens_default(image_message) == 4800);
    Message tool_message{ Role::Assistant, { ToolCall{ "c", "get_weather", nlohmann::json{{"location", "杭州"}} } } };
    // name(11)/4 + args dump(约 18)/4
    CHECK(estimate_tokens_default(tool_message) > 0);
    Message result_message{ Role::ToolResult, { ToolResult{ "c", std::string(20, 'r'), false } } };
    CHECK(estimate_tokens_default(result_message) == 7);   // (20+8)/4
}

TEST_CASE("estimate_context_tokens：无锚点全量估算")
{
    std::vector<Message> messages{ user(std::string(16, 'a')), assistant(std::string(16, 'b')) };
    CHECK(estimate_context_tokens(messages) == 8);   // (16+16)/4
}

TEST_CASE("estimate_context_tokens：usage 锚点真实优先")
{
    // 中间一条 assistant 带真实 token：锚点 = 该请求累计，其后消息逐条估算
    std::vector<Message> messages{
        user(std::string(400, 'a')),                       // 估算 100（锚点已含，忽略）
        assistant_with_usage(std::string(20, 'b'), 500, 30),  // 锚点 530
        user(std::string(16, 'c')),                        // 锚点后估算 4
    };
    CHECK(estimate_context_tokens(messages) == 534);
}

TEST_CASE("estimate_context_tokens：锚点取最近一条")
{
    std::vector<Message> messages{
        assistant_with_usage("x", 100, 10),
        assistant_with_usage("y", 200, 20),   // 最近锚点 220
        user(std::string(16, 'c')),           // 其后估算 4
    };
    CHECK(estimate_context_tokens(messages) == 224);
}

TEST_CASE("should_compact：边界")
{
    CompactionSettings settings{ .reserve_tokens = 100, .keep_recent_tokens = 50, .tail_turns = 2 };
    // context_window=1000 → 阈值 900
    CHECK(!should_compact(900, 1000, settings));
    CHECK(should_compact(901, 1000, settings));
}

TEST_CASE("find_cut_point：空 / 无 user 消息 → 0")
{
    CompactionSettings settings{ 100, 50, 2 };
    CHECK(find_cut_point({}, settings) == 0);
    CHECK(find_cut_point({ assistant("aa"), assistant("bb") }, settings) == 0);
}

TEST_CASE("find_cut_point：全部在保留预算内 → 0")
{
    // 2 轮（user16+asst16=8 token/轮），keep_recent=1000
    CompactionSettings settings{ 100, 1000, 2 };
    std::vector<Message> messages{
        user(std::string(16, 'a')), assistant(std::string(16, 'b')),
        user(std::string(16, 'c')), assistant(std::string(16, 'd')),
    };
    CHECK(find_cut_point(messages, settings) == 0);
}

TEST_CASE("find_cut_point：tail_turns 封顶")
{
    // 4 轮 × 8 token，keep_recent=1000 → 只保留最后 tail_turns=2 轮
    CompactionSettings settings{ 100, 1000, 2 };
    std::vector<Message> messages{
        user(std::string(16, 'a')), assistant(std::string(16, 'b')),
        user(std::string(16, 'c')), assistant(std::string(16, 'd')),
        user(std::string(16, 'e')), assistant(std::string(16, 'f')),
        user(std::string(16, 'g')), assistant(std::string(16, 'h')),
    };
    // turn 起点 index：0, 2, 4, 6；最后 2 轮 = 起点 4
    CHECK(find_cut_point(messages, settings) == 4);
}

TEST_CASE("find_cut_point：轮超预算 → 轮内切（保留后半段）")
{
    // 每轮 user200字符(50token)+asst200字符(50token)=100；keep_recent=60
    CompactionSettings settings{ 100, 60, 2 };
    std::vector<Message> messages{
        user(std::string(200, 'a')), assistant(std::string(200, 'b')),
        user(std::string(200, 'c')), assistant(std::string(200, 'd')),
    };
    // 最后一轮(100)超预算 → 轮内切：asst(50)<=60 保留，user(50)>10 整条保留
    // cut = user2 起点 = 2
    CHECK(find_cut_point(messages, settings) == 2);
}

TEST_CASE("is_compaction_summary / extract_summary_text")
{
    Message summary{ Role::User, { Text{ std::string(kCompactionSummaryPrefix) + "用户问了天气" } } };
    CHECK(is_compaction_summary(summary));
    CHECK(extract_summary_text(summary) == "用户问了天气");

    CHECK(!is_compaction_summary(user("正常消息")));
    CHECK(extract_summary_text(user("正常消息")).empty());
    // 前缀但角色不是 user → 不算摘要
    Message asst_summary{ Role::Assistant, { Text{ std::string(kCompactionSummaryPrefix) + "x" } } };
    CHECK(!is_compaction_summary(asst_summary));
}

TEST_CASE("build_summary_instruction：含 previous-summary 增量段")
{
    std::string instruction = build_summary_instruction("");
    CHECK(instruction.find("Please compress the conversation above") != std::string::npos);
    CHECK(instruction.find("previous summary") == std::string::npos);

    std::string with_prev = build_summary_instruction("old summary");
    CHECK(with_prev.find("previous summary") != std::string::npos);
    CHECK(with_prev.find("old summary") != std::string::npos);
}

TEST_CASE("build_summary_request：未触发 → nullopt")
{
    CompactionSettings settings{ 100, 60, 2 };
    ModelView model = [] {
        ModelView mv;
        static const std::string id = "m";
        mv.id = id;
        mv.context_window = 100000;   // 大窗口 → 不触发
        mv.max_output_tokens = 4096;
        return mv;
    }();
    std::vector<Message> messages{ user("hi") };
    CHECK(!build_summary_request(messages, model, settings, "sys", "").has_value());
}

TEST_CASE("build_summary_request：触发 → 摘要请求构造正确")
{
    CompactionSettings settings{ 100, 60, 2 };
    ModelView model = [] {
        ModelView mv;
        static const std::string id = "m";
        mv.id = id;
        mv.context_window = 200;   // 阈值 200-100=100；上下文 200 > 100 触发
        mv.max_output_tokens = 4096;
        return mv;
    }();
    std::vector<Message> messages{
        user(std::string(200, 'a')), assistant(std::string(200, 'b')),
        user(std::string(200, 'c')), assistant(std::string(200, 'd')),
    };
    std::optional<SummaryRequest> request = build_summary_request(messages, model, settings, "sys", "");
    REQUIRE(request.has_value());
    CHECK(request->cut == 2);
    // 摘要段 = [0,2) + 尾部 user 指令；tools 留空；system 不变
    CHECK(request->ctx.messages.size() == 3);
    CHECK(request->ctx.messages[2].role == Role::User);
    CHECK(message_text(request->ctx.messages[2]).find("Please compress the conversation above") != std::string::npos);
    CHECK(request->ctx.tools.empty());
    CHECK(request->ctx.system_prompt == "sys");
    // 摘要请求采样：max_tokens = min(4096,4096)、thinking 关、温度 0
    REQUIRE(request->opts.max_tokens.has_value());
    CHECK(*request->opts.max_tokens == 4096);
    REQUIRE(request->opts.reasoning.has_value());
    CHECK(*request->opts.reasoning == ThinkingLevel::Off);
    REQUIRE(request->opts.temperature.has_value());
    CHECK(*request->opts.temperature == 0);
}

TEST_CASE("validate_summary_response：ToolCall / 空摘要 → CompactionFailed")
{
    ChatResponse tool_response;
    tool_response.content.push_back(ToolCall{ "c", "get_weather", nlohmann::json::object() });
    Result<std::string> r1 = validate_summary_response(tool_response);
    REQUIRE(!r1);
    CHECK(r1.error().code == Errc::CompactionFailed);
    CHECK(r1.error().message.find("工具调用") != std::string::npos);

    ChatResponse empty_response;
    Result<std::string> r2 = validate_summary_response(empty_response);
    REQUIRE(!r2);
    CHECK(r2.error().code == Errc::CompactionFailed);

    ChatResponse ok_response;
    ok_response.content.push_back(Text{ "摘要内容" });
    Result<std::string> r3 = validate_summary_response(ok_response);
    REQUIRE(r3);
    CHECK(*r3 == "摘要内容");
}

TEST_CASE("apply_summary：新对话 = [摘要] + 保留段")
{
    std::vector<Message> messages{
        user(std::string(200, 'a')), assistant(std::string(200, 'b')),
        user(std::string(200, 'c')), assistant(std::string(200, 'd')),
    };
    std::vector<Message> result = apply_summary(messages, 2, "摘要文本");
    REQUIRE(result.size() == 3);
    CHECK(is_compaction_summary(result[0]));
    CHECK(extract_summary_text(result[0]) == "摘要文本");
    CHECK(result[1].role == Role::User);
    CHECK(message_text(result[1]) == std::string(200, 'c'));
    CHECK(result[2].role == Role::Assistant);
}
