/// @brief Content 类型层单元测试
///
/// 覆盖：
/// - Text / Image / Thinking / ToolCall 构造与字段
/// - ContentBlock variant + std::visit
/// - Role 枚举值
/// - Message：构造、拷贝、迭代器、Extra 存取、列表操作
/// - Content：构造、索引、迭代器、增删改查
/// - Extra 深拷贝独立性
///
/// @input  直接构造的 Content / Message 对象
/// @output CHECK 断言
/// @behavior 所有断言通过则测试通过

#include <doctest/doctest.h>
#include <agent/llm/content.hpp>

using namespace agent;

// ═══════════════════════════════════════════════════
// Text
// ═══════════════════════════════════════════════════

/// @brief Text 通过构造函数设置文本内容
TEST_CASE("Text construction") {
    Text t{"hello"};
    CHECK(t.text == "hello");
}

/// @brief Text 默认构造为字符串
TEST_CASE("Text default") {
    Text t{};
    CHECK(t.text.empty());
}

// ═══════════════════════════════════════════════════
// Image
// ═══════════════════════════════════════════════════

/// @brief Image 同时存储 base64 数据和 MIME 类型
TEST_CASE("Image construction") {
    Image img{"base64data", "image/png"};
    CHECK(img.data == "base64data");
    CHECK(img.mime_type == "image/png");
}

// ═══════════════════════════════════════════════════
// Thinking
// ═══════════════════════════════════════════════════

/// @brief Thinking 包含文本和 redacted 标记
TEST_CASE("Thinking construction") {
    Thinking th{"thinking text", false};
    CHECK(th.text == "thinking text");
    CHECK(th.redacted == false);
}

/// @brief redacted 默认为 false
TEST_CASE("Thinking default redacted") {
    Thinking th{"text"};
    CHECK(th.text == "text");
    CHECK(th.redacted == false);
}

// ═══════════════════════════════════════════════════
// ToolCall
// ═══════════════════════════════════════════════════

/// @brief ToolCall 包含 id、工具名和 JSON 参数对象
TEST_CASE("ToolCall construction") {
    ToolCall tc{"id1", "get_weather", {{"city", "杭州"}}};
    CHECK(tc.id == "id1");
    CHECK(tc.name == "get_weather");
    CHECK(tc.arguments["city"] == "杭州");
}

/// @brief ToolCall arguments 可接受空 JSON
TEST_CASE("ToolCall empty arguments") {
    ToolCall tc{"id2", "fn", nullptr};
    CHECK(tc.id == "id2");
    CHECK(tc.name == "fn");
}

// ═══════════════════════════════════════════════════
// ContentBlock variant
// ═══════════════════════════════════════════════════

/// @brief ContentBlock 可存储 Text 并通过 visit 访问
TEST_CASE("ContentBlock visit Text") {
    ContentBlock cb = Text{"hello"};
    std::visit([](auto const& b) {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, Text>)
            CHECK(b.text == "hello");
        else
            FAIL("wrong type");
    }, cb);
}

/// @brief ContentBlock 可存储 Image 并通过 visit 访问
TEST_CASE("ContentBlock visit Image") {
    ContentBlock cb = Image{"data", "image/jpeg"};
    std::visit([](auto const& b) {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, Image>) {
            CHECK(b.data == "data");
            CHECK(b.mime_type == "image/jpeg");
        } else {
            FAIL("wrong type");
        }
    }, cb);
}

/// @brief ContentBlock 可存储 Thinking 并通过 visit 访问
TEST_CASE("ContentBlock visit Thinking") {
    ContentBlock cb = Thinking{"think", true};
    std::visit([](auto const& b) {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, Thinking>) {
            CHECK(b.text == "think");
            CHECK(b.redacted == true);
        } else {
            FAIL("wrong type");
        }
    }, cb);
}

/// @brief ContentBlock 可存储 ToolCall 并通过 visit 访问
TEST_CASE("ContentBlock visit ToolCall") {
    ContentBlock cb = ToolCall{"id", "tool", {{"k", "v"}}};
    std::visit([](auto const& b) {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, ToolCall>) {
            CHECK(b.id == "id");
            CHECK(b.name == "tool");
            CHECK(b.arguments["k"] == "v");
        } else {
            FAIL("wrong type");
        }
    }, cb);
}

// ═══════════════════════════════════════════════════
// Role
// ═══════════════════════════════════════════════════

/// @brief Role 枚举值按序排列
TEST_CASE("Role values") {
    CHECK(static_cast<int>(Role::User) == 0);
    CHECK(static_cast<int>(Role::Assistant) == 1);
    CHECK(static_cast<int>(Role::ToolResult) == 2);
}

// ═══════════════════════════════════════════════════
// Message
// ═══════════════════════════════════════════════════

/// @brief 默认构造：内容块列表为空
TEST_CASE("Message default construction") {
    Message msg;
    CHECK(msg.content.empty());
}

/// @brief 指定角色和内容块列表构造
TEST_CASE("Message construction with blocks") {
    Message msg{Role::User, {Text{"hello"}, Text{"world"}}};
    CHECK(msg.role == Role::User);
    CHECK(msg.content.size() == 2);
}

/// @brief 拷贝构造深拷贝 Extra（若存在）
TEST_CASE("Message copy construction") {
    Message orig{Role::Assistant, {Text{"hi"}}};
    orig.set_input_tokens(42);
    Message copy(orig);
    CHECK(copy.role == Role::Assistant);
    CHECK(copy.content.size() == 1);
    CHECK(copy.input_tokens() == 42);
}

/// @brief 拷贝赋值深拷贝 Extra，源无 Extra 则目标释放
TEST_CASE("Message copy assignment") {
    Message a{Role::User, {Text{"a"}}};
    a.set_output_tokens(10);
    Message b{Role::Assistant, {Text{"b"}}};
    b = a;
    CHECK(b.role == Role::User);
    CHECK(b.output_tokens() == 10);
}

/// @brief 自赋值安全（不复制、不崩溃）
TEST_CASE("Message self assignment") {
    Message m{Role::User, {Text{"x"}}};
    m.set_input_tokens(5);
    m = m;
    CHECK(m.role == Role::User);
    CHECK(m.input_tokens() == 5);
}

/// @brief operator[] 按索引访问内容块
TEST_CASE("Message operator[]") {
    Message msg{Role::User, {Text{"first"}, Text{"second"}}};
    CHECK(std::get<Text>(msg[0]).text == "first");
    CHECK(std::get<Text>(msg[1]).text == "second");
}

/// @brief 可变迭代器可遍历内容块列表
TEST_CASE("Message iterators") {
    Message msg{Role::User, {Text{"a"}, Text{"b"}}};
    int count = 0;
    for (auto& cb : msg) { ++count; (void)cb; }
    CHECK(count == 2);
}

/// @brief 常迭代器可遍历 const Message
TEST_CASE("Message const iterators") {
    Message const msg{Role::User, {Text{"a"}}};
    int count = 0;
    for (auto const& cb : msg) { ++count; (void)cb; }
    CHECK(count == 1);
}

/// @brief 未设置 Extra 时 has_extra 为 false
TEST_CASE("Message has_extra initially false") {
    Message msg{Role::User, {Text{"test"}}};
    CHECK(msg.has_extra() == false);
}

/// @brief set_input_tokens 自动创建 Extra，getter 返回设置值
TEST_CASE("Message input_tokens") {
    Message msg{Role::User, {Text{"test"}}};
    CHECK(msg.input_tokens() == std::nullopt);
    msg.set_input_tokens(100);
    CHECK(msg.has_extra() == true);
    CHECK(msg.input_tokens() == 100);
}

/// @brief set_output_tokens 自动创建 Extra
TEST_CASE("Message output_tokens") {
    Message msg{Role::User, {Text{"test"}}};
    CHECK(msg.output_tokens() == std::nullopt);
    msg.set_output_tokens(50);
    CHECK(msg.output_tokens() == 50);
}

/// @brief set_response_id 自动创建 Extra
TEST_CASE("Message response_id") {
    Message msg{Role::Assistant, {Text{"resp"}}};
    CHECK(msg.response_id() == std::nullopt);
    msg.set_response_id("resp-1");
    CHECK(msg.response_id() == "resp-1");
}

/// @brief set_tool_call_id 用于关联 ToolResult
TEST_CASE("Message tool_call_id") {
    Message msg{Role::ToolResult, {Text{"result"}}};
    CHECK(msg.tool_call_id() == std::nullopt);
    msg.set_tool_call_id("call-1");
    CHECK(msg.tool_call_id() == "call-1");
}

/// @brief is_error 表示工具执行是否出错
TEST_CASE("Message is_error") {
    Message msg{Role::ToolResult, {Text{"err"}}};
    CHECK(msg.is_error() == std::nullopt);
    msg.set_is_error(true);
    CHECK(msg.is_error() == true);
    msg.set_is_error(false);
    CHECK(msg.is_error() == false);
}

/// @brief push_back / pop_back 增删内容块
TEST_CASE("Message push_back and pop_back") {
    Message msg{Role::User, {}};
    CHECK(msg.content.empty());
    msg.push_back(Text{"a"});
    CHECK(msg.content.size() == 1);
    msg.push_back(Text{"b"});
    CHECK(msg.content.size() == 2);
    msg.pop_back();
    CHECK(msg.content.size() == 1);
}

/// @brief clear 清空所有内容块
TEST_CASE("Message clear") {
    Message msg{Role::User, {Text{"a"}, Text{"b"}}};
    msg.clear();
    CHECK(msg.content.empty());
}

/// @brief back 访问末尾内容块
TEST_CASE("Message back") {
    Message msg{Role::User, {Text{"first"}, Text{"last"}}};
    CHECK(std::get<Text>(msg.back()).text == "last");
    CHECK(std::get<Text>(msg.back()).text == "last");
}

// ═══════════════════════════════════════════════════
// Content
// ═══════════════════════════════════════════════════

/// @brief 默认构造：空提示词、空消息列表
TEST_CASE("Content default construction") {
    Content ctx;
    CHECK(ctx.system_prompt.empty());
    CHECK(ctx.messages.empty());
}

/// @brief 仅指定系统提示词
TEST_CASE("Content with system prompt") {
    Content ctx{"You are a helpful assistant"};
    CHECK(ctx.system_prompt == "You are a helpful assistant");
    CHECK(ctx.messages.empty());
}

/// @brief 完整构造：系统提示词 + 消息列表
TEST_CASE("Content with messages") {
    Content ctx{"sys", {
        Message{Role::User, {Text{"hi"}}},
        Message{Role::Assistant, {Text{"hello"}}}
    }};
    CHECK(ctx.system_prompt == "sys");
    CHECK(ctx.size() == 2);
    CHECK(ctx[0].role == Role::User);
    CHECK(ctx[1].role == Role::Assistant);
}

/// @brief push_back 添加消息
TEST_CASE("Content push_back") {
    Content ctx{"sys"};
    ctx.push_back(Message{Role::User, {Text{"q"}}});
    CHECK(ctx.size() == 1);
}

/// @brief pop_back 移除末尾消息
TEST_CASE("Content pop_back") {
    Content ctx{"sys", {
        Message{Role::User, {Text{"a"}}},
        Message{Role::User, {Text{"b"}}}
    }};
    ctx.pop_back();
    CHECK(ctx.size() == 1);
    CHECK(ctx[0].role == Role::User);
}

/// @brief clear 清空所有消息
TEST_CASE("Content clear") {
    Content ctx{"sys", {Message{Role::User, {Text{"a"}}}}};
    ctx.clear();
    CHECK(ctx.messages.empty());
}

/// @brief back 访问末尾消息
TEST_CASE("Content back") {
    Content ctx{"sys", {
        Message{Role::User, {Text{"first"}}},
        Message{Role::Assistant, {Text{"last"}}}
    }};
    CHECK(ctx.back().role == Role::Assistant);
}

/// @brief 可变迭代器可遍历消息列表
TEST_CASE("Content iterators") {
    Content ctx{"sys", {
        Message{Role::User, {Text{"a"}}},
        Message{Role::User, {Text{"b"}}}
    }};
    int count = 0;
    for (auto& msg : ctx) { ++count; (void)msg; }
    CHECK(count == 2);
}

/// @brief 常迭代器可遍历 const Content
TEST_CASE("Content const iterators") {
    Content const ctx{"sys", {Message{Role::User, {Text{"a"}}}}};
    int count = 0;
    for (auto const& msg : ctx) { ++count; (void)msg; }
    CHECK(count == 1);
}

/// @brief size 随增删操作动态变化
TEST_CASE("Content size after modifications") {
    Content ctx{"sys"};
    CHECK(ctx.size() == 0);
    ctx.push_back(Message{Role::User, {Text{"m1"}}});
    CHECK(ctx.size() == 1);
    ctx.push_back(Message{Role::User, {Text{"m2"}}});
    CHECK(ctx.size() == 2);
    ctx.pop_back();
    CHECK(ctx.size() == 1);
    ctx.clear();
    CHECK(ctx.size() == 0);
}

/// @brief operator[] 可访问嵌套的内容块
TEST_CASE("Content operator[] returns correct message") {
    Content ctx{"sys", {
        Message{Role::User, {Text{"user msg"}}},
        Message{Role::Assistant, {Text{"assistant msg"}}}
    }};
    CHECK(std::get<Text>(ctx[0][0]).text == "user msg");
    CHECK(std::get<Text>(ctx[1][0]).text == "assistant msg");
}

// ═══════════════════════════════════════════════════
// Integration
// ═══════════════════════════════════════════════════

/// @brief 拷贝构造时 Extra 完整深拷贝，所有字段保留
TEST_CASE("Extra preserved through message copy") {
    Message orig{Role::User, {Text{"test"}}};
    orig.set_input_tokens(10);
    orig.set_output_tokens(20);
    orig.set_response_id("resp-123");

    Message copy = orig;
    CHECK(copy.has_extra());
    CHECK(copy.input_tokens() == 10);
    CHECK(copy.output_tokens() == 20);
    CHECK(copy.response_id() == "resp-123");
}

/// @brief 拷贝后修改不共享，确认深拷贝
TEST_CASE("Extra not shared after copy") {
    Message a{Role::User, {Text{"a"}}};
    a.set_input_tokens(5);
    Message b = a;
    b.set_input_tokens(99);
    CHECK(a.input_tokens() == 5);
    CHECK(b.input_tokens() == 99);
}
