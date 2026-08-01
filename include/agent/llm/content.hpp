#pragma once

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace agent {

/// @brief 文本内容块
///
/// 纯文本载体，用于表示消息中的文字内容。
/// 适配 OpenAI text / Anthropic text / Gemini text 内容块。
struct Text {
    /// UTF-8 编码的文本字符串
    std::string text;
};

/// @brief 图片内容块（base64 编码）
///
/// 以 base64 编码数据和 MIME 类型表示图片。
/// 适配 OpenAI image_url（base64 模式）/ Anthropic image 内容块。
struct Image {
    /// base64 编码的图片二进制数据
    std::string data;
    /// 图片 MIME 类型，如 "image/png"、"image/jpeg"、"image/webp"
    std::string mime_type;
};

/// @brief 思考链内容块
///
/// 表示模型推理过程中的内部思考内容。
/// 适配 DeepSeek reasoning_content、Anthropic thinking 内容块。
struct Thinking {
    /// 思考内容文本
    std::string text;
    /// 思考内容是否因安全策略被隐藏（Anthropic redacted_thinking）
    bool redacted = false;
    /// Anthropic thinking 块签名（signature_delta 累积）。
    /// 多轮对话回传 assistant 思考块时必须原样带回，否则 Anthropic 拒绝请求；
    /// 无签名（如中断流）时引擎会降级为普通文本块。其他厂商恒为空。
    std::string signature;
};

/// @brief 工具调用内容块
///
/// 由模型发出的工具调用请求，包含工具名称和参数。
/// 执行结果应作为 Role::ToolResult 消息返回。
struct ToolCall {
    /// 工具调用唯一标识符，用于关联执行结果
    std::string id;
    /// 要调用的工具名称
    std::string name;
    /// 工具参数，JSON 对象格式。
    /// 键为参数名，值为参数值，结构与对应工具的 params_type 字段对齐。
    /// 例如 get_weather: {"location":"杭州","unit":"celsius"}
    nlohmann::json arguments;
};

/// @brief 工具执行结果内容块
///
/// 工具调用执行完成后回传给模型的工具输出。
/// 作为 Role::ToolResult 消息的内容承载，与 ToolCall.id 关联。
struct ToolResult {
    /// 关联的 ToolCall.id
    std::string tool_call_id;
    /// 工具输出文本（或 JSON 字符串）
    std::string output;
    /// 工具执行是否出错
    bool is_error = false;
};

/// @brief 消息中的一块内容
///
/// 通过 std::variant 实现多态，支持五种类型：
/// - Text：文本
/// - Image：图片
/// - Thinking：思考链
/// - ToolCall：工具调用
/// - ToolResult：工具执行结果
///
/// @usage 使用 std::visit 访问具体类型：
///        std::visit([](auto const& b) { ... }, block);
using ContentBlock = std::variant<Text, Image, Thinking, ToolCall, ToolResult>;

/// @brief 消息角色
///
/// 标识消息的发送方，用于构建多轮对话历史。
enum class Role {
    /// 用户发送的消息（query / instruction）
    User,
    /// 助手（模型）的回复
    Assistant,
    /// 工具执行结果回传
    ToolResult,
};

/// @brief 一条消息，包含角色和内容块数组
///
/// 消息由角色和内容块列表组成。
/// 部分元数据（token 计数、API 响应 ID 等）存储在 Extra 中，
/// 通过 unique_ptr 延迟分配以节约内存。
class Message {
public:
    /// 消息角色：User / Assistant / ToolResult
    Role role;

    /// @brief 默认构造空消息（角色未初始化）
    Message() = default;

    /// @brief 构造消息
    ///
    /// @param r      消息角色
    /// @param blocks 内容块列表
    Message(Role r, std::vector<ContentBlock> blocks)
        : role(r), content(std::move(blocks)) {}

    /// @brief 拷贝构造
    ///
    /// 深拷贝 Extra（若存在）。
    ///
    /// @param other 源消息
    Message(Message const& other)
        : role(other.role)
        , content(other.content)
    {
        if (other.extra_)
            extra_ = std::make_unique<Extra>(*other.extra_);
    }

    /// @brief 拷贝赋值
    ///
    /// 深拷贝 Extra（若存在），反之释放当前 Extra。
    ///
    /// @param other 源消息
    /// @return 自身引用
    Message& operator=(Message const& other)
    {
        if (this != &other) {
            role = other.role;
            content = other.content;
            if (other.extra_)
                extra_ = std::make_unique<Extra>(*other.extra_);
            else
                extra_.reset();
        }
        return *this;
    }

    /// @brief 按索引访问内容块（可变版本）
    ///
    /// @param i 内容块索引
    /// @return 对应位置的 ContentBlock 引用
    /// @pre  i < content.size()
    /// @note 不进行边界检查，越界行为同 std::vector
    ContentBlock& operator[](size_t i) { return content[i]; }

    /// @brief 按索引访问内容块（只读版本）
    ///
    /// @param i 内容块索引
    /// @return 对应位置的 ContentBlock const 引用
    /// @pre  i < content.size()
    ContentBlock const& operator[](size_t i) const { return content[i]; }

    /// @brief 内容块列表起始迭代器（可变）
    auto begin() { return content.begin(); }
    /// @brief 内容块列表末尾迭代器（可变）
    auto end()   { return content.end(); }
    /// @brief 内容块列表起始迭代器（只读）
    auto begin() const { return content.begin(); }
    /// @brief 内容块列表末尾迭代器（只读）
    auto end() const { return content.end(); }

    /// 消息的内容块列表
    std::vector<ContentBlock> content;

    /// @brief 检查 Extra 是否已分配
    ///
    /// Extra 存储可选元数据（token 计数、API 响应 ID 等），
    /// 仅在需要时通过 unique_ptr 分配。
    ///
    /// @return true 表示 Extra 已分配，setter/getter 可正常工作
    /// @note  getter 不触发分配，setter 在 Extra 为空时自动创建
    bool has_extra() const { return extra_ != nullptr; }

    /// @brief 获取输入 token 数
    ///
    /// @return Extra 未分配时返回 std::nullopt；
    ///         否则返回输入 token 计数值。
    std::optional<size_t> input_tokens() const
    {
        if (!extra_) return std::nullopt;
        return extra_->input_tokens;
    }

    /// @brief 设置输入 token 数
    ///
    /// Extra 未分配时自动创建。
    ///
    /// @param v 输入 token 计数值
    void set_input_tokens(size_t v) { ensure_extra().input_tokens = v; }

    /// @brief 获取输出 token 数
    ///
    /// @return Extra 未分配时返回 std::nullopt；
    ///         否则返回输出 token 计数值。
    std::optional<size_t> output_tokens() const
    {
        if (!extra_) return std::nullopt;
        return extra_->output_tokens;
    }

    /// @brief 设置输出 token 数
    ///
    /// Extra 未分配时自动创建。
    ///
    /// @param v 输出 token 计数值
    void set_output_tokens(size_t v) { ensure_extra().output_tokens = v; }

    /// @brief 获取 API 响应 ID
    ///
    /// @return Extra 未分配时返回 std::nullopt；
    ///         否则返回 API 返回的响应标识符。
    std::optional<std::string> response_id() const
    {
        if (!extra_) return std::nullopt;
        return extra_->response_id;
    }

    /// @brief 设置 API 响应 ID
    ///
    /// Extra 未分配时自动创建。
    ///
    /// @param v API 响应标识符
    void set_response_id(std::string v) { ensure_extra().response_id = std::move(v); }

    /// @brief 获取工具调用 ID
    ///
    /// 用于关联 Role::ToolResult 消息与对应的工具调用。
    ///
    /// @return Extra 未分配时返回 std::nullopt；
    ///         否则返回工具调用标识符。
    std::optional<std::string> tool_call_id() const
    {
        if (!extra_) return std::nullopt;
        return extra_->tool_call_id;
    }

    /// @brief 设置工具调用 ID
    ///
    /// Extra 未分配时自动创建。
    ///
    /// @param v 工具调用标识符
    void set_tool_call_id(std::string v) { ensure_extra().tool_call_id = std::move(v); }

    /// @brief 获取工具执行错误状态
    ///
    /// @return Extra 未分配时返回 std::nullopt；
    ///         否则返回工具执行是否出错。
    std::optional<bool> is_error() const
    {
        if (!extra_) return std::nullopt;
        return extra_->is_error;
    }

    /// @brief 设置工具执行错误状态
    ///
    /// Extra 未分配时自动创建。
    ///
    /// @param v true 表示工具执行出错
    void set_is_error(bool v) { ensure_extra().is_error = v; }

    /// @brief 在末尾添加内容块
    ///
    /// @param b 要添加的内容块
    void push_back(ContentBlock b) { content.push_back(std::move(b)); }

    /// @brief 移除末尾的内容块
    ///
    /// @pre  !content.empty()
    void pop_back()  { content.pop_back(); }

    /// @brief 清空所有内容块
    void clear()     { content.clear(); }

    /// @brief 访问末尾的内容块（可变版本）
    ///
    /// @return 末尾 ContentBlock 的引用
    /// @pre  !content.empty()
    ContentBlock& back()       { return content.back(); }

    /// @brief 访问末尾的内容块（只读版本）
    ///
    /// @return 末尾 ContentBlock 的 const 引用
    /// @pre  !content.empty()
    ContentBlock const& back() const { return content.back(); }

private:
    /// @brief 可选元数据
    ///
    /// 通过 unique_ptr 延迟分配，仅在需要时占用内存。
    struct Extra {
        /// API 返回的响应标识符（用于追踪和调试）
        std::string response_id;
        /// 输入 token 计数值
        size_t input_tokens = 0;
        /// 输出 token 计数值
        size_t output_tokens = 0;
        /// 工具调用关联 ID
        std::string tool_call_id;
        /// 工具执行是否出错
        bool is_error = false;
    };
    std::unique_ptr<Extra> extra_;

    /// @brief 确保 Extra 已分配并返回引用
    ///
    /// @return Extra 引用（分配后保证存在）
    Extra& ensure_extra()
    {
        if (!extra_) extra_ = std::make_unique<Extra>();
        return *extra_;
    }
};

/// @brief 完整对话上下文
///
/// 包含系统提示词和按对话顺序排列的消息列表。
/// 作为 LLM 请求的输入结构，传递给各 provider 的接口。
class Content {
public:
    /// 系统提示词，设定助手的行为和角色
    std::string system_prompt;
    /// 消息列表，按对话时间顺序排列
    std::vector<Message> messages;

    /// @brief 默认构造空上下文
    Content() = default;

    /// @brief 仅设置系统提示词
    ///
    /// @param sp 系统提示词字符串
    explicit Content(std::string sp)
        : system_prompt(std::move(sp)) {}

    /// @brief 设置系统提示词和消息列表
    ///
    /// @param sp   系统提示词字符串
    /// @param msgs 消息列表
    Content(std::string sp, std::vector<Message> msgs)
        : system_prompt(std::move(sp)), messages(std::move(msgs)) {}

    /// @brief 按索引访问消息（可变版本）
    ///
    /// @param i 消息索引
    /// @return 对应位置的 Message 引用
    /// @pre  i < messages.size()
    Message& operator[](size_t i) { return messages[i]; }

    /// @brief 按索引访问消息（只读版本）
    ///
    /// @param i 消息索引
    /// @return 对应位置的 Message const 引用
    /// @pre  i < messages.size()
    Message const& operator[](size_t i) const { return messages[i]; }

    /// @brief 在末尾添加消息
    ///
    /// @param m 要添加的消息
    void push_back(Message m) { messages.push_back(std::move(m)); }

    /// @brief 移除末尾消息
    ///
    /// @pre  !messages.empty()
    void pop_back()  { messages.pop_back(); }

    /// @brief 清空所有消息
    void clear()     { messages.clear(); }

    /// @brief 访问末尾消息（可变版本）
    ///
    /// @return 末尾 Message 的引用
    /// @pre  !messages.empty()
    Message& back()       { return messages.back(); }

    /// @brief 访问末尾消息（只读版本）
    ///
    /// @return 末尾 Message 的 const 引用
    /// @pre  !messages.empty()
    Message const& back() const { return messages.back(); }

    /// @brief 返回消息数量
    ///
    /// @return messages.size()
    auto size() const { return messages.size(); }

    /// @brief 消息列表起始迭代器（可变）
    auto begin() { return messages.begin(); }
    /// @brief 消息列表末尾迭代器（可变）
    auto end()   { return messages.end(); }
    /// @brief 消息列表起始迭代器（只读）
    auto begin() const { return messages.begin(); }
    /// @brief 消息列表末尾迭代器（只读）
    auto end() const { return messages.end(); }
};

} // namespace agent
