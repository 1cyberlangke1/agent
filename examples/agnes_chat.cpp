// Agnes (sglang) Provider 真实验证：注册 agnes-2.5-flash，complete/stream 塞图，
// 验证 reasoning_content 思考提取 + 图片识别。
//
// 编译：
//   cmake --build --preset default --target agnes_chat
//   ./build/examples/agnes_chat.exe <api_key>

#include <agent/llm.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <variant>

using namespace agent;

namespace {

std::string base64_encode(unsigned char const* data, size_t len)
{
    static constexpr char kTable[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        unsigned v = static_cast<unsigned>(data[i]) << 16;
        if (i + 1 < len) v |= static_cast<unsigned>(data[i + 1]) << 8;
        if (i + 2 < len) v |= static_cast<unsigned>(data[i + 2]);
        out += kTable[(v >> 18) & 0x3f];
        out += kTable[(v >> 12) & 0x3f];
        out += (i + 1 < len) ? kTable[(v >> 6) & 0x3f] : '=';
        out += (i + 2 < len) ? kTable[v & 0x3f] : '=';
    }
    return out;
}

void print_blocks(std::vector<ContentBlock> const& blocks)
{
    for (auto const& b : blocks) {
        std::visit([](auto const& blk) {
            using T = std::decay_t<decltype(blk)>;
            if constexpr (std::is_same_v<T, Text>)
                std::cout << "[正文] " << blk.text << "\n";
            else if constexpr (std::is_same_v<T, Thinking>)
                std::cout << "[思考] " << blk.text << "\n";
            else if constexpr (std::is_same_v<T, ToolCall>)
                std::cout << "[工具] " << blk.name << " " << blk.arguments.dump() << "\n";
            else
                std::cout << "[其他块]\n";
        }, b);
    }
}

}  // namespace

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "用法：agnes_chat <api_key> [image_file]\n";
        return 1;
    }
    std::string const key = argv[1];

    // 注册 agnes-2.5-flash：推理 toggle 型 + 视觉模型（非 Off 档都映射 "on"）
    std::array<std::optional<std::string>, thinking_level_count> thinking_map{};
    thinking_map[0] = "off";
    for (std::size_t i = 1; i < thinking_level_count; ++i)
        thinking_map[i] = "on";
    ModelRegistry::register_model(RuntimeModel{
        .id = "agnes-2.5-flash",
        .context_window = 512000,
        .max_output_tokens = 65536,
        .reasoning = true,
        .supports_image_input = true,
        .thinking_level_map = thinking_map,
        .thinking_field = "",
    });

    ModelView model = *ModelRegistry::find_model("agnes-2.5-flash");
    AgnesProvider agnes({.name = "agnes",
                         .api_key = key,
                         .base_url = "https://apihub.agnes-ai.com/v1" });

    // ── 测试 1：非流式 complete 纯文本，验证 reasoning_content 提取 ──
    std::cout << "═══ 测试 1：complete 纯文本（reasoning=High）═══\n" << std::flush;
    Context ctx;
    ctx.messages.push_back(Message{ Role::User, { Text{ "用一句话解释什么是协程" } } });
    StreamOptions opts;
    opts.reasoning = ThinkingLevel::High;
    Result<ChatResponse> resp = agnes.complete(model, ctx, opts);
    if (!resp) {
        std::cout << "失败：" << resp.error().message << "\n";
        return 1;
    }
    print_blocks(resp->content);
    std::cout << "[用量] 输入 " << resp->usage.input_tokens << " 输出 " << resp->usage.output_tokens
              << " 总计 " << resp->usage.total_tokens << "\n\n";

    // ── 测试 2：非流式 complete 带图（读本地图片文件 → base64 Image 块）──
    std::cout << "═══ 测试 2：complete 带图（base64 image_url）═══\n" << std::flush;
    if (argc < 3) {
        std::cout << "跳过：未提供 image_file\n\n";
        return 0;
    }
    std::ifstream img_file(argv[2], std::ios::binary);
    if (!img_file) {
        std::cout << "打开图片失败：" << argv[2] << "\n";
        return 1;
    }
    std::ostringstream img_buf;
    img_buf << img_file.rdbuf();
    std::string const img_bytes = img_buf.str();
    std::string b64 = base64_encode(
        reinterpret_cast<unsigned char const*>(img_bytes.data()), img_bytes.size());
    std::cout << "[图片] " << img_bytes.size() << " 字节 → base64\n";
    Context ctx2;
    ctx2.messages.push_back(Message{ Role::User,
                                     { Text{ "What is in this image? Answer in English briefly." },
                                       Image{ b64, "image/jpeg" } } });
    Result<ChatResponse> resp2 = agnes.complete(model, ctx2, StreamOptions{});
    if (!resp2) {
        std::cout << "失败：" << resp2.error().message << "\n";
        return 1;
    }
    print_blocks(resp2->content);
    std::cout << "[用量] 输入 " << resp2->usage.input_tokens << " 输出 " << resp2->usage.output_tokens
              << " 总计 " << resp2->usage.total_tokens << "\n\n";

    // ── 测试 3：流式 ThinkingDelta 增量提取 ──
    std::cout << "═══ 测试 3：stream 流式（ThinkingDelta 增量）═══\n" << std::flush;
    Context ctx3;
    ctx3.messages.push_back(Message{ Role::User,
                                     { Text{ "What is in this image? Answer in English briefly." },
                                       Image{ b64, "image/jpeg" } } });
    int thinking_chunks = 0;
    int text_chunks = 0;
    for (StreamEvent const& ev : agnes.stream(model, ctx3, opts)) {
        switch (ev.type()) {
            case StreamEvent::Type::ThinkingDelta:
                ++thinking_chunks;
                break;
            case StreamEvent::Type::TextDelta:
                ++text_chunks;
                break;
            case StreamEvent::Type::Error:
                std::cout << "[流错误] " << std::get<Error>(ev.data).message << "\n";
                break;
            default:
                break;
        }
    }
    std::cout << "思考增量块 " << thinking_chunks << " 个 · 正文增量块 " << text_chunks << " 个\n";
    return 0;
}
