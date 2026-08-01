// 多模态图片降级实现（模型不支持图片时的占位符替换，对齐 pi transform-messages.ts）。

#include <agent/llm/engine/image_support.hpp>

#include <string>
#include <vector>

namespace agent::detail {

void downgrade_unsupported_images(std::vector<Message>& messages, bool supports_image)
{
    if (supports_image)
        return;
    for (auto& msg : messages) {
        if (msg.role != Role::User && msg.role != Role::ToolResult)
            continue;
        std::string_view placeholder = msg.role == Role::ToolResult
            ? kNonVisionToolImagePlaceholder
            : kNonVisionUserImagePlaceholder;
        std::vector<ContentBlock> out;
        out.reserve(msg.content.size());
        bool previous_was_placeholder = false;
        for (auto& block : msg.content) {
            if (std::get_if<Image>(&block)) {
                // 连续图片只保留一个占位符（对齐 pi：previousWasPlaceholder 去重）
                if (!previous_was_placeholder)
                    out.push_back(Text{ std::string(placeholder) });
                previous_was_placeholder = true;
                continue;
            }
            bool is_placeholder_text = false;
            if (auto t = std::get_if<Text>(&block))
                is_placeholder_text = (t->text == placeholder);
            out.push_back(std::move(block));
            previous_was_placeholder = is_placeholder_text;
        }
        msg.content = std::move(out);
    }
}

}  // namespace agent::detail
