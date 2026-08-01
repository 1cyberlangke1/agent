#pragma once

// 多模态图片支持：模型不支持图片输入时的降级（对齐 pi transform-messages.ts
// 的 downgradeUnsupportedImages）。三引擎 convert_messages 统一调用。

#include <agent/llm/content.hpp>

#include <string_view>
#include <vector>

namespace agent::detail {

/// @brief 模型不支持图片时，user 消息里的图片替换为的占位符文本（对齐 pi）。
inline constexpr std::string_view kNonVisionUserImagePlaceholder =
    "(image omitted: model does not support images)";
/// @brief 模型不支持图片时，tool_result 消息里的图片占位符（对齐 pi）。
inline constexpr std::string_view kNonVisionToolImagePlaceholder =
    "(tool image omitted: model does not support images)";

/// @brief 模型不支持图片（supports_image == false）时，把消息里的 Image 内容块
///        替换为占位符文本；连续多张图合并成一个占位符（对齐 pi replaceImagesWithPlaceholder）。
///        直接修改传入的 messages；supports_image 为 true 时原样返回。
/// @param messages       待转换的消息（引擎 convert_messages 内部拷贝后传入）
/// @param supports_image 模型是否支持图片输入（ModelView.supports_image_input）
void downgrade_unsupported_images(std::vector<Message>& messages, bool supports_image);

}  // namespace agent::detail
