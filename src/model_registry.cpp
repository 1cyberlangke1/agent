// ModelRegistry 纯静态类实现。
// 静态成员（index_/runtime_models_ 等）定义在 llm.hpp（inline static），
// 方法实现在本文件；init 加载生成器产物的内置表（kAllProviders）。

#include <agent/llm.hpp>
#include <agent/models/generated.hpp>

#include <mutex>

namespace agent {

/// 惰性加载内置表：把生成器产物的全部模型灌入 index_ 与 static_order_。
/// call_once 保证只执行一次；内部持 unique_lock 写容器，
/// 与各方法的锁互斥（先 init 再拿锁，call_once 完成后不再执行，无死锁）。
void ModelRegistry::init()
{
    std::call_once(init_flag_, [] {
        std::unique_lock lock(mutex_);
        for (auto const& table : detail::kAllProviders) {
            for (auto const& model : table.models) {
                static_order_.push_back(model.id);
                index_.emplace(model.id, to_view(model));
            }
        }
    });
}

ModelView ModelRegistry::to_view(detail::BuiltinModel const& m)
{
    ModelView v;
    v.id = m.id;
    v.context_window = m.context_window;
    v.max_output_tokens = m.max_output_tokens;
    v.reasoning = m.reasoning;
    v.supports_image_input = m.supports_image_input;
    v.thinking_level_map = m.thinking_level_map;
    v.thinking_field = m.thinking_field;
    return v;
}

ModelView ModelRegistry::to_view(RuntimeModel const& m)
{
    ModelView v;
    v.id = m.id;
    v.context_window = m.context_window;
    v.max_output_tokens = m.max_output_tokens;
    v.reasoning = m.reasoning;
    v.supports_image_input = m.supports_image_input;
    for (std::size_t i = 0; i < thinking_level_count; ++i) {
        if (m.thinking_level_map[i])
            v.thinking_level_map[i] = *m.thinking_level_map[i];
    }
    v.thinking_field = m.thinking_field;
    return v;
}

bool ModelRegistry::register_model(RuntimeModel model)
{
    init();
    std::unique_lock lock(mutex_);
    bool is_new = !index_.contains(model.id);
    // deque 只追加、旧条目永不改写/删除——已发出的 ModelView 的 string_view
    // 指向旧条目仍有效；index_ 改指新条目（覆盖语义）。
    runtime_models_.push_back(std::move(model));
    RuntimeModel& entry = runtime_models_.back();
    dynamic_ids_.insert(entry.id);
    index_[entry.id] = to_view(entry);
    return is_new;
}

std::optional<ModelView> ModelRegistry::find_model(std::string_view id)
{
    init();
    std::shared_lock lock(mutex_);
    auto it = index_.find(id);
    if (it == index_.end())
        return std::nullopt;
    return it->second;
}

/// 把请求等级收敛到模型支持的范围：从请求等级向下找第一个支持的档位。
/// Off 恒「支持」（引擎不发 thinking 参数即关闭）；全部不支持 → Off。
ThinkingLevel clamp_thinking_level(ModelView const& model, ThinkingLevel level)
{
    int idx = static_cast<int>(level);
    while (idx >= 1 && !model.thinking_level_map[static_cast<std::size_t>(idx)].has_value())
        --idx;
    return static_cast<ThinkingLevel>(idx);
}

}  // namespace agent
