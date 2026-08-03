// ModelRegistry 纯静态类实现（COW 不可变快照）。
// 读路径（find/for_each）只 atomic load 快照，零锁；写路径（register）持 write_mutex_
// 重建快照原子替换。runtime_models_ deque 地址稳定、只追加，快照的 ModelView 跨快照有效。

#include <agent/llm/model.hpp>
#include <agent/llm/models/generated.hpp>

#include <mutex>

namespace agent {

/// @brief 惰性加载内置表（假定已持 write_mutex_）：构建初始快照。幂等。
void ModelRegistry::init()
{
    if (snapshot_.load())
        return;
    std::shared_ptr<detail::RegistrySnapshot> snap = std::make_shared<detail::RegistrySnapshot>();
    for (auto const& table : detail::kAllProviders) {
        for (auto const& model : table.models) {
            snap->static_order.push_back(model.id);
            snap->index.emplace(model.id, to_view(model));
        }
    }
    snapshot_.store(std::move(snap));
}

std::shared_ptr<detail::RegistrySnapshot> ModelRegistry::load_snapshot()
{
    if (std::shared_ptr<detail::RegistrySnapshot> snap = snapshot_.load())
        return snap;
    std::lock_guard lock(write_mutex_);
    init();   // init 假定已持锁（幂等）
    return snapshot_.load();
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
    v.price_input = m.price_input;
    v.price_output = m.price_output;
    v.price_cache_read = m.price_cache_read;
    v.price_cache_write = m.price_cache_write;
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
    v.price_input = m.price_input;
    v.price_output = m.price_output;
    v.price_cache_read = m.price_cache_read;
    v.price_cache_write = m.price_cache_write;
    return v;
}

bool ModelRegistry::register_model(RuntimeModel model)
{
    std::lock_guard lock(write_mutex_);
    init();   // init 假定已持锁（幂等）
    std::shared_ptr<detail::RegistrySnapshot> old = snapshot_.load();
    bool is_new = !old->index.contains(model.id);

    // deque 只追加、旧条目永不改写/删除——已发出的 ModelView 的 string_view
    // 指向旧条目仍有效；快照把 id 改指新条目（覆盖语义）。
    runtime_models_.push_back(std::move(model));
    RuntimeModel& entry = runtime_models_.back();

    // COW：复制旧快照 → 改 → 原子替换
    std::shared_ptr<detail::RegistrySnapshot> snap = std::make_shared<detail::RegistrySnapshot>();
    snap->index = old->index;
    snap->dynamic_ids = old->dynamic_ids;
    snap->static_order = old->static_order;
    snap->dynamic_order = old->dynamic_order;
    snap->dynamic_ids.insert(entry.id);
    snap->dynamic_order.push_back(entry.id);
    snap->index[entry.id] = to_view(entry);
    snapshot_.store(std::move(snap));
    return is_new;
}

std::optional<ModelView> ModelRegistry::find_model(std::string_view id)
{
    std::shared_ptr<detail::RegistrySnapshot> snap = load_snapshot();
    auto it = snap->index.find(id);
    if (it == snap->index.end())
        return std::nullopt;
    return it->second;
}

/// @brief 把请求等级收敛到模型支持的范围：从请求等级向下找第一个支持的档位。
///        Off 恒「支持」（引擎不发 thinking 参数即关闭）；全部不支持 → Off。
ThinkingLevel clamp_thinking_level(ModelView const& model, ThinkingLevel level)
{
    int idx = static_cast<int>(level);
    while (idx >= 1 && !model.thinking_level_map[static_cast<std::size_t>(idx)].has_value())
        --idx;
    return static_cast<ThinkingLevel>(idx);
}

}  // namespace agent
