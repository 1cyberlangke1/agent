#pragma once

// 模型表三结构 + 全局注册表（L1 模型层）。
// 包含：BuiltinModel / RuntimeModel / ModelView、ModelRegistry（纯静态类）、
// clamp_thinking_level。实现见 src/model/model_registry.cpp。

#include <agent/llm/types.hpp>

#include <array>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace agent {

// ─────────────────────────────────────────────────────────────
// 模型表三结构（双轨：静态内置表 + 运行时注册表）
// ─────────────────────────────────────────────────────────────

namespace detail {

/// @brief 内置静态表模型（生成器产物，真 constexpr，零分配）。
///        数据源：models.dev/api.json（LLM 过滤后按配置的 provider 生成）。
///        由 scripts/update_models.py 生成到 include/agent/models/generated.hpp，
///        放 detail 命名空间，非 API，重新生成不通知。
struct BuiltinModel {
    std::string_view id;                    // "deepseek-chat"
    int context_window = 0;                 // 总上下文窗口
    int max_output_tokens = 0;              // 最大输出 token
    bool reasoning = false;                 // 支持思维链
    bool supports_image_input = false;      // 视觉能力
    /// 统一 ThinkingLevel → 厂商原生值映射。值唯一语义，无混用：
    ///   nullopt → 模型无思考能力（reasoning=false）；
    ///   "off"   → Off 档：关闭思考（引擎不发 thinking 参数）；
    ///   "on"    → toggle 型（支持思考但无强度细分）的启用档；
    ///   effort 值（"low".."max"）→ effort 型档位强度；
    ///   budget 值（"1024".."32768"）→ budget 型档位预算。
    /// 固定数组按下标访问（枚举按序 0..6），无 map 树。
    /// effort 型非 Off 档已预填「向下收敛最近有效值」——引擎取 map[level]
    /// 非 Off 恒有值，免判空。
    /// 值统一为字符串形态，指向静态字面量。
    std::array<std::optional<std::string_view>, thinking_level_count> thinking_level_map;
    /// thinking 值写入的请求字段名（Gemini 3 → "thinkingLevel"，2.5 → "thinkingBudget"）；
    /// 空 = 用引擎/ThinkingPolicy 的默认字段。
    std::string_view thinking_field;
    /// 单价（美元/百万 token）。0 = 未知或免费。
    double price_input = 0;
    double price_output = 0;
    double price_cache_read = 0;
    double price_cache_write = 0;
};

/// @brief 生成器产物的汇总表项：provider 名 → 模型子表。
struct BuiltinProviderTable {
    std::string_view provider;
    std::span<const BuiltinModel> models;
};

}  // namespace detail

/// @brief 运行时注册的模型（全 std::string 自持）。
///        语义同 detail::BuiltinModel，字符串动态，由 ModelRegistry 拥有生命周期。
struct RuntimeModel {
    std::string id;
    int context_window = 0;
    int max_output_tokens = 0;
    bool reasoning = false;
    bool supports_image_input = false;
    std::array<std::optional<std::string>, thinking_level_count> thinking_level_map;
    std::string thinking_field;
    double price_input = 0;          // 美元/百万 token；0 = 未知或免费
    double price_output = 0;
    double price_cache_read = 0;
    double price_cache_write = 0;
};

/// @brief 统一模型视图：引擎/调用方唯一消费类型。
///        string_view 指向底层存储（静态 → 全局字面量；动态 → ModelRegistry 的 deque 元素，
///        全局存储、地址稳定、进程生命周期），返回后长期有效。
///        从 BuiltinModel / RuntimeModel 均可构造（ModelRegistry 内部转换）。
struct ModelView {
    std::string_view id;
    int context_window = 0;
    int max_output_tokens = 0;
    bool reasoning = false;
    bool supports_image_input = false;
    std::array<std::optional<std::string_view>, thinking_level_count> thinking_level_map;
    std::string_view thinking_field;
    double price_input = 0;          // 美元/百万 token；0 = 未知或免费
    double price_output = 0;
    double price_cache_read = 0;
    double price_cache_write = 0;
};

// ─────────────────────────────────────────────────────────────
// ModelRegistry — 全局模型注册表（纯静态类，多线程安全）
// ─────────────────────────────────────────────────────────────

/// @brief 全局模型注册表。**纯静态类**：禁实例化，全 static 成员，进程生命周期。
///        独立于 API 调用层——只关心「有哪些模型、元数据是什么」，零协议概念。
///        内置表（生成器产物）+ 运行时注册 + 合并索引；动态覆盖静态。
///        线程安全：shared_mutex（register 唯一锁 / find、for_each 共享锁）+
///        call_once 惰性加载内置表。
class ModelRegistry {
public:
    ModelRegistry() = delete;

    /// @brief 运行时注册/覆盖模型（同 id 覆盖内置或已有动态项）。线程安全。
    ///        覆盖实现约束：deque 只追加新条目、旧条目永不改写/删除——
    ///        原地改写会使已发出的 ModelView 的 string_view 悬空。
    ///        index_ 改指新条目；for_each 输出以 index_ 当前指向为准（旧重复条目跳过）。
    ///        注册支持思考的模型：须填 reasoning=true 且按新语义填 thinking_level_map
    ///        （"off"/"on"/effort 值/budget 值）；thinking_level_map 全 nullopt = 无思考能力，
    ///        thinking 静默失效——非推理模型才应留空。
    /// @param model 模型数据（拷贝进注册表）
    /// @return true = 新增 id，false = 覆盖已有 id
    static bool register_model(RuntimeModel model);

    /// @brief 统一查找（动态覆盖静态）。O(1) 哈希索引。线程安全。
    /// @param id 模型 id
    /// @return 匹配的 ModelView；未找到返回 nullopt
    static std::optional<ModelView> find_model(std::string_view id);

    /// @brief 遍历全部可用模型（稳定顺序：静态表顺序 → 动态首次注册序，动态覆盖不重复）。
    ///        内部先持共享锁快照再回调 → 回调内可安全再调 register_model，无重入死锁。
    /// @param f 回调，接收 const ModelView&
    template<typename F>
    static void for_each_model(F&& f)
    {
        std::shared_lock lock(mutex_);
        init();
        std::vector<ModelView> snapshot;
        snapshot.reserve(index_.size());
        std::unordered_set<std::string_view> emitted;
        for (std::string_view id : static_order_) {
            if (dynamic_ids_.contains(id)) continue;   // 被动态覆盖 → 跳过静态项
            auto it = index_.find(id);
            if (it != index_.end()) {
                snapshot.push_back(it->second);
                emitted.insert(id);
            }
        }
        for (RuntimeModel const& rtm : runtime_models_) {
            // 首次注册位置输出，内容取 index_ 当前指向（最新注册）
            if (emitted.insert(rtm.id).second)
                snapshot.push_back(index_.at(rtm.id));
        }
        lock.unlock();
        for (ModelView const& mv : snapshot)
            f(mv);
    }

private:
    static void init();                                   // call_once 惰性加载内置表
    static ModelView to_view(detail::BuiltinModel const& m);
    static ModelView to_view(RuntimeModel const& m);

    inline static std::once_flag init_flag_;
    inline static std::shared_mutex mutex_;               // 多线程安全
    inline static std::deque<RuntimeModel> runtime_models_;   // 动态表（地址稳定，只追加）
    inline static std::unordered_map<std::string_view, ModelView> index_;  // 合并索引
    inline static std::unordered_set<std::string_view> dynamic_ids_;       // 动态覆盖的 id
    inline static std::vector<std::string_view> static_order_;             // 静态表遍历顺序
};

/// @brief 把用户请求的思考等级收敛到模型支持的范围内（从低向高找最近支持档）。
/// @param model  目标模型
/// @param level  用户请求的等级
/// @return 模型支持的最高等级（若全部不支持则返回 Off）
ThinkingLevel clamp_thinking_level(ModelView const& model, ThinkingLevel level);

}  // namespace agent
