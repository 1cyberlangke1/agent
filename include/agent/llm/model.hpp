#pragma once

// 模型表三结构 + 全局注册表（L1 模型层）。
// 包含：BuiltinModel / RuntimeModel / ModelView、ModelRegistry（纯静态类）、
// clamp_thinking_level。实现见 src/model/model_registry.cpp。

#include <agent/llm/types.hpp>

#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
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
    std::string_view id;                    ///< "deepseek-chat"
    int context_window = 0;                 ///< 总上下文窗口
    int max_output_tokens = 0;              ///< 最大输出 token
    bool reasoning = false;                 ///< 支持思维链
    bool supports_image_input = false;      ///< 视觉能力
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
    std::string id;                        ///< 模型标识（如 "deepseek-chat"）
    int context_window = 0;                ///< 上下文窗口大小（token）
    int max_output_tokens = 0;             ///< 单次最大输出 token
    bool reasoning = false;                ///< 是否支持思考模式
    bool supports_image_input = false;     ///< 是否支持图片输入（多模态）
    std::array<std::optional<std::string>, thinking_level_count> thinking_level_map;  ///< 思考等级 → 厂商字段值映射
    std::string thinking_field;            ///< 思考输出字段名（如 reasoning_content / thoughts）
    double price_input = 0;                ///< 美元/百万 token；0 = 未知或免费
    double price_output = 0;               ///< 输出价（美元/百万 token；0 = 未知或免费）
    double price_cache_read = 0;           ///< 缓存读取价
    double price_cache_write = 0;          ///< 缓存写入价
};

/// @brief 统一模型视图：引擎/调用方唯一消费类型。
///        string_view 指向底层存储（静态 → 全局字面量；动态 → ModelRegistry 的 deque 元素，
///        全局存储、地址稳定、进程生命周期），返回后长期有效。
///        从 BuiltinModel / RuntimeModel 均可构造（ModelRegistry 内部转换）。
struct ModelView {
    std::string_view id;                   ///< 模型标识（string_view，指向全局存储长期有效）
    int context_window = 0;                ///< 上下文窗口大小（token）
    int max_output_tokens = 0;             ///< 单次最大输出 token
    bool reasoning = false;                ///< 是否支持思考模式
    bool supports_image_input = false;     ///< 是否支持图片输入（多模态）
    std::array<std::optional<std::string_view>, thinking_level_count> thinking_level_map;  ///< 思考等级 → 厂商字段值映射
    std::string_view thinking_field;       ///< 思考输出字段名
    double price_input = 0;                ///< 美元/百万 token；0 = 未知或免费
    double price_output = 0;               ///< 输出价（美元/百万 token；0 = 未知或免费）
    double price_cache_read = 0;           ///< 缓存读取价
    double price_cache_write = 0;          ///< 缓存写入价
};

namespace detail {

/// @brief 注册表不可变快照（COW 写一次读多次）：读路径只 atomic load 这个指针，零锁。
///        快照的 ModelView 指向静态字面量 / runtime_models_ deque（地址稳定、只追加），
///        跨快照仍有效；register 重建新快照原子替换，旧快照由 shared_ptr 保活。
struct RegistrySnapshot {
    std::unordered_map<std::string_view, ModelView> index;        ///< id → 当前 ModelView
    std::unordered_set<std::string_view> dynamic_ids;             ///< 被动态覆盖/注册的 id
    std::vector<std::string_view> static_order;                   ///< 内置表 id 顺序
    std::vector<std::string_view> dynamic_order;                  ///< 运行时注册 id 顺序（插入序）
};

}  // namespace detail

// ─────────────────────────────────────────────────────────────
// ModelRegistry — 全局模型注册表（纯静态类，多线程安全）
// ─────────────────────────────────────────────────────────────

/// @brief 全局模型注册表。**纯静态类**：禁实例化，全 static 成员，进程生命周期。
///        独立于 API 调用层——只关心「有哪些模型、元数据是什么」，零协议概念。
///        内置表（生成器产物）+ 运行时注册 + 合并索引；动态覆盖静态。
///        线程安全：**COW 不可变快照**——写路径（register）持 write_mutex_ 重建快照
///        原子替换；读路径（find/for_each）只 atomic load 快照，**零锁**（单线程热路径
///        零开销）。旧快照由 shared_ptr 保活，并发读看到一致（可能略旧）视图。
class ModelRegistry {
public:
    ModelRegistry() = delete;

    /// @brief 运行时注册/覆盖模型（同 id 覆盖内置或已有动态项）。线程安全。
    ///        覆盖实现约束：deque 只追加新条目、旧条目永不改写/删除——
    ///        原地改写会使已发出的 ModelView 的 string_view 悬空。
    ///        新快照把 id 指向最新条目；旧快照引用旧条目仍有效（地址稳定）。
    ///        注册支持思考的模型：须填 reasoning=true 且按新语义填 thinking_level_map
    ///        （"off"/"on"/effort 值/budget 值）；thinking_level_map 全 nullopt = 无思考能力，
    ///        thinking 静默失效——非推理模型才应留空。
    /// @param model 模型数据（拷贝进注册表）
    /// @return true = 新增 id，false = 覆盖已有 id
    static bool register_model(RuntimeModel model);

    /// @brief 统一查找（动态覆盖静态）。O(1) 哈希索引。线程安全（锁自由读）。
    /// @param id 模型 id
    /// @return 匹配的 ModelView；未找到返回 nullopt
    static std::optional<ModelView> find_model(std::string_view id);

    /// @brief 遍历全部可用模型（稳定顺序：静态表顺序 → 动态首次注册序，动态覆盖不重复）。
    ///        读锁自由（COW 快照）；回调内可安全再调 register_model（快照不可变，无重入问题）。
    /// @param f 回调，接收 const ModelView&
    template<typename F>
    static void for_each_model(F&& f)
    {
        std::shared_ptr<detail::RegistrySnapshot> snap = load_snapshot();
        for (std::string_view id : snap->static_order) {
            if (snap->dynamic_ids.contains(id))
                continue;   ///< 被动态覆盖 → 在动态序里输出新视图
            auto it = snap->index.find(id);
            if (it != snap->index.end())
                f(it->second);
        }
        for (std::string_view id : snap->dynamic_order)
            f(snap->index.at(id));
    }

private:
    /// init（懒加载内置表）与 register 共用：假定已持 write_mutex_，幂等。
    static void init();
    static ModelView to_view(detail::BuiltinModel const& m);
    static ModelView to_view(RuntimeModel const& m);
    /// 取当前快照；首次调用触发 init（持写锁建初始快照）。读路径入口，无锁。
    static std::shared_ptr<detail::RegistrySnapshot> load_snapshot();
    inline static std::mutex write_mutex_;                              ///< 写锁（register/init，罕见）
    inline static std::deque<RuntimeModel> runtime_models_;             ///< 地址稳定、只追加
    inline static std::atomic<std::shared_ptr<detail::RegistrySnapshot>> snapshot_;   ///< 不可变快照
};

/// @brief 把用户请求的思考等级收敛到模型支持的范围内（从低向高找最近支持档）。
/// @param model  目标模型
/// @param level  用户请求的等级
/// @return 模型支持的最高等级（若全部不支持则返回 Off）
ThinkingLevel clamp_thinking_level(ModelView const& model, ThinkingLevel level);

}  // namespace agent
