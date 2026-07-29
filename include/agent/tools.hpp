#pragma once

#include <cstddef>
#include <expected>
#include <functional>
#include <memory>
#include <meta>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <agent/result.hpp>
#include <nlohmann/json.hpp>

namespace agent {

/// @brief 注解值载体。字符串通过 define_static_string 转为 static-storage 指针后传入。
///        必须为 structural type 才能在 NTTP 中作为注解值。
struct DescArg
{
    const char* msg;
};

/// @brief 注解构造辅助。[[= Desc("文本")]] 语法在 P3394R4 中定义。
/// @param s  源字符串，在编译期转为 static-storage char* 存入 DescArg
consteval DescArg Desc(std::string_view s)
{
    return DescArg{ std::define_static_string(s) };
}

/// @brief 工具注册信息。包含名称、描述、参数 JSON Schema。
struct ToolInfo
{
    std::string             name;
    std::string             description;
    nlohmann::json          parameters;
};

/// @brief 内部实现。不视为 API，用户不应直接使用。
///
/// 包含 JSON Schema 生成、类型萃取、递归反序列化等核心实现。
namespace detail {

/// @brief JSON Schema 字段类型枚举
enum class FieldType
{
    String,
    Integer,
    Number,
    Boolean,
    Array,
    Object,
    Enum,
};

/// @brief 检测 T 是否是 Primary<Args...> 的实例化。
///        通过部分特化匹配模板模板参数实现。
template<typename T, template<typename...> class Primary>
struct is_specialization_impl : std::false_type {};

template<template<typename...> class Primary, typename... Args>
struct is_specialization_impl<Primary<Args...>, Primary> : std::true_type {};

template<typename T, template<typename...> class Primary>
concept is_specialization_c = is_specialization_impl<T, Primary>::value;

/// @brief 检测 T 是否是 std::array<T,N>。
///        通用版 is_specialization_c 不支持非类型模板参数（N），需单独特化。
template<typename T> struct is_std_array : std::false_type {};
template<typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

/// @brief 剥除 std::optional 包装层，保留裸类型。
template<typename T>
struct strip_optional { using type = T; };
template<typename T>
struct strip_optional<std::optional<T>> { using type = T; };
template<typename T>
using strip_optional_t = typename strip_optional<T>::type;

/// @brief C++ 类型 → JSON Schema 类型映射。
///        对 optional<T> 递归剥壳后映射内层类型。
template<typename T>
consteval FieldType type_to_field_type()
{
    if constexpr (std::is_same_v<T, bool>)                            return FieldType::Boolean;
    else if constexpr (std::is_same_v<T, std::string>)               return FieldType::String;
    else if constexpr (std::is_integral_v<T>)                         return FieldType::Integer;
    else if constexpr (std::is_floating_point_v<T>)                   return FieldType::Number;
    else if constexpr (std::is_enum_v<T>)                             return FieldType::Enum;
    else if constexpr (is_specialization_c<T, std::vector>)           return FieldType::Array;
    else if constexpr (is_std_array<T>::value)                        return FieldType::Array;
    else if constexpr (is_specialization_c<T, std::optional>)         return type_to_field_type<typename T::value_type>();
    else if constexpr (std::is_class_v<T>)                            return FieldType::Object;
    else                                                              static_assert(sizeof(T) == 0, "unsupported field type");
}

consteval std::string_view field_type_to_json(FieldType ft)
{
    switch (ft) {
        case FieldType::String:  return "string";
        case FieldType::Integer: return "integer";
        case FieldType::Number:  return "number";
        case FieldType::Boolean: return "boolean";
        case FieldType::Array:   return "array";
        case FieldType::Object:  return "object";
        case FieldType::Enum:    return "string";
    }
    return "unknown";
}

/// @brief 编译期获取 struct T 的所有非静态成员（反射数据）。
///        返回值是 std::span<const std::meta::info>，用于 template for 展开。
template<typename T>
constexpr auto reflect_members()
{
    return std::define_static_array(
        std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::unchecked()));
}

/// @brief 提取字段上的 [[= Desc("文本")]] 注解值。
///        extract<DescArg> 将 vector<info> 中的第一个注解转型为 DescArg 后取出 msg 指针。
consteval const char* field_annotation(std::meta::info m)
{
    auto annots = std::meta::annotations_of(m);
    if (annots.size() > 0)
        return std::meta::extract<DescArg>(annots.data()[0]).msg;
    return nullptr;
}

/// @brief 前向声明 build_schema<T>()，build_type_schema 中递归调用它处理 Object 类型。
template<typename T> nlohmann::json build_schema();

/// @brief 递归构建非 struct 类型的 JSON Schema。
///        处理 vector、std::array、enum、optional、标量。struct 类型转交 build_schema<T>()。
template<typename T>
nlohmann::json build_type_schema()
{
    using RawT = strip_optional_t<T>;
    constexpr auto t = type_to_field_type<T>();
    nlohmann::json s;

    if constexpr (t == FieldType::Object) {
        return build_schema<RawT>();
    }
    else if constexpr (t == FieldType::Array) {
        s["type"] = "array";
        using ElemT = typename RawT::value_type;
        s["items"] = build_type_schema<ElemT>();
        if constexpr (is_std_array<RawT>::value) {
            s["minItems"] = RawT().size();
            s["maxItems"] = RawT().size();
        }
    }
    else if constexpr (t == FieldType::Enum) {
        s["type"] = "string";
        s["enum"] = nlohmann::json::array();
        template for (constexpr auto e : std::define_static_array(
            std::meta::enumerators_of(^^RawT)))
        {
            s["enum"].push_back(std::string(std::meta::identifier_of(e)));
        }
    }
    else {
        s["type"] = field_type_to_json(t);
    }

    // optional 包装：type 从 "xxx" 变为 ["xxx", "null"]
    if constexpr (is_specialization_c<T, std::optional>) {
        s["type"] = nlohmann::json::array({s["type"].get<std::string>(), "null"});
    }

    return s;
}

/// @brief 为 struct T 生成 JSON Schema。
///        遍历所有非静态成员，对每个成员递归调用 build_schema / build_type_schema。
///        支持嵌套 Object、Array（vector/array）、Enum、optional、默认值、注解。
template<typename T>
nlohmann::json build_schema()
{
    nlohmann::json schema = nlohmann::json::object();
    schema["type"] = "object";
    schema["properties"] = nlohmann::json::object();
    schema["required"] = nlohmann::json::array();

    template for (constexpr auto m : reflect_members<T>())
    {
        using RawType = std::decay_t<decltype(std::declval<T>().[:m:])>;
        using InnerRaw = strip_optional_t<RawType>;
        constexpr auto schema_type = type_to_field_type<RawType>();
        auto field_name = std::meta::identifier_of(m);
        auto desc = field_annotation(m);

        nlohmann::json prop;

        if constexpr (schema_type == FieldType::Object) {
            static_assert(!std::is_same_v<InnerRaw, T>,
                "recursive type detected: struct cannot contain itself");
            prop = build_schema<InnerRaw>();
            if constexpr (is_specialization_c<RawType, std::optional>)
                prop["type"] = nlohmann::json::array({"object", "null"});
        }
        else if constexpr (schema_type == FieldType::Array) {
            prop["type"] = "array";
            using ElemT = typename InnerRaw::value_type;
            using InnerElem = strip_optional_t<ElemT>;
            constexpr auto elem_type = type_to_field_type<ElemT>();
            if constexpr (elem_type == FieldType::Object) {
                static_assert(!std::is_same_v<InnerElem, T>,
                    "recursive type detected: array element type cannot be the struct itself");
                prop["items"] = build_schema<InnerElem>();
                if constexpr (is_specialization_c<ElemT, std::optional>)
                    prop["items"]["type"] = nlohmann::json::array({"object", "null"});
            } else if constexpr (elem_type == FieldType::Enum && std::is_enum_v<InnerElem>) {
                prop["items"]["type"] = "string";
                prop["items"]["enum"] = nlohmann::json::array();
                template for (constexpr auto e : std::define_static_array(
                    std::meta::enumerators_of(^^InnerElem)))
                {
                    prop["items"]["enum"].push_back(
                        std::string(std::meta::identifier_of(e)));
                }
                if constexpr (is_specialization_c<ElemT, std::optional>)
                    prop["items"]["type"] = nlohmann::json::array({"string", "null"});
            } else {
                prop["items"] = build_type_schema<ElemT>();
            }
            if constexpr (is_std_array<InnerRaw>::value) {
                prop["minItems"] = InnerRaw().size();
                prop["maxItems"] = InnerRaw().size();
            }
            if constexpr (is_specialization_c<RawType, std::optional>)
                prop["type"] = nlohmann::json::array({"array", "null"});
        }
        else if constexpr (schema_type == FieldType::Enum && std::is_enum_v<RawType>) {
            prop["type"] = "string";
            prop["enum"] = nlohmann::json::array();
            template for (constexpr auto e : std::define_static_array(
                std::meta::enumerators_of(^^RawType)))
            {
                prop["enum"].push_back(std::string(std::meta::identifier_of(e)));
            }
        }
        else {
            if constexpr (is_specialization_c<RawType, std::optional>) {
                prop["type"] = nlohmann::json::array(
                    {field_type_to_json(schema_type), "null"});
            } else {
                prop["type"] = field_type_to_json(schema_type);
            }
        }

        if (desc != nullptr)
            prop["description"] = desc;

        if constexpr (std::meta::has_default_member_initializer(m)) {
            if constexpr (std::is_trivially_default_constructible_v<RawType>) {
                constexpr auto default_val = T().[:m:];
                if constexpr (std::is_same_v<RawType, bool>)
                    prop["default"] = default_val;
                else if constexpr (std::is_integral_v<RawType>)
                    prop["default"] = default_val;
                else if constexpr (std::is_floating_point_v<RawType>)
                    prop["default"] = default_val;
            } else if constexpr (std::is_same_v<RawType, std::string>) {
                T tmp{};
                prop["default"] = tmp.[:m:];
            }
        }

        schema["properties"][std::string(field_name)] = std::move(prop);

        if constexpr (!std::meta::has_default_member_initializer(m)
            && !is_specialization_c<RawType, std::optional>)
        {
            schema["required"].push_back(std::string(field_name));
        }
    }

    return schema;
}

/// @brief 返回 struct T 的 JSON Schema（函数内 static 缓存，线程安全）。
template<typename T>
inline const nlohmann::json& schema_of()
{
    static const auto schema = build_schema<T>();
    return schema;
}

/// @brief 递归验证 JSON 值是否符合 JSON Schema。
///        支持 type（含 ["type","null"]）、required、properties、items、enum、minItems/maxItems。
inline void validate_value(
    nlohmann::json const& value,
    nlohmann::json const& schema,
    std::string const& path,
    std::vector<std::string>& errors);

inline void validate_schema(
    nlohmann::json const& value,
    nlohmann::json const& schema,
    std::string const& path,
    std::vector<std::string>& errors)
{
    if (!schema.contains("type"))
        return;

    auto const& type_val = schema["type"];
    bool has_null = false;
    bool type_matched = false;

    if (type_val.is_string()) {
        std::string const& t = type_val.get_ref<std::string const&>();
        if (t == "null") {
            has_null = true;
            if (value.is_null()) return;
        }
        if (t == "object" && value.is_object()) type_matched = true;
        else if (t == "array" && value.is_array()) type_matched = true;
        else if (t == "string" && value.is_string()) type_matched = true;
        else if (t == "integer" && value.is_number_integer()) type_matched = true;
        else if (t == "number" && value.is_number()) type_matched = true;
        else if (t == "boolean" && value.is_boolean()) type_matched = true;
    } else if (type_val.is_array()) {
        for (auto const& t : type_val) {
            std::string const& ts = t.get_ref<std::string const&>();
            if (ts == "null") {
                has_null = true;
                if (value.is_null()) return;
            }
            if (ts == "object" && value.is_object()) type_matched = true;
            else if (ts == "array" && value.is_array()) type_matched = true;
            else if (ts == "string" && value.is_string()) type_matched = true;
            else if (ts == "integer" && value.is_number_integer()) type_matched = true;
            else if (ts == "number" && value.is_number()) type_matched = true;
            else if (ts == "boolean" && value.is_boolean()) type_matched = true;
        }
    }

    if (!type_matched && !has_null)
        errors.push_back(path + " type mismatch");

    if (value.is_object() && schema.contains("required")) {
        for (auto const& req : schema["required"]) {
            std::string const& key = req.get_ref<std::string const&>();
            if (!value.contains(key) || value[key].is_null())
                errors.push_back(path + "." + key + " is required");
        }
    }

    if (value.is_object() && schema.contains("properties")) {
        for (auto it = value.begin(); it != value.end(); ++it) {
            auto prop_it = schema["properties"].find(it.key());
            if (prop_it != schema["properties"].end()) {
                validate_value(it.value(), *prop_it,
                    path + "." + it.key(), errors);
            }
        }
    }

    if (value.is_array() && schema.contains("items")) {
        if (schema.contains("minItems") && value.size() < schema["minItems"].get<std::size_t>())
            errors.push_back(path + " too few items");
        if (schema.contains("maxItems") && value.size() > schema["maxItems"].get<std::size_t>())
            errors.push_back(path + " too many items");
        for (std::size_t i = 0; i < value.size(); ++i)
            validate_value(value[i], schema["items"],
                path + "[" + std::to_string(i) + "]", errors);
    }

    if (schema.contains("enum")) {
        bool found = false;
        for (auto const& e : schema["enum"])
            if (e == value) { found = true; break; }
        if (!found)
            errors.push_back(path + " not in enum");
    }
}

inline void validate_value(
    nlohmann::json const& value,
    nlohmann::json const& schema,
    std::string const& path,
    std::vector<std::string>& errors)
{
    validate_schema(value, schema, path, errors);
}

/// @brief 递归提取 JSON 值到 C++ 类型 T。
///        支持 vector、std::array、optional、enum、标量。
///        对 enum 用 template for + define_static_array 提取枚举名→值映射。
///        对 struct 类型转交 assign_from_json<T>()。
/// @tparam T 目标 C++ 类型
/// @param j  JSON 输入
/// @param path  当前路径（用于错误信息）
template<typename T>
    requires std::is_class_v<T>
Result<T> assign_from_json(nlohmann::json const& j);

template<typename T>
Result<T> extract_json(nlohmann::json const& j, std::string const& path, std::size_t depth = 0)
{
    if constexpr (is_specialization_c<T, std::optional>) {
        using InnerT = typename T::value_type;
        if (j.is_null())
            return T(std::nullopt);
        auto inner = extract_json<InnerT>(j, path, depth + 1);
        if (!inner) return std::unexpected(inner.error());
        return T(std::move(*inner));
    }
    else if constexpr (is_specialization_c<T, std::vector>) {
        using ElemT = typename T::value_type;
        if (!j.is_array())
            return std::unexpected(Error{Errc::InvalidArgs, path + " expects array"});
        std::vector<ElemT> result;
        for (std::size_t i = 0; i < j.size(); ++i) {
            auto elem = extract_json<ElemT>(j[i], path + "[" + std::to_string(i) + "]", depth + 1);
            if (!elem) return std::unexpected(elem.error());
            result.push_back(std::move(*elem));
        }
        return result;
    }
    else if constexpr (is_std_array<T>::value) {
        using ElemT = typename T::value_type;
        if (!j.is_array() || j.size() != T().size())
            return std::unexpected(Error{Errc::InvalidArgs,
                path + " expected " + std::to_string(T().size()) + " elements, got "
                + (j.is_array() ? std::to_string(j.size()) : std::string("non-array"))});
        T result{};
        for (std::size_t i = 0; i < j.size(); ++i) {
            auto elem = extract_json<ElemT>(j[i], path + "[" + std::to_string(i) + "]", depth + 1);
            if (!elem) return std::unexpected(elem.error());
            result[i] = std::move(*elem);
        }
        return result;
    }
    else if constexpr (std::is_enum_v<T>) {
        if (!j.is_string())
            return std::unexpected(Error{Errc::InvalidArgs, path + " expects string for enum"});
        std::string str = j.template get<std::string>();
        bool found = false;
        T val{};
        template for (constexpr auto e : std::define_static_array(
            std::meta::enumerators_of(^^T)))
        {
            if (std::meta::identifier_of(e) == str) {
                val = std::meta::extract<T>(e);
                found = true;
            }
        }
        if (!found)
            return std::unexpected(Error{Errc::InvalidArgs,
                path + " unknown enum value: " + str});
        return val;
    }
    else if constexpr (std::is_same_v<T, bool>) {
        if (!j.is_boolean())
            return std::unexpected(Error{Errc::InvalidArgs, path + " expects boolean"});
        return j.template get<bool>();
    }
    else if constexpr (std::is_integral_v<T>) {
        if (!j.is_number_integer())
            return std::unexpected(Error{Errc::InvalidArgs, path + " expects integer"});
        auto raw = j.template get<std::int64_t>();
        if (raw < std::numeric_limits<T>::min() || raw > std::numeric_limits<T>::max())
            return std::unexpected(Error{Errc::InvalidArgs, path + " out of range"});
        return static_cast<T>(raw);
    }
    else if constexpr (std::is_floating_point_v<T>) {
        if (!j.is_number())
            return std::unexpected(Error{Errc::InvalidArgs, path + " expects number"});
        return j.template get<T>();
    }
    else if constexpr (std::is_same_v<T, std::string>) {
        if (!j.is_string())
            return std::unexpected(Error{Errc::InvalidArgs, path + " expects string"});
        return j.template get<std::string>();
    }
    else if constexpr (std::is_class_v<T>) {
        return assign_from_json<T>(j);
    }
    else {
        static_assert(sizeof(T) == 0, "unsupported type in extract_json");
    }
}

/// @brief 使用反射将 JSON object 反序列化为 struct T。
///        遍历 T 的所有非静态成员，用 extract_json<T> 逐个提取。
///        缺 required 字段→Error，非 optional 字段遇 null→Error，缺 optional 字段→保持默认。
template<typename T>
    requires std::is_class_v<T>
Result<T> assign_from_json(nlohmann::json const& j)
{
    constexpr auto ctx = std::meta::access_context::unchecked();
    T obj{};

    template for (constexpr auto m : std::define_static_array(
        std::meta::nonstatic_data_members_of(^^T, ctx)))
    {
        using RawType = std::decay_t<decltype(std::declval<T>().[:m:])>;
        auto name = std::meta::identifier_of(m);
        std::string path = "." + std::string(name);

        auto it = j.find(name);
        if (it == j.end()) {
            if constexpr (!std::meta::has_default_member_initializer(m)
                && !is_specialization_c<RawType, std::optional>)
            {
                return std::unexpected(Error{Errc::InvalidArgs,
                    path + " is required"});
            }
            continue;
        }

        if (it->is_null() && !is_specialization_c<RawType, std::optional>)
            return std::unexpected(Error{Errc::InvalidArgs,
                path + " cannot be null (not optional)"});

        auto val = extract_json<RawType>(*it, path);
        if (!val) return std::unexpected(val.error());
        obj.[:m:] = std::move(*val);
    }

    return obj;
}

struct RegisteredTool
{
    ToolInfo                info;
    std::function<Result<std::string>(nlohmann::json)> fn;
};

/// @brief 编译期工具元数据（名称 + 注解）。
///        consteval 提取后在 Registrar 运行时直接读指针，避免调用 consteval-only 函数。
template<typename T>
struct ToolMeta
{
    const char* name;
    const char* description;
};

template<typename T>
consteval ToolMeta<T> reflect_tool_meta()
{
    constexpr auto id = std::meta::identifier_of(^^T);
    auto annots = std::meta::annotations_of(^^T);
    const char* desc = nullptr;
    if (annots.size() > 0)
        desc = std::meta::extract<DescArg>(annots.data()[0]).msg;
    return ToolMeta<T>{ std::define_static_string(id), desc };
}

/// @brief CRTP 工具基类。
///        继承者在各自 .cpp 中显式实例化触发静态注册。
///        inline static Registrar reg{} 的构造函数在 main() 前运行。
/// @tparam T 工具类型。须提供：
///           - params_type 嵌套 struct（字段定义 JSON 参数 Schema）
///           - static Result<std::string> invoke(Params const&)
template<typename T>
struct ToolBase
{
    struct Registrar
    {
        Registrar();
    };
    inline static Registrar reg{};
};

} // namespace detail

template<typename T>
inline const nlohmann::json& schema_of()
{
    return detail::schema_of<T>();
}

template<typename T>
    requires std::is_class_v<T>
Result<T> assign_from_json(nlohmann::json const& j)
{
    return detail::assign_from_json<T>(j);
}

/// @brief 工具注册中心。纯静态类。
///
/// 使用方式：
///   1. Tools::reg(ToolInfo{...}, fn) — 运行时注册
///   2. Tools::exec("name", args_json) — 执行工具
///   3. Tools::list() — 列出所有已注册工具
///   4. Tools::get("name") — 查询指定工具信息
///
/// 线程安全：内部使用 shared_mutex，读操作共享锁，写操作用独占锁。
class Tools
{
public:
    static Result<void> reg(ToolInfo info,
        std::function<Result<std::string>(nlohmann::json)> fn);

    static Result<std::string> exec(std::string_view name,
        nlohmann::json args);

    static std::vector<ToolInfo> list();

    static Result<ToolInfo> get(std::string_view name);

private:
    struct State {
        std::unordered_map<std::string,
            std::shared_ptr<detail::RegisteredTool>> registry;
        std::shared_mutex mtx;
    };
    static State& state();
};

namespace detail {

template<typename T>
ToolBase<T>::Registrar::Registrar()
{
    constexpr auto meta = reflect_tool_meta<T>();
    using Params = typename T::params_type;
    auto schema = schema_of<Params>();

    ToolInfo info{
        std::string(meta.name),
        meta.description ? std::string(meta.description) : std::string(),
        std::move(schema)
    };
    auto result = Tools::reg(std::move(info), [](nlohmann::json args) -> Result<std::string> {
        auto params = assign_from_json<Params>(args);
        if (!params) return std::unexpected(params.error());
        return T::invoke(std::move(*params));
    });
    (void)result;
}

} // namespace detail
} // namespace agent
