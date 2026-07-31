#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <meta>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <agent/tools.hpp>

namespace agent {

/// @brief 注解构造辅助。[[= Desc("文本")]] 语法在 P3394R4 中定义。
/// @param s  源字符串，在编译期转为 static-storage char* 存入 DescArg
consteval DescArg Desc(std::string_view s)
{
    return DescArg{ std::define_static_string(s) };
}

namespace detail {

/// @brief JSON Schema 字段类型枚举。
///        三家 provider（OpenAI/Anthropic/Gemini）通用支持的类型集合。
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

/// @brief 剥除一层 std::optional 包装，保留内层类型。
template<typename T>
struct strip_optional { using type = T; };
template<typename T>
struct strip_optional<std::optional<T>> { using type = T; };
template<typename T>
using strip_optional_t = typename strip_optional<T>::type;

/// @brief 递归剥除全部 std::optional 包装层（处理 optional<optional<T>> 等罕见嵌套）。
template<typename T>
struct strip_optional_all { using type = T; };
template<typename T>
struct strip_optional_all<std::optional<T>> {
    using type = typename strip_optional_all<T>::type;
};
template<typename T>
using strip_optional_all_t = typename strip_optional_all<T>::type;

/// @brief C++ 类型 → JSON Schema 类型映射。
///        对 optional<T> 递归剥壳后映射内层类型。
template<typename T>
consteval FieldType type_to_field_type()
{
    if constexpr (std::is_same_v<T, bool>)                            return FieldType::Boolean;
    else if constexpr (std::is_same_v<T, std::string>)                return FieldType::String;
    else if constexpr (std::is_integral_v<T>)                         return FieldType::Integer;
    else if constexpr (std::is_floating_point_v<T>)                   return FieldType::Number;
    else if constexpr (std::is_enum_v<T>)                             return FieldType::Enum;
    else if constexpr (is_specialization_c<T, std::vector>)           return FieldType::Array;
    else if constexpr (is_std_array<T>::value)                        return FieldType::Array;
    else if constexpr (is_specialization_c<T, std::optional>)         return type_to_field_type<typename T::value_type>();
    else if constexpr (std::is_class_v<T>)                            return FieldType::Object;
    else                                                              static_assert(sizeof(T) == 0,
        "type not supported as tool parameter: use bool, std::string, integral, "
        "floating point, enum, vector<T>, array<T,N>, optional<T>, or a struct");
}

/// @brief field_type 转字符串
///        返回 std::string_view。
consteval std::string_view field_type_to_json(FieldType field_type)
{
    switch (field_type) {
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
///        返回 std::span<const std::meta::info>，用于 template for 展开。
template<typename T>
constexpr auto reflect_members()
{
    return std::define_static_array(
        std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::unchecked()));
}

/// @brief 在实体（字段或类型）的注解列表里按类型查找 DescArg，返回其中的描述文本。
///
/// 遍历所有注解并按类型过滤，而不是假设第一个注解就是 Desc——
/// 同一声明可以携带多个注解（[[= Desc("x")]] [[= Other{}]]），
/// 盲取 annots[0] 再 extract<DescArg> 会在注解顺序不同时编译失败。
///
/// 注意：type_of(注解) 返回 const 修饰的类型（gcc 16.1 实测为 const DescArg），
/// 必须 remove_cv 后再与 ^^DescArg 比较，否则永远不相等。
///
/// @return 描述文本指针（static storage）；无 Desc 注解时返回 nullptr
consteval const char* find_desc(std::meta::info entity)
{
    auto annotations = std::meta::annotations_of(entity);
    for (std::size_t i = 0; i < annotations.size(); ++i) {
        auto a = annotations.data()[i];
        if (std::meta::remove_cv(std::meta::dealias(std::meta::type_of(a))) == ^^DescArg)
            return std::meta::extract<DescArg>(a).msg;
    }
    return nullptr;
}

/// @brief 检测字段类型是否（穿过 optional / vector / array 包装后）引用外层结构体本身。
///
/// 该情况会导致 schema 构建无限模板递归，用 static_assert 编译期拒绝并给出可读错误。
/// 间接递归（A 含 B、B 含 A）无法在此检测，会以模板实例化深度超限的形式报错。
template<typename Outer, typename Field>
consteval bool directly_recursive()
{
    if constexpr (std::is_same_v<Field, Outer>) {
        return true;
    } else if constexpr (is_specialization_c<Field, std::optional>) {
        return directly_recursive<Outer, typename Field::value_type>();
    } else if constexpr (is_specialization_c<Field, std::vector> || is_std_array<Field>::value) {
        return directly_recursive<Outer, typename Field::value_type>();
    } else {
        return false;
    }
}

/// @brief 枚举值 → 枚举名（static storage 字符串）。用于把字段默认值写进 schema。
/// @return 匹配的枚举名指针；value 不是任何具名枚举值时返回 nullptr
template<typename E>
consteval const char* enum_default_name(E value)
{
    auto enumerators = std::meta::enumerators_of(^^E);
    for (std::size_t i = 0; i < enumerators.size(); ++i) {
        auto e = enumerators.data()[i];
        if (std::meta::extract<E>(e) == value)
            return std::define_static_string(std::meta::identifier_of(e));
    }
    return nullptr;
}

/// @brief 前向声明 build_schema<T>()，build_type_schema 递归调用它处理 Object。
template<typename T> nlohmann::json build_schema();

/// @brief 递归构建任意受支持类型的 JSON Schema 节点。
///
/// 处理 struct（转交 build_schema）、vector、std::array、enum、optional、标量。
/// optional<T> 统一在出口处把 type 包装为 ["原类型","null"]——
/// 对 Object 同样生效（optional<Struct> → type: ["object","null"]），
/// 深层容器嵌套（vector<vector<optional<Struct>>>）中的 null 不会丢失。
template<typename T>
nlohmann::json build_type_schema()
{
    // 全剥 optional：optional<optional<X>> 只包一层 null，语义与 JSON 一致
    using BareT = strip_optional_all_t<T>;
    constexpr FieldType field_type = type_to_field_type<T>();
    nlohmann::json schema;

    if constexpr (field_type == FieldType::Object) {
        schema = build_schema<BareT>();
    }
    else if constexpr (field_type == FieldType::Array) {
        schema["type"] = "array";
        schema["items"] = build_type_schema<typename BareT::value_type>();
        if constexpr (is_std_array<BareT>::value) {
            schema["minItems"] = std::tuple_size_v<BareT>;
            schema["maxItems"] = std::tuple_size_v<BareT>;
        }
    }
    else if constexpr (field_type == FieldType::Enum) {
        schema["type"] = "string";
        schema["enum"] = nlohmann::json::array();
        template for (constexpr auto e : std::define_static_array(
            std::meta::enumerators_of(^^BareT)))
        {
            schema["enum"].push_back(std::string(std::meta::identifier_of(e)));
        }
    }
    else {
        schema["type"] = field_type_to_json(field_type);
    }

    // optional 包装：type 从 "xxx" 变为 ["xxx", "null"]
    if constexpr (is_specialization_c<T, std::optional>) {
        schema["type"] = nlohmann::json::array(
            {schema["type"].template get<std::string>(), "null"});
    }

    return schema;
}

/// @brief 为 struct T 生成 JSON Schema（type/properties/required）。
///
/// 遍历所有非静态成员，每个字段的类型节点交给 build_type_schema 递归构建，
/// 本函数只补充字段级信息：description（注解）、default（默认值）、required 归属。
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
        static_assert(!directly_recursive<T, RawType>(),
            "recursive type detected: struct cannot contain itself "
            "(e.g. struct Node { std::vector<Node> children; })");

        auto field_name = std::meta::identifier_of(m);
        nlohmann::json prop = build_type_schema<RawType>();

        if constexpr (find_desc(m) != nullptr)
            prop["description"] = find_desc(m);

        if constexpr (std::meta::has_default_member_initializer(m)) {
            if constexpr (std::is_trivially_default_constructible_v<RawType>) {
                // T() 在常量求值中构造（string 等成员走 transient 分配），
                // 提取的字段值本身必须是编译期常量（标量/枚举满足）
                constexpr auto default_value = T().[:m:];
                if constexpr (std::is_same_v<RawType, bool>)
                    prop["default"] = default_value;
                else if constexpr (std::is_integral_v<RawType>)
                    prop["default"] = default_value;
                else if constexpr (std::is_floating_point_v<RawType>)
                    prop["default"] = default_value;
                else if constexpr (std::is_enum_v<RawType>) {
                    // 枚举默认值以枚举名字符串写入，与 enum 约束的取值一致
                    if constexpr (enum_default_name<RawType>(default_value) != nullptr)
                        prop["default"] = enum_default_name<RawType>(default_value);
                }
            } else if constexpr (std::is_same_v<RawType, std::string>) {
                // string 默认值非编译期常量（堆分配），运行时构造一次提取
                T defaults{};
                prop["default"] = defaults.[:m:];
            }
        }

        schema["properties"][std::string(field_name)] = std::move(prop);

        // 无默认值且非 optional 的字段才是 required
        if constexpr (!std::meta::has_default_member_initializer(m)
            && !is_specialization_c<RawType, std::optional>)
        {
            schema["required"].push_back(std::string(field_name));
        }
    }

    return schema;
}

/// @brief 返回 struct T 的 JSON Schema（函数内 static 缓存，线程安全，只构建一次）。
template<typename T>
inline nlohmann::json const& schema_of()
{
    static nlohmann::json const schema = build_schema<T>();
    return schema;
}

/// @brief 前向声明：struct 反序列化，extract_json 的 class 分支递归调用。
template<typename T>
    requires std::is_class_v<T>
Result<T> assign_from_json(nlohmann::json const& j);

/// @brief 递归提取 JSON 值到 C++ 类型 T。零异常：所有类型先预检后取值。
///
/// 支持 optional、vector、std::array、enum、bool、整数（含范围检查）、浮点、
/// string；struct 类型转交 assign_from_json<T>()。
///
/// 整数字段：
/// - 有符号/无符号值分别按原始位宽取出，std::in_range 做混合符号安全的范围检查
///   （避免 int64/uint64 直接比较时的符号提升陷阱）
/// - 数值为整数的浮点（LLM 常发 3.0）在 ±2^53 精确范围内接受
///
/// @param j    JSON 输入
/// @param path 当前路径（用于错误信息定位）
template<typename T>
Result<T> extract_json(nlohmann::json const& j, std::string const& path)
{
    if constexpr (is_specialization_c<T, std::optional>) {
        using InnerT = typename T::value_type;
        if (j.is_null())
            return T(std::nullopt);
        Result<InnerT> inner = extract_json<InnerT>(j, path);
        if (!inner)
            return std::unexpected(inner.error());
        return T(std::move(*inner));
    }
    else if constexpr (is_specialization_c<T, std::vector>) {
        using ElemT = typename T::value_type;
        if (!j.is_array())
            return std::unexpected(Error{Errc::InvalidArgs, path + " expects array"});
        T result;
        result.reserve(j.size());
        for (std::size_t i = 0; i < j.size(); ++i) {
            Result<ElemT> elem = extract_json<ElemT>(
                j[i], path + "[" + std::to_string(i) + "]");
            if (!elem)
                return std::unexpected(elem.error());
            result.push_back(std::move(*elem));
        }
        return result;
    }
    else if constexpr (is_std_array<T>::value) {
        using ElemT = typename T::value_type;
        constexpr std::size_t expected_size = std::tuple_size_v<T>;
        if (!j.is_array() || j.size() != expected_size)
            return std::unexpected(Error{Errc::InvalidArgs,
                path + " expected " + std::to_string(expected_size) + " elements, got "
                + (j.is_array() ? std::to_string(j.size()) : std::string("non-array"))});
        T result{};
        for (std::size_t i = 0; i < expected_size; ++i) {
            Result<ElemT> elem = extract_json<ElemT>(
                j[i], path + "[" + std::to_string(i) + "]");
            if (!elem)
                return std::unexpected(elem.error());
            result[i] = std::move(*elem);
        }
        return result;
    }
    else if constexpr (std::is_enum_v<T>) {
        if (!j.is_string())
            return std::unexpected(Error{Errc::InvalidArgs,
                path + " expects string for enum"});
        std::string const& str = j.template get_ref<std::string const&>();
        bool found = false;
        T value{};
        template for (constexpr auto e : std::define_static_array(
            std::meta::enumerators_of(^^T)))
        {
            if (std::meta::identifier_of(e) == str) {
                value = std::meta::extract<T>(e);
                found = true;
            }
        }
        if (!found)
            return std::unexpected(Error{Errc::InvalidArgs,
                path + " unknown enum value: " + str});
        return value;
    }
    else if constexpr (std::is_same_v<T, bool>) {
        if (!j.is_boolean())
            return std::unexpected(Error{Errc::InvalidArgs, path + " expects boolean"});
        return j.template get<bool>();
    }
    else if constexpr (std::is_integral_v<T>) {
        if (j.is_number_unsigned()) {
            std::uint64_t raw = j.template get<std::uint64_t>();
            if (!std::in_range<T>(raw))
                return std::unexpected(Error{Errc::InvalidArgs, path + " out of range"});
            return static_cast<T>(raw);
        }
        if (j.is_number_integer()) {
            std::int64_t raw = j.template get<std::int64_t>();
            if (!std::in_range<T>(raw))
                return std::unexpected(Error{Errc::InvalidArgs, path + " out of range"});
            return static_cast<T>(raw);
        }
        if (j.is_number_float()) {
            double value = j.template get<double>();
            if (std::trunc(value) == value && std::abs(value) <= 9007199254740992.0) {
                std::int64_t raw = static_cast<std::int64_t>(value);
                if (!std::in_range<T>(raw))
                    return std::unexpected(Error{Errc::InvalidArgs, path + " out of range"});
                return static_cast<T>(raw);
            }
        }
        return std::unexpected(Error{Errc::InvalidArgs, path + " expects integer"});
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

/// @brief 使用反射将 JSON object 反序列化为 struct T。零异常。
///
/// 遍历 T 的所有非静态成员，用 extract_json 逐个提取：
/// - root 非 object → Error（防误导性的 "xxx is required" 级联报错）
/// - 缺 required 字段（无默认值且非 optional）→ Error
/// - 非 optional 字段值为 null → Error
/// - 缺 optional / 有默认值字段 → 保持默认
/// - schema 未声明的多余字段 → 忽略（宽容输入）
template<typename T>
    requires std::is_class_v<T>
Result<T> assign_from_json(nlohmann::json const& j)
{
    if (!j.is_object())
        return std::unexpected(Error{Errc::InvalidArgs,
            "arguments must be a JSON object"});

    T obj{};

    template for (constexpr auto m : reflect_members<T>())
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

        Result<RawType> value = extract_json<RawType>(*it, path);
        if (!value)
            return std::unexpected(value.error());
        obj.[:m:] = std::move(*value);
    }

    return obj;
}

/// @brief 编译期工具元数据。consteval 提取后运行时直接读 static-storage 指针。
struct ToolMeta
{
    const char* name;
    const char* description;
};

/// @brief 从工具类 T 的反射提取名称（struct 标识符）和描述（struct 头的 Desc 注解）。
template<typename T>
consteval ToolMeta reflect_tool_meta()
{
    return ToolMeta{
        std::define_static_string(std::meta::identifier_of(^^T)),
        find_desc(^^T),
    };
}

} // namespace detail

/// @brief 返回 struct T 反射生成的 JSON Schema（缓存，线程安全）。
template<typename T>
inline nlohmann::json const& schema_of()
{
    return detail::schema_of<T>();
}

/// @brief JSON → struct 反序列化（反射驱动，零异常，全量校验）。
template<typename T>
    requires std::is_class_v<T>
Result<T> assign_from_json(nlohmann::json const& j)
{
    return detail::assign_from_json<T>(j);
}

/// @brief CRTP 自注册工具基类。
///
/// 继承者须提供：
///   - using params_type = 参数结构体（字段可带 [[= Desc("...")]] 注解）
///   - static Result<std::string> invoke(params_type const&)
/// 工具名取 struct 标识符，描述取 struct 头的 [[= Desc("...")]] 注解，
/// 参数 schema 从 params_type 反射自动生成——三者都不需要手写。
///
/// 在定义工具的 .cpp 中显式实例化触发静态注册（缺了链接器会丢弃注册代码）：
///   template struct agent::ToolBase<MyTool>;
/// inline static Registrar reg{} 的构造函数在 main() 前运行完成注册。
template<typename T>
struct ToolBase
{
    struct Registrar
    {
        Registrar();
    };
    inline static Registrar reg{};
};

template<typename T>
ToolBase<T>::Registrar::Registrar()
{
    using Params = typename T::params_type;
    constexpr detail::ToolMeta meta = detail::reflect_tool_meta<T>();

    ToolInfo info{
        std::string(meta.name),
        meta.description != nullptr ? std::string(meta.description) : std::string(),
        detail::schema_of<Params>()
    };

    // ArgsCheck::Tool：assign_from_json 已全量校验，跳过 exec 的 schema 预校验
    Result<void> registered = Tools::reg(std::move(info),
        [](nlohmann::json args) -> Result<std::string> {
            Result<Params> params = detail::assign_from_json<Params>(args);
            if (!params)
                return std::unexpected(params.error());
            return T::invoke(std::move(*params));
        },
        ArgsCheck::Tool);

    // 静态初始化阶段无处返回错误，stderr 警告是唯一可靠通道
    // （重复注册几乎必然是程序员错误，宜早暴露）
    if (!registered)
        std::fprintf(stderr, "[agent] tool '%s' registration failed: %s\n",
            meta.name, registered.error().message.c_str());
}

} // namespace agent
