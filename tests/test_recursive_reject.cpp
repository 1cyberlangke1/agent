// 测试递归类型在 schema_of<T>() 时编译期拒绝
// 预期：static_assert 在 build_schema<RecursiveBad>() 中触发，编译失败
// 编译此文件应失败

#include <agent/tools_reflection.hpp>

struct RecursiveBad
{
    std::string                    name;
    std::vector<RecursiveBad>      children;
};

auto force_instantiation = agent::schema_of<RecursiveBad>();
