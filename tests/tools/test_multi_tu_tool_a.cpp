// 跨翻译单元静态注册验证（工具 A）。
// 工具定义与显式实例化只出现在本 .cpp，其他 TU 无任何符号引用——
// 链接后注册仍应生效（验证 inline static Registrar 的 ODR-use 语义）。
// 断言在 test_tools.cpp 的 "cross-TU explicit instantiation registers tools"。

#include <agent/tools/tools_reflection.hpp>

using agent::Desc;
using agent::Result;
using agent::ToolBase;

struct MultiTuAlphaInput
{
    [[= Desc("回显文本")]] std::string text;
};

struct [[= Desc("跨 TU 注册工具 A")]]
MultiTuAlpha : ToolBase<MultiTuAlpha>
{
    using params_type = MultiTuAlphaInput;
    static Result<std::string> invoke(MultiTuAlphaInput const& p)
    {
        return "alpha:" + p.text;
    }
};
template struct agent::ToolBase<MultiTuAlpha>;
