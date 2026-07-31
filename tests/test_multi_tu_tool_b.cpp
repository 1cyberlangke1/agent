// 跨翻译单元静态注册验证（工具 B）。与 test_multi_tu_tool_a.cpp 配对，
// 验证多个 TU 各自显式实例化不同工具时全部完成注册。

#include <agent/tools_reflection.hpp>

using agent::Desc;
using agent::Result;
using agent::ToolBase;

struct MultiTuBetaInput
{
    [[= Desc("回显文本")]] std::string text;
};

struct [[= Desc("跨 TU 注册工具 B")]]
MultiTuBeta : ToolBase<MultiTuBeta>
{
    using params_type = MultiTuBetaInput;
    static Result<std::string> invoke(MultiTuBetaInput const& p)
    {
        return "beta:" + p.text;
    }
};
template struct agent::ToolBase<MultiTuBeta>;
