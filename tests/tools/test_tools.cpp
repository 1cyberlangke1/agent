#include <cstdint>

#include <agent/core/result.hpp>
#include <agent/tools/tools_reflection.hpp>
#include <doctest/doctest.h>

using agent::ArgsCheck;
using agent::Desc;
using agent::Errc;
using agent::Error;
using agent::Result;
using agent::ToolInfo;
using agent::Tools;

// ================================================================== //
// 测试用的反射结构体：覆盖 PLAN.md 定义的所有类型组合
// ================================================================== //

enum class Color { Red, Green, Blue };
enum class Unit { Celsius, Fahrenheit };

// ── 1. 原始标量 ── //
struct PrimitiveTypes
{
    [[= Desc("名字")]]           std::string name;
    [[= Desc("数量")]]           int         count;
    [[= Desc("比率")]]           double      ratio;
    [[= Desc("是否启用")]]       bool        enabled;
    [[= Desc("描述")]]           std::string desc = "默认描述";
    [[= Desc("步进")]]           int         step = 10;
};

// ── 2. 枚举 ── //
struct WithEnum
{
    [[= Desc("颜色选择")]]       Color       color;
    [[= Desc("温度单位")]]       Unit        unit = Unit::Celsius;
};

// ── 3. Optional 全系列 ── //
struct WithOptional
{
    [[= Desc("名称")]]                      std::string             name;
    [[= Desc("年龄")]]                      std::optional<int>      age;
    [[= Desc("是否激活")]]                  std::optional<bool>     active;
    [[= Desc("喜欢的颜色")]]                std::optional<Color>    favorite_color;
    [[= Desc("备注")]]                      std::optional<std::string> comment;
};

// ── 4. Vector ── //
struct TaskItem
{
    [[= Desc("任务标题")]]                  std::string             title;
    [[= Desc("优先级")]]                    int                     priority;
};

struct WithVector
{
    [[= Desc("项目名称")]]                  std::string             project_name;
    [[= Desc("任务列表")]]                  std::vector<TaskItem>   tasks;
    [[= Desc("分数列表")]]                  std::vector<int>        scores;
    [[= Desc("标签")]]                      std::vector<std::string> tags;
};

// ── 5. std::array ── //
struct Coordinate
{
    [[= Desc("X 坐标")]]                    double                  x;
    [[= Desc("Y 坐标")]]                    double                  y;
};

struct WithArray
{
    [[= Desc("名称")]]                      std::string             name;
    [[= Desc("RGB")]]                       std::array<int, 3>      rgb;
    [[= Desc("范围")]]                      std::array<double, 2>   range;
    [[= Desc("四个角")]]                    std::array<Coordinate, 4> corners;
};

// ── 6. vector<optional<T>> ── //
struct WithVecOptional
{
    [[= Desc("可选标签")]]                  std::vector<std::optional<std::string>> tags;
    [[= Desc("可选分数")]]                  std::vector<std::optional<int>>         scores;
};

// ── 7. struct 包含 struct ── //
struct NestedStruct
{
    [[= Desc("内部坐标")]]                  Coordinate              coord;
    [[= Desc("内部标签")]]                  std::string             label;
};

// ── 8. 5 层深嵌套：Organization → vector<Department>
//              → array<Team,3> → vector<Project>
//              → array<Milestone,4> → optional<string> ── //

/// 第 4 层：里程碑
struct Milestone
{
    [[= Desc("里程碑名称")]]                std::string             name;
    [[= Desc("截止日期")]]                  std::optional<std::string> due_date;
};

/// 第 3 层：项目，含固定数量里程碑
struct Project
{
    [[= Desc("项目名称")]]                  std::string                 name;
    [[= Desc("里程碑列表")]]                std::array<Milestone, 4>    milestones;
    [[= Desc("项目描述")]]                  std::optional<std::string>  description;
};

/// 第 2 层：团队，含项目列表
struct Team
{
    [[= Desc("团队名称")]]                  std::string                 name;
    [[= Desc("项目列表")]]                  std::vector<Project>        projects;
    [[= Desc("负责人")]]                    std::optional<std::string>  lead;
};

/// 第 1 层：部门，含固定数量团队
struct Department
{
    [[= Desc("部门名称")]]                  std::string                 name;
    [[= Desc("团队")]]                      std::array<Team, 3>         teams;
    [[= Desc("预算编号")]]                  std::optional<std::string>  budget_code;
};

/// 第 0 层：组织，vector<Department> 结尾，共 5 层反射嵌套
struct OrganizationTree
{
    [[= Desc("组织名称")]]                  std::string                 name;
    [[= Desc("部门列表")]]                  std::vector<Department>     departments;
    [[= Desc("备注")]]                      std::optional<std::string>  description;
};

// ── 9. 递归类型（编译期应拒绝）── //
struct RecursiveBad
{
    std::string                    name;
    std::vector<RecursiveBad>      children;
};

// ── 10. 综合大爆炸 ── //
struct MegaWeather
{
    [[= Desc("城市名，如「杭州」")]]        std::string             location;
    [[= Desc("温度值")]]                    double                  temperature;
    [[= Desc("温度单位")]]                  Unit                    unit = Unit::Celsius;
    [[= Desc("是否启用")]]                  std::optional<bool>     enabled;
    [[= Desc("小时级温度")]]                std::vector<int>        hourly;
    [[= Desc("坐标")]]                      std::array<double, 2>  coord;
    [[= Desc("描述的标签")]]                std::optional<std::string> description;
    [[= Desc("子任务")]]                    std::vector<TaskItem>   sub_tasks;
};

// ================================================================== //
// 以下是模版套模版的极端嵌套模式，PLAN.md 没有覆盖
// ================================================================== //

// ── 11. optional<vector<T>> ── //
struct WithOptVector
{
    [[= Desc("名称")]]                      std::string                    name;
    [[= Desc("分数列表")]]                  std::optional<std::vector<int>> scores;
};

// ── 12. optional<array<T, N>> ── //
struct WithOptArray
{
    [[= Desc("名称")]]                      std::string                       name;
    [[= Desc("坐标")]]                      std::optional<std::array<int, 4>> coords;
};

// ── 13. optional<struct> ── //
struct WithOptStruct
{
    [[= Desc("名称")]]                      std::string               name;
    [[= Desc("中心点")]]                    std::optional<Coordinate> center;
};

// ── 14. vector<vector<T>>（嵌套 vector）── //
struct WithNestedVec
{
    [[= Desc("名称")]]                      std::string                   name;
    [[= Desc("矩阵")]]                      std::vector<std::vector<int>> matrix;
};

// ── 15. vector<array<T, N>> ── //
struct WithVecArray
{
    [[= Desc("名称")]]                      std::string                     name;
    [[= Desc("点集")]]                      std::vector<std::array<int, 3>> points;
};

// ── 16. array<vector<T>, N> ── //
struct WithArrayVec
{
    [[= Desc("名称")]]                      std::string                     name;
    [[= Desc("分组")]]                      std::array<std::vector<int>, 2> groups;
};

// ── 17. 二维 array（array<array<T, N>, M>）── //
struct With2DArray
{
    [[= Desc("名称")]]                      std::string                          name;
    [[= Desc("网格")]]                      std::array<std::array<int, 3>, 2>    grid;
};

// ── 18. double optional ── //
struct WithDoubleOpt
{
    [[= Desc("名称")]]                      std::string                                 name;
    [[= Desc("别名")]]                      std::optional<std::optional<std::string>>   nick;
};

// ── 19. 枚举在容器中 ── //
struct WithEnumContainer
{
    [[= Desc("名称")]]                      std::string            name;
    [[= Desc("颜色列表")]]                  std::vector<Color>     colors;
    [[= Desc("前三")]]                      std::array<Color, 3>   top3;
    [[= Desc("最爱")]]                      std::optional<Color>   favorite;
};

// ── 20. optional 枚举在 array 中 ── //
struct WithOptEnumArray
{
    [[= Desc("名称")]]                      std::string                       name;
    [[= Desc("颜色槽")]]                    std::array<std::optional<Color>, 3> slots;
};

// ── 21. vector<optional<enum>> ── //
struct WithVecOptEnum
{
    [[= Desc("名称")]]                      std::string                        name;
    [[= Desc("可选颜色")]]                  std::vector<std::optional<Color>>  colors;
};

// ── 22. 终极嵌套怪 ── //
struct UltimateNested
{
    [[= Desc("ID")]]                                                std::string                                              id;
    [[= Desc("颜色矩阵")]]                                          std::optional<std::vector<std::array<std::optional<Color>, 2>>> color_grids;
    [[= Desc("二维矩阵")]]                                          std::vector<std::vector<int>>                            matrix;
    [[= Desc("标签")]]                                              std::array<std::optional<std::string>, 3>                labels;
    [[= Desc("额外里程碑")]]                                        std::optional<Milestone>                               extra_milestone;
};

// ── 23. null 用于非 optional 字段 ── //
struct WithRequired
{
    [[= Desc("名称")]]                      std::string             name;
    [[= Desc("数量")]]                      int                     count;
};

// ================================================================== //
// 以下测试：tools.hpp 不存在时编译即 RED
// ================================================================== //

// ── schema_of：PrimitiveTypes ── //

static_assert(agent::detail::reflect_members<PrimitiveTypes>().size() == 6,
    "reflect_members should see 6 fields");

TEST_CASE("schema_of PrimitiveTypes")
{
    auto j = agent::schema_of<PrimitiveTypes>();
    CHECK(j.is_object());
    CHECK(j["type"] == "object");
    CHECK(j["properties"].size() == 6);
    CHECK(j["properties"]["name"]["type"] == "string");
    CHECK(j["properties"]["name"]["description"] == "名字");
    CHECK(j["properties"]["count"]["type"] == "integer");
    CHECK(j["properties"]["ratio"]["type"] == "number");
    CHECK(j["properties"]["enabled"]["type"] == "boolean");
    CHECK(j["properties"]["desc"]["default"] == "默认描述");
    CHECK(j["properties"]["desc"]["default"] == "默认描述");
    CHECK(j["properties"]["step"]["default"] == 10);
    CHECK(j["required"].size() == 4);
    CHECK(j["required"][0] == "name");
    CHECK(j["required"][1] == "count");
    CHECK(j["required"][2] == "ratio");
    CHECK(j["required"][3] == "enabled");
}

// ── assign_from_json：PrimitiveTypes ── //

TEST_CASE("assign_from_json<PrimitiveTypes> valid")
{
    auto r = agent::assign_from_json<PrimitiveTypes>({
        {"name", "test"}, {"count", 42}, {"ratio", 3.14}, {"enabled", true}
    });
    REQUIRE(r.has_value());
    CHECK(r->name == "test");
    CHECK(r->count == 42);
    CHECK(r->ratio == doctest::Approx(3.14));
    CHECK(r->enabled == true);
    CHECK(r->desc == "默认描述");
    CHECK(r->step == 10);
}

TEST_CASE("assign_from_json<PrimitiveTypes> missing required")
{
    auto r = agent::assign_from_json<PrimitiveTypes>({{"name", "test"}});
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::InvalidArgs);
}

TEST_CASE("assign_from_json<PrimitiveTypes> wrong type")
{
    auto r = agent::assign_from_json<PrimitiveTypes>({
        {"name", 123}, {"count", "not_int"}, {"ratio", false}, {"enabled", 1}
    });
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::InvalidArgs);
}

TEST_CASE("assign_from_json<PrimitiveTypes> int overflow")
{
    auto r = agent::assign_from_json<PrimitiveTypes>({
        {"name", "x"}, {"count", 999999999999}, {"ratio", 1.0}, {"enabled", true}
    });
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::InvalidArgs);
}


// ── schema_of：WithEnum ── //

TEST_CASE("schema_of WithEnum")
{
    auto j = agent::schema_of<WithEnum>();
    CHECK(j.is_object());
    auto const& props = j["properties"];
    CHECK(props.contains("color"));
    CHECK(props.contains("unit"));
    if (props.contains("color")) {
        auto const& c = props["color"];
        CHECK(c.value("type", std::string("")) == "string");
    }
}

TEST_CASE("assign_from_json<WithEnum> valid")
{
    auto r = agent::assign_from_json<WithEnum>({
        {"color", "Red"}, {"unit", "Fahrenheit"}
    });
    REQUIRE(r.has_value());
    CHECK(r->color == Color::Red);
    CHECK(r->unit == Unit::Fahrenheit);
}

TEST_CASE("assign_from_json<WithEnum> invalid enum value")
{
    auto r = agent::assign_from_json<WithEnum>({{"color", "Purple"}});
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::InvalidArgs);
}

TEST_CASE("assign_from_json<WithEnum> wrong enum type")
{
    auto r = agent::assign_from_json<WithEnum>({{"color", 42}});
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::InvalidArgs);
}


// ── schema_of：WithOptional ── //

TEST_CASE("schema_of WithOptional")
{
    auto j = agent::schema_of<WithOptional>();
    CHECK(j["properties"]["name"]["type"] == "string");
    CHECK(j["properties"]["age"]["type"].is_array());
    CHECK(j["properties"]["age"]["type"][0] == "integer");
    CHECK(j["properties"]["age"]["type"][1] == "null");
    CHECK(j["required"].size() == 1);
    CHECK(j["required"][0] == "name");
}

TEST_CASE("assign_from_json<WithOptional> valid with all fields")
{
    auto r = agent::assign_from_json<WithOptional>({
        {"name", "hello"}, {"age", 25}, {"active", true},
        {"favorite_color", "Green"}, {"comment", "nice"}
    });
    REQUIRE(r.has_value());
    CHECK(r->name == "hello");
    CHECK(r->age == 25);
    CHECK(r->active == true);
    CHECK(r->favorite_color == Color::Green);
    CHECK(r->comment == "nice");
}

TEST_CASE("assign_from_json<WithOptional> null optionals")
{
    auto r = agent::assign_from_json<WithOptional>({
        {"name", "hello"}, {"age", nullptr}, {"active", nullptr},
        {"favorite_color", nullptr}, {"comment", nullptr}
    });
    REQUIRE(r.has_value());
    CHECK(r->name == "hello");
    CHECK(!r->age.has_value());
    CHECK(!r->active.has_value());
    CHECK(!r->favorite_color.has_value());
    CHECK(!r->comment.has_value());
}

TEST_CASE("assign_from_json<WithOptional> missing optionals")
{
    auto r = agent::assign_from_json<WithOptional>({{"name", "hello"}});
    REQUIRE(r.has_value());
    CHECK(!r->age.has_value());
}

TEST_CASE("assign_from_json<WithOptional> missing required")
{
    auto r = agent::assign_from_json<WithOptional>({});
    CHECK(!r.has_value());
}


// ── schema_of：WithVector ── //

TEST_CASE("schema_of WithVector")
{
    auto j = agent::schema_of<WithVector>();
    CHECK(j["properties"]["scores"]["type"] == "array");
    CHECK(j["properties"]["scores"]["items"]["type"] == "integer");
    CHECK(j["properties"]["tasks"]["items"]["type"] == "object");
}

TEST_CASE("assign_from_json<WithVector> valid")
{
    auto r = agent::assign_from_json<WithVector>({
        {"project_name", "proj"},
        {"tasks", nlohmann::json::array({
            {{"title", "t1"}, {"priority", 1}},
            {{"title", "t2"}, {"priority", 2}}
        })},
        {"scores", {90, 80, 70}},
        {"tags", {"c++", "test"}}
    });
    REQUIRE(r.has_value());
    CHECK(r->project_name == "proj");
    CHECK(r->tasks.size() == 2);
    CHECK(r->tasks[0].title == "t1");
    CHECK(r->tasks[0].priority == 1);
    CHECK(r->tasks[1].priority == 2);
    CHECK(r->scores.size() == 3);
    CHECK(r->tags[1] == "test");
}

TEST_CASE("assign_from_json<WithVector> empty arrays")
{
    auto r = agent::assign_from_json<WithVector>({
        {"project_name", "empty"},
        {"tasks", nlohmann::json::array()},
        {"scores", nlohmann::json::array()},
        {"tags", nlohmann::json::array()}
    });
    REQUIRE(r.has_value());
    CHECK(r->tasks.empty());
    CHECK(r->scores.empty());
    CHECK(r->tags.empty());
}

TEST_CASE("assign_from_json<WithVector> array element wrong type")
{
    auto r = agent::assign_from_json<WithVector>({
        {"project_name", "x"},
        {"tasks", nlohmann::json::array()},
        {"scores", {"not_int"}},
        {"tags", nlohmann::json::array()}
    });
    CHECK(!r.has_value());
}

TEST_CASE("assign_from_json<WithVector> missing required in nested")
{
    auto r = agent::assign_from_json<WithVector>({
        {"project_name", "x"},
        {"tasks", nlohmann::json::array({
            {{"title", "t1"}}
        })},
        {"scores", {1}},
        {"tags", {"a"}}
    });
    CHECK(!r.has_value());
}


// ── schema_of：WithArray ── //

TEST_CASE("schema_of WithArray")
{
    auto j = agent::schema_of<WithArray>();
    CHECK(j["properties"]["rgb"]["type"] == "array");
    CHECK(j["properties"]["rgb"]["minItems"] == 3);
    CHECK(j["properties"]["rgb"]["maxItems"] == 3);
    CHECK(j["properties"]["range"]["minItems"] == 2);
    CHECK(j["properties"]["corners"]["minItems"] == 4);
}

TEST_CASE("assign_from_json<WithArray> valid")
{
    auto r = agent::assign_from_json<WithArray>({
        {"name", "arr"},
        {"rgb", {255, 128, 0}},
        {"range", {-1.0, 1.0}},
        {"corners", nlohmann::json::array({
            {{"x", 0.0}, {"y", 0.0}},
            {{"x", 1.0}, {"y", 0.0}},
            {{"x", 1.0}, {"y", 1.0}},
            {{"x", 0.0}, {"y", 1.0}}
        })}
    });
    REQUIRE(r.has_value());
    CHECK(r->rgb[0] == 255);
    CHECK(r->rgb[1] == 128);
    CHECK(r->rgb[2] == 0);
    CHECK(r->range[0] == doctest::Approx(-1.0));
    CHECK(r->corners[3].x == doctest::Approx(0.0));
}

TEST_CASE("assign_from_json<WithArray> wrong length")
{
    auto r = agent::assign_from_json<WithArray>({
        {"name", "x"},
        {"rgb", {1, 2}},
        {"range", {0.0, 1.0}},
        {"corners", nlohmann::json::array({
            {{"x", 0}, {"y", 0}},
            {{"x", 0}, {"y", 0}},
            {{"x", 0}, {"y", 0}},
            {{"x", 0}, {"y", 0}}
        })}
    });
    CHECK(!r.has_value());
}

TEST_CASE("assign_from_json<WithArray> element wrong type")
{
    auto r = agent::assign_from_json<WithArray>({
        {"name", "x"},
        {"rgb", {"red", "green", "blue"}},
        {"range", {0.0, 1.0}},
        {"corners", nlohmann::json::array({
            {{"x", 0}, {"y", 0}},
            {{"x", 0}, {"y", 0}},
            {{"x", 0}, {"y", 0}},
            {{"x", 0}, {"y", 0}}
        })}
    });
    CHECK(!r.has_value());
}


// ── schema_of：WithVecOptional ── //

TEST_CASE("schema_of WithVecOptional")
{
    auto j = agent::schema_of<WithVecOptional>();
    CHECK(j["properties"]["tags"]["type"] == "array");
    CHECK(j["properties"]["tags"]["items"]["type"].is_array());
    CHECK(j["properties"]["tags"]["items"]["type"][0] == "string");
    CHECK(j["properties"]["tags"]["items"]["type"][1] == "null");
}

TEST_CASE("assign_from_json<WithVecOptional> with null elements")
{
    auto r = agent::assign_from_json<WithVecOptional>({
        {"tags", {"a", nullptr, "b"}},
        {"scores", {1, nullptr, 3}}
    });
    REQUIRE(r.has_value());
    CHECK(r->tags.size() == 3);
    CHECK(r->tags[0] == "a");
    CHECK(!r->tags[1].has_value());
    CHECK(r->tags[2] == "b");
    CHECK(!r->scores[1].has_value());
}

TEST_CASE("assign_from_json<WithVecOptional> empty array")
{
    auto r = agent::assign_from_json<WithVecOptional>({
        {"tags", nlohmann::json::array()},
        {"scores", nlohmann::json::array()}
    });
    REQUIRE(r.has_value());
    CHECK(r->tags.empty());
}


// ── schema_of：NestedStruct ── //

TEST_CASE("schema_of NestedStruct")
{
    auto j = agent::schema_of<NestedStruct>();
    CHECK(j["properties"]["coord"]["type"] == "object");
    CHECK(j["properties"]["coord"]["properties"]["x"]["type"] == "number");
}

TEST_CASE("assign_from_json<NestedStruct> valid")
{
    auto r = agent::assign_from_json<NestedStruct>({
        {"coord", {{"x", 1.5}, {"y", 2.5}}},
        {"label", "pt"}
    });
    REQUIRE(r.has_value());
    CHECK(r->coord.x == doctest::Approx(1.5));
    CHECK(r->coord.y == doctest::Approx(2.5));
    CHECK(r->label == "pt");
}

TEST_CASE("assign_from_json<NestedStruct> nested missing field")
{
    auto r = agent::assign_from_json<NestedStruct>({
        {"coord", {{"x", 1.0}}},
        {"label", "pt"}
    });
    CHECK(!r.has_value());
}


// ── schema_of：OrganizationTree（5 层嵌套）── //

TEST_CASE("schema_of OrganizationTree")
{
    auto j = agent::schema_of<OrganizationTree>();
    CHECK(j["properties"]["departments"]["type"] == "array");
    CHECK(j["properties"]["departments"]["items"]["type"] == "object");
    CHECK(j["properties"]["departments"]["items"]["properties"]["teams"]["minItems"] == 3);
    CHECK(j["properties"]["departments"]["items"]["properties"]["teams"]["items"]["type"] == "object");
    CHECK(j["properties"]["departments"]["items"]["properties"]["teams"]["items"]["properties"]["projects"]["items"]["type"] == "object");
    CHECK(j["properties"]["departments"]["items"]["properties"]["teams"]["items"]["properties"]["projects"]["items"]["properties"]["milestones"]["minItems"] == 4);
}

TEST_CASE("assign_from_json<OrganizationTree> 5-level nesting")
{
    auto r = agent::assign_from_json<OrganizationTree>(
        nlohmann::json::parse(R"({
            "name": "ACME Corp",
            "departments": [{
                "name": "Engineering",
                "budget_code": "E-001",
                "teams": [
                    {
                        "name": "Platform",
                        "lead": "Alice",
                        "projects": [{
                            "name": "Core v2",
                            "description": "Rewrite",
                            "milestones": [
                                {"name": "Design", "due_date": "2026-01"},
                                {"name": "Impl"},
                                {"name": "Test", "due_date": "2026-06"},
                                {"name": "Ship"}
                            ]
                        }]
                    },
                    {"name": "Backend", "projects": []},
                    {"name": "Frontend", "projects": []}
                ]
            }]
        })")
    );
    REQUIRE(r.has_value());
    CHECK(r->name == "ACME Corp");
    CHECK(r->departments.size() == 1);
    CHECK(r->departments[0].name == "Engineering");
    CHECK(r->departments[0].budget_code == "E-001");
    CHECK(r->departments[0].teams.size() == 3);
    CHECK(r->departments[0].teams[0].name == "Platform");
    CHECK(r->departments[0].teams[0].lead == "Alice");
    CHECK(r->departments[0].teams[0].projects.size() == 1);
    CHECK(r->departments[0].teams[0].projects[0].name == "Core v2");
    CHECK(r->departments[0].teams[0].projects[0].milestones.size() == 4);
    CHECK(r->departments[0].teams[0].projects[0].milestones[0].name == "Design");
    CHECK(r->departments[0].teams[0].projects[0].milestones[0].due_date == "2026-01");
    CHECK(!r->departments[0].teams[0].projects[0].milestones[1].due_date.has_value());
}

TEST_CASE("assign_from_json<OrganizationTree> missing nested required")
{
    auto r = agent::assign_from_json<OrganizationTree>(
        nlohmann::json::parse(R"({
            "name": "X",
            "departments": [{
                "name": "D1",
                "teams": [{
                    "name": "T1",
                    "projects": [{
                        "milestones": [
                            {"name": "M1", "due_date": "2026-01"},
                            {"name": "M2"},
                            {"name": "M3"},
                            {"name": "M4"}
                        ]
                    }]
                }]
            }]
        })")
    );
    CHECK(!r.has_value());
}


// ── schema_of：MegaWeather ── //

TEST_CASE("schema_of MegaWeather")
{
    auto j = agent::schema_of<MegaWeather>();
    CHECK(j["properties"]["hourly"]["type"] == "array");
    CHECK(j["properties"]["coord"]["minItems"] == 2);
    CHECK(j["properties"]["unit"]["default"] == "Celsius");  // 枚举默认值反射为枚举名
}

TEST_CASE("assign_from_json<MegaWeather> full")
{
    auto r = agent::assign_from_json<MegaWeather>({
        {"location", "杭州"},
        {"temperature", 28.5},
        {"unit", "Fahrenheit"},
        {"enabled", true},
        {"hourly", {26, 27, 28, 29}},
        {"coord", {120.2, 30.3}},
        {"description", "热"},
        {"sub_tasks", nlohmann::json::array({{{"title", "t1"}, {"priority", 1}}})}
    });
    REQUIRE(r.has_value());
    CHECK(r->location == "杭州");
    CHECK(r->temperature == doctest::Approx(28.5));
    CHECK(r->unit == Unit::Fahrenheit);
    CHECK(r->enabled == true);
    CHECK(r->hourly.size() == 4);
    CHECK(r->coord[0] == doctest::Approx(120.2));
    CHECK(r->description == "热");
    CHECK(r->sub_tasks.size() == 1);
}

TEST_CASE("assign_from_json<MegaWeather> minimal")
{
    auto r = agent::assign_from_json<MegaWeather>({
        {"location", "x"}, {"temperature", 0.0},
        {"hourly", nlohmann::json::array()},
        {"coord", {0, 0}},
        {"sub_tasks", nlohmann::json::array()}
    });
    REQUIRE(r.has_value());
    CHECK(r->unit == Unit::Celsius);
    CHECK(!r->enabled.has_value());
    CHECK(!r->description.has_value());
}

TEST_CASE("assign_from_json<MegaWeather> missing required")
{
    auto r = agent::assign_from_json<MegaWeather>({
        {"temperature", 0.0}, {"hourly", {}},
        {"coord", {0, 0}}, {"sub_tasks", nlohmann::json::array()}
    });
    CHECK(!r.has_value());
}


// ═══════════════════════════════════════════════════════ //
//  模版套模版极端嵌套测试                                 //
// ═══════════════════════════════════════════════════════ //

// ── schema_of：WithOptVector ── //

TEST_CASE("schema_of WithOptVector")
{
    auto j = agent::schema_of<WithOptVector>();
    CHECK(j["properties"]["scores"]["type"].is_array());
    CHECK(j["properties"]["scores"]["type"][0] == "array");
    CHECK(j["properties"]["scores"]["type"][1] == "null");
    CHECK(j["properties"]["scores"]["items"]["type"] == "integer");
}

TEST_CASE("assign_from_json<WithOptVector> valid")
{
    auto r = agent::assign_from_json<WithOptVector>({
        {"name", "test"}, {"scores", {90, 80, 70}}
    });
    REQUIRE(r.has_value());
    CHECK(r->name == "test");
    REQUIRE(r->scores.has_value());
    CHECK(r->scores->size() == 3);
}

TEST_CASE("assign_from_json<WithOptVector> null")
{
    auto r = agent::assign_from_json<WithOptVector>({
        {"name", "test"}, {"scores", nullptr}
    });
    REQUIRE(r.has_value());
    CHECK(!r->scores.has_value());
}

TEST_CASE("assign_from_json<WithOptVector> missing")
{
    auto r = agent::assign_from_json<WithOptVector>({{"name", "test"}});
    REQUIRE(r.has_value());
    CHECK(!r->scores.has_value());
}


// ── schema_of：WithOptArray ── //

TEST_CASE("schema_of WithOptArray")
{
    auto j = agent::schema_of<WithOptArray>();
    CHECK(j["properties"]["coords"]["type"][0] == "array");
    CHECK(j["properties"]["coords"]["type"][1] == "null");
    CHECK(j["properties"]["coords"]["minItems"] == 4);
}

TEST_CASE("assign_from_json<WithOptArray> valid")
{
    auto r = agent::assign_from_json<WithOptArray>({
        {"name", "x"}, {"coords", {1, 2, 3, 4}}
    });
    REQUIRE(r.has_value());
    REQUIRE(r->coords.has_value());
    CHECK((*r->coords)[0] == 1);
}

TEST_CASE("assign_from_json<WithOptArray> null")
{
    auto r = agent::assign_from_json<WithOptArray>({{"name", "x"}, {"coords", nullptr}});
    REQUIRE(r.has_value());
    CHECK(!r->coords.has_value());
}


// ── schema_of：WithOptStruct ── //

TEST_CASE("schema_of WithOptStruct")
{
    auto j = agent::schema_of<WithOptStruct>();
    CHECK(j["properties"]["center"]["type"][0] == "object");
    CHECK(j["properties"]["center"]["type"][1] == "null");
}

TEST_CASE("assign_from_json<WithOptStruct> valid")
{
    auto r = agent::assign_from_json<WithOptStruct>({
        {"name", "x"}, {"center", {{"x", 1.5}, {"y", 2.5}}}
    });
    REQUIRE(r.has_value());
    REQUIRE(r->center.has_value());
    CHECK(r->center->x == doctest::Approx(1.5));
}

TEST_CASE("assign_from_json<WithOptStruct> null")
{
    auto r = agent::assign_from_json<WithOptStruct>({{"name", "x"}, {"center", nullptr}});
    REQUIRE(r.has_value());
    CHECK(!r->center.has_value());
}


// ── schema_of：WithNestedVec ── //

TEST_CASE("schema_of WithNestedVec")
{
    auto j = agent::schema_of<WithNestedVec>();
    CHECK(j["properties"]["matrix"]["type"] == "array");
    CHECK(j["properties"]["matrix"]["items"]["type"] == "array");
    CHECK(j["properties"]["matrix"]["items"]["items"]["type"] == "integer");
}

TEST_CASE("assign_from_json<WithNestedVec> valid")
{
    auto r = agent::assign_from_json<WithNestedVec>({
        {"name", "mat"}, {"matrix", {{1, 2}, {3, 4}, {5, 6}}}
    });
    REQUIRE(r.has_value());
    CHECK(r->matrix.size() == 3);
    CHECK(r->matrix[0].size() == 2);
    CHECK(r->matrix[1][1] == 4);
}

TEST_CASE("assign_from_json<WithNestedVec> empty outer")
{
    auto r = agent::assign_from_json<WithNestedVec>({
        {"name", "e"}, {"matrix", nlohmann::json::array()}
    });
    REQUIRE(r.has_value());
    CHECK(r->matrix.empty());
}


// ── schema_of：WithVecArray ── //

TEST_CASE("schema_of WithVecArray")
{
    auto j = agent::schema_of<WithVecArray>();
    CHECK(j["properties"]["points"]["items"]["type"] == "array");
    CHECK(j["properties"]["points"]["items"]["minItems"] == 3);
}

TEST_CASE("assign_from_json<WithVecArray> valid")
{
    auto r = agent::assign_from_json<WithVecArray>({
        {"name", "x"}, {"points", {{1, 2, 3}, {4, 5, 6}}}
    });
    REQUIRE(r.has_value());
    CHECK(r->points.size() == 2);
    CHECK(r->points[0][1] == 2);
}

TEST_CASE("assign_from_json<WithVecArray> wrong element length")
{
    auto r = agent::assign_from_json<WithVecArray>({
        {"name", "x"}, {"points", {{1, 2}}}
    });
    CHECK(!r.has_value());
}


// ── schema_of：WithArrayVec ── //

TEST_CASE("schema_of WithArrayVec")
{
    auto j = agent::schema_of<WithArrayVec>();
    CHECK(j["properties"]["groups"]["type"] == "array");
    CHECK(j["properties"]["groups"]["minItems"] == 2);
    CHECK(j["properties"]["groups"]["items"]["type"] == "array");
}

TEST_CASE("assign_from_json<WithArrayVec> valid")
{
    auto r = agent::assign_from_json<WithArrayVec>({
        {"name", "x"}, {"groups", {{1, 2}, {3, 4, 5}}}
    });
    REQUIRE(r.has_value());
    CHECK(r->groups.size() == 2);
    CHECK(r->groups[0].size() == 2);
    CHECK(r->groups[1].size() == 3);
}


// ── schema_of：With2DArray ── //

TEST_CASE("schema_of With2DArray")
{
    auto j = agent::schema_of<With2DArray>();
    CHECK(j["properties"]["grid"]["type"] == "array");
    CHECK(j["properties"]["grid"]["minItems"] == 2);
    CHECK(j["properties"]["grid"]["items"]["type"] == "array");
    CHECK(j["properties"]["grid"]["items"]["minItems"] == 3);
}

TEST_CASE("assign_from_json<With2DArray> valid")
{
    auto r = agent::assign_from_json<With2DArray>({
        {"name", "g"}, {"grid", {{1, 2, 3}, {4, 5, 6}}}
    });
    REQUIRE(r.has_value());
    CHECK(r->grid[0][0] == 1);
    CHECK(r->grid[1][2] == 6);
}


// ── schema_of：WithDoubleOpt ── //

TEST_CASE("schema_of WithDoubleOpt")
{
    auto j = agent::schema_of<WithDoubleOpt>();
    CHECK(j["properties"]["nick"]["type"].is_array());
    CHECK(j["properties"]["nick"]["type"][0] == "string");
    CHECK(j["properties"]["nick"]["type"][1] == "null");
}

TEST_CASE("assign_from_json<WithDoubleOpt> string")
{
    auto r = agent::assign_from_json<WithDoubleOpt>({{"name", "x"}, {"nick", "hello"}});
    REQUIRE(r.has_value());
    REQUIRE(r->nick.has_value());
    CHECK(r->nick->has_value());
    CHECK(**r->nick == "hello");
}

TEST_CASE("assign_from_json<WithDoubleOpt> null")
{
    auto r = agent::assign_from_json<WithDoubleOpt>({{"name", "x"}, {"nick", nullptr}});
    REQUIRE(r.has_value());
    CHECK(!r->nick.has_value());
}


// ── schema_of：WithEnumContainer ── //

TEST_CASE("schema_of WithEnumContainer")
{
    auto j = agent::schema_of<WithEnumContainer>();
    CHECK(j["properties"]["colors"]["items"]["type"] == "string");
    CHECK(j["properties"]["colors"]["items"]["enum"].is_array());
    CHECK(j["properties"]["top3"]["minItems"] == 3);
    CHECK(j["properties"]["favorite"]["type"][0] == "string");
    CHECK(j["properties"]["favorite"]["type"][1] == "null");
}

TEST_CASE("assign_from_json<WithEnumContainer> valid")
{
    auto r = agent::assign_from_json<WithEnumContainer>({
        {"name", "x"}, {"colors", {"Red", "Green"}},
        {"top3", {"Red", "Green", "Blue"}}, {"favorite", "Red"}
    });
    REQUIRE(r.has_value());
    CHECK(r->colors.size() == 2);
    CHECK(r->colors[0] == Color::Red);
    CHECK(r->top3[2] == Color::Blue);
    CHECK(r->favorite == Color::Red);
}

TEST_CASE("assign_from_json<WithEnumContainer> invalid enum in vector")
{
    auto r = agent::assign_from_json<WithEnumContainer>({
        {"name", "x"}, {"colors", {"Red", "Purple"}},
        {"top3", {"Red", "Green", "Blue"}}
    });
    CHECK(!r.has_value());
}

TEST_CASE("assign_from_json<WithEnumContainer> wrong array length")
{
    auto r = agent::assign_from_json<WithEnumContainer>({
        {"name", "x"}, {"colors", {"Red"}},
        {"top3", {"Red", "Green"}}, {"favorite", nullptr}
    });
    CHECK(!r.has_value());
}


// ── schema_of：WithOptEnumArray ── //

TEST_CASE("schema_of WithOptEnumArray")
{
    auto j = agent::schema_of<WithOptEnumArray>();
    CHECK(j["properties"]["slots"]["type"] == "array");
    CHECK(j["properties"]["slots"]["minItems"] == 3);
    CHECK(j["properties"]["slots"]["items"]["type"].is_array());
    CHECK(j["properties"]["slots"]["items"]["type"][0] == "string");
    CHECK(j["properties"]["slots"]["items"]["type"][1] == "null");
}

TEST_CASE("assign_from_json<WithOptEnumArray> valid")
{
    auto r = agent::assign_from_json<WithOptEnumArray>({
        {"name", "x"}, {"slots", {"Red", nullptr, "Blue"}}
    });
    REQUIRE(r.has_value());
    CHECK(r->slots[0] == Color::Red);
    CHECK(!r->slots[1].has_value());
    CHECK(r->slots[2] == Color::Blue);
}


// ── schema_of：WithVecOptEnum ── //

TEST_CASE("schema_of WithVecOptEnum")
{
    auto j = agent::schema_of<WithVecOptEnum>();
    CHECK(j["properties"]["colors"]["type"] == "array");
    CHECK(j["properties"]["colors"]["items"]["type"].is_array());
    CHECK(j["properties"]["colors"]["items"]["type"][0] == "string");
    CHECK(j["properties"]["colors"]["items"]["type"][1] == "null");
}

TEST_CASE("assign_from_json<WithVecOptEnum> valid")
{
    auto r = agent::assign_from_json<WithVecOptEnum>({
        {"name", "x"}, {"colors", {"Red", nullptr, "Blue"}}
    });
    REQUIRE(r.has_value());
    CHECK(r->colors.size() == 3);
    CHECK(r->colors[0] == Color::Red);
    CHECK(!r->colors[1].has_value());
    CHECK(r->colors[2] == Color::Blue);
}


// ── schema_of：UltimateNested ── //

TEST_CASE("schema_of UltimateNested")
{
    auto j = agent::schema_of<UltimateNested>();
    CHECK(j["properties"]["color_grids"]["type"][0] == "array");
    CHECK(j["properties"]["color_grids"]["type"][1] == "null");
    CHECK(j["properties"]["matrix"]["items"]["type"] == "array");
    CHECK(j["properties"]["labels"]["type"] == "array");
    CHECK(j["properties"]["labels"]["minItems"] == 3);
    CHECK(j["properties"]["extra_milestone"]["type"][0] == "object");
    CHECK(j["properties"]["extra_milestone"]["type"][1] == "null");
}

TEST_CASE("assign_from_json<UltimateNested> full")
{
    auto r = agent::assign_from_json<UltimateNested>({
        {"id", "u1"},
        {"color_grids", nlohmann::json::array({
            nlohmann::json::array({"Red", nullptr}),
            nlohmann::json::array({"Green", "Blue"})
        })},
        {"matrix", {{1, 2}, {3, 4}}},
        {"labels", {"a", nullptr, "c"}},
        {"extra_milestone", {{"name", "extra"}, {"due_date", "2026-07"}}}
    });
    REQUIRE(r.has_value());
    CHECK(r->id == "u1");
    REQUIRE(r->color_grids.has_value());
    CHECK(r->color_grids->size() == 2);
    CHECK((*r->color_grids)[0].size() == 2);
    CHECK((*r->color_grids)[0][0] == Color::Red);
    CHECK(!(*r->color_grids)[0][1].has_value());
    CHECK(r->matrix.size() == 2);
    CHECK(r->labels.size() == 3);
    CHECK(r->labels[0] == "a");
    CHECK(!r->labels[1].has_value());
    REQUIRE(r->extra_milestone.has_value());
    CHECK(r->extra_milestone->name == "extra");
}

TEST_CASE("assign_from_json<UltimateNested> minimal")
{
    auto r = agent::assign_from_json<UltimateNested>({
        {"id", "u1"},
        {"matrix", nlohmann::json::array()},
        {"labels", {nullptr, nullptr, nullptr}}
    });
    REQUIRE(r.has_value());
    CHECK(!r->color_grids.has_value());
    CHECK(r->matrix.empty());
    CHECK(!r->extra_milestone.has_value());
}


// ── null 用于非 optional 字段 ── //

TEST_CASE("assign_from_json<WithRequired> null on required")
{
    auto r = agent::assign_from_json<WithRequired>({
        {"name", nullptr}, {"count", 1}
    });
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::InvalidArgs);
}

TEST_CASE("assign_from_json<WithRequired> missing required")
{
    auto r = agent::assign_from_json<WithRequired>({{"name", "x"}});
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::InvalidArgs);
}

TEST_CASE("assign_from_json unsupported extra keys")
{
    auto r = agent::assign_from_json<WithRequired>({
        {"name", "x"}, {"count", 1}, {"extra_field", "ignored"}
    });
    REQUIRE(r.has_value());
    CHECK(r->name == "x");
    CHECK(r->count == 1);
}


// ── 空字符串与 UTF-8 ── //

TEST_CASE("assign_from_json empty string")
{
    auto r = agent::assign_from_json<PrimitiveTypes>({
        {"name", ""}, {"count", 0}, {"ratio", 0.0}, {"enabled", false}
    });
    REQUIRE(r.has_value());
    CHECK(r->name.empty());
}

TEST_CASE("assign_from_json UTF-8 special chars")
{
    auto r = agent::assign_from_json<PrimitiveTypes>({
        {"name", "日本語🔥🚀"}, {"count", 1}, {"ratio", 1.0}, {"enabled", true}
    });
    REQUIRE(r.has_value());
    CHECK(r->name == "日本語🔥🚀");
}


// ── 浮点数极端值 ── //

TEST_CASE("assign_from_json double extreme values")
{
    auto r = agent::assign_from_json<PrimitiveTypes>({
        {"name", "x"}, {"count", 0}, {"ratio", -1e100}, {"enabled", false}
    });
    REQUIRE(r.has_value());
    CHECK(r->ratio == doctest::Approx(-1e100));
}

TEST_CASE("assign_from_json double fractional")
{
    auto r = agent::assign_from_json<PrimitiveTypes>({
        {"name", "x"}, {"count", 0}, {"ratio", 1e-10}, {"enabled", false}
    });
    REQUIRE(r.has_value());
    CHECK(r->ratio == doctest::Approx(1e-10));
}


// ── 字段可见性与命名 ── //

TEST_CASE("assign_from_json mixed field order")
{
    auto r = agent::assign_from_json<PrimitiveTypes>({
        {"enabled", true}, {"count", 42}, {"ratio", 1.0}, {"name", "ordered"}
    });
    REQUIRE(r.has_value());
    CHECK(r->name == "ordered");
    CHECK(r->count == 42);
}


// 递归类型编译期拒绝测试在 test_recursive_reject.cpp 中（预期编译失败）

// ── validate_args 非法输入鲁棒性测试 ── //

TEST_CASE("validate_args bizarre inputs do not crash")
{
    auto const& s = agent::schema_of<OrganizationTree>();
    std::vector<std::string> errors;

    // 非法类型：传标量代替 object
    agent::detail::validate_args(42, s, "", errors);
    agent::detail::validate_args("str", s, "", errors);
    agent::detail::validate_args(true, s, "", errors);
    agent::detail::validate_args(nullptr, s, "", errors);

    // 深层嵌套
    nlohmann::json deep = nlohmann::json::object();
    deep["a"]["b"]["c"]["d"]["e"] = 1;
    agent::detail::validate_args(deep, s, "", errors);

    // 超大数组
    nlohmann::json big_arr = nlohmann::json::array();
    for (int i = 0; i < 10000; i++) big_arr.push_back(i);
    agent::detail::validate_args(big_arr, s, "", errors);

    // 空 object vs 期望 object
    agent::detail::validate_args(nlohmann::json::object(), s, "", errors);

    // 类型完全错配
    nlohmann::json wrong = {
        {"name", 123},
        {"groups", "not_an_array"}
    };
    agent::detail::validate_args(wrong, s, "", errors);

    // 枚举 schema 对非字符串值
    auto const& es = agent::schema_of<WithEnum>();
    agent::detail::validate_args({{"color", 999}}, es, "", errors);
    agent::detail::validate_args({{"color", nullptr}}, es, "", errors);

    // 所有错误信息应为字符串
    for (auto const& e : errors)
        CHECK(!e.empty());

    CHECK(true);  // 如果能走到这里说明没炸
}

// ── 端到端：ToolBase CRTP 注册 + Tools::exec + 运行时注册 ── //

struct [[= Desc("获取指定城市的天气信息")]] GetWeatherTool
{
    struct params_type {
        [[= Desc("城市名，如「杭州」")]] std::string location;
        [[= Desc("温度单位")]]           std::string unit = "celsius";
    };

    static Result<std::string> invoke(params_type const& p)
    {
        return nlohmann::json{
            {"location", p.location}, {"unit", p.unit}, {"temp", 28}
        }.dump();
    }
};

template struct agent::ToolBase<GetWeatherTool>;

TEST_CASE("ToolBase auto-registers tool")
{
    auto tools = Tools::list();
    bool found = false;
    for (auto const& t : tools)
        if (t.name == "GetWeatherTool") found = true;
    CHECK(found);
}

TEST_CASE("Tools::get returns ToolBase-registered tool")
{
    auto r = Tools::get("GetWeatherTool");
    REQUIRE(r.has_value());
    CHECK(r->name == "GetWeatherTool");
    CHECK(r->description == "获取指定城市的天气信息");
    CHECK(r->parameters["type"] == "object");
}

TEST_CASE("Tools::exec ToolBase tool valid")
{
    auto r = Tools::exec("GetWeatherTool", {
        {"location", "杭州"}, {"unit", "celsius"}
    });
    REQUIRE(r.has_value());
    auto j = nlohmann::json::parse(*r);
    CHECK(j["location"] == "杭州");
    CHECK(j["temp"] == 28);
}

TEST_CASE("Tools::exec ToolBase tool missing required")
{
    auto r = Tools::exec("GetWeatherTool", {{"unit", "celsius"}});
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::InvalidArgs);
}

TEST_CASE("Tools::exec ToolBase tool wrong type")
{
    auto r = Tools::exec("GetWeatherTool", {
        {"location", 123}, {"unit", "celsius"}
    });
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::InvalidArgs);
}

TEST_CASE("Tools::exec ToolBase tool unknown extra fields")
{
    auto r = Tools::exec("GetWeatherTool", {
        {"location", "x"}, {"unit", "celsius"}, {"extra", 1}
    });
    REQUIRE(r.has_value());
}

TEST_CASE("Tools::exec tool not found")
{
    auto r = Tools::exec("NoSuchTool", {});
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::NotFound);
}

TEST_CASE("Tools::get tool not found")
{
    auto r = Tools::get("NoSuchTool");
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::NotFound);
}

// ── 运行时注册（不使用 ToolBase） ── //

TEST_CASE("Tools::reg runtime registration")
{
    auto fn = [](nlohmann::json args) -> Result<std::string> {
        return "hello " + args["name"].get<std::string>();
    };

    auto r = Tools::reg(
        ToolInfo{"greet", "向用户问好", R"({
            "type":"object",
            "properties":{"name":{"type":"string"}},
            "required":["name"]
        })"_json},
        std::move(fn)
    );
    CHECK(r.has_value());
}

TEST_CASE("Tools::exec runtime-registered tool")
{
    auto r = Tools::exec("greet", {{"name", "world"}});
    REQUIRE(r.has_value());
    CHECK(*r == "hello world");
}

TEST_CASE("Tools::reg duplicate returns Duplicate")
{
    auto fn = [](nlohmann::json) -> Result<std::string> { return ""; };
    auto r = Tools::reg(
        ToolInfo{"greet", "", R"({"type":"object"})"_json},
        std::move(fn)
    );
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::Duplicate);
}

TEST_CASE("Tools::exec runtime-registered missing required")
{
    auto r = Tools::exec("greet", {});
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::InvalidArgs);
}

TEST_CASE("Tools::exec runtime-registered wrong type")
{
    auto r = Tools::exec("greet", {{"name", 123}});
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::InvalidArgs);
}

TEST_CASE("Tools::exec runtime-registered null on non-optional")
{
    auto r = Tools::exec("greet", {{"name", nullptr}});
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::InvalidArgs);
}

// ── Tools::list 完整性 ── //

TEST_CASE("Tools::list contains both registered tools")
{
    auto tools = Tools::list();
    bool has_greet = false, has_weather = false;
    for (auto const& t : tools) {
        if (t.name == "greet") has_greet = true;
        if (t.name == "GetWeatherTool") has_weather = true;
    }
    CHECK(has_greet);
    CHECK(has_weather);
}

// ═══════════════════════════════════════════════════════ //
//  注册校验链：空名 / 空函数 / 畸形 schema 全部拒绝       //
// ═══════════════════════════════════════════════════════ //

TEST_CASE("Tools::reg rejects empty name")
{
    auto r = Tools::reg(
        ToolInfo{"", "无名工具", R"({"type":"object"})"_json},
        [](nlohmann::json) -> Result<std::string> { return ""; });
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::InvalidArgs);
}

TEST_CASE("Tools::reg rejects empty function")
{
    std::function<Result<std::string>(nlohmann::json)> empty_fn;
    auto r = Tools::reg(
        ToolInfo{"empty_fn_tool", "空函数工具", R"({"type":"object"})"_json},
        std::move(empty_fn));
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::InvalidArgs);
}

TEST_CASE("Tools::reg rejects malformed schemas")
{
    auto fn = [](nlohmann::json) -> Result<std::string> { return ""; };

    // root 缺 type
    auto r1 = Tools::reg(ToolInfo{"bad1", "", nlohmann::json::object()}, fn);
    CHECK(!r1.has_value());

    // root type 不是 object
    auto r2 = Tools::reg(ToolInfo{"bad2", "", R"({"type":"string"})"_json}, fn);
    CHECK(!r2.has_value());

    // 属性的 type 是数字——执行期校验器防崩溃的关键场景
    auto r3 = Tools::reg(ToolInfo{"bad3", "",
        R"({"type":"object","properties":{"x":{"type":123}}})"_json}, fn);
    CHECK(!r3.has_value());

    // required 引用未在 properties 声明的字段（拼写错误检测）
    auto r4 = Tools::reg(ToolInfo{"bad4", "",
        R"({"type":"object","properties":{"x":{"type":"string"}},"required":["y"]})"_json}, fn);
    CHECK(!r4.has_value());

    // enum 空数组
    auto r5 = Tools::reg(ToolInfo{"bad5", "",
        R"({"type":"object","properties":{"x":{"type":"string","enum":[]}}})"_json}, fn);
    CHECK(!r5.has_value());

    // root 是数组不是 object
    auto r6 = Tools::reg(ToolInfo{"bad6", "", nlohmann::json::array()}, fn);
    CHECK(!r6.has_value());

    // 拒绝的工具不应出现在注册表
    for (auto const* bad_name : {"bad1", "bad2", "bad3", "bad4", "bad5", "bad6"})
        CHECK(!Tools::get(bad_name).has_value());
}

// ═══════════════════════════════════════════════════════ //
//  整数边界：无符号范围 / 大 uint64 / 整值浮点            //
// ═══════════════════════════════════════════════════════ //

struct WithUnsigned
{
    [[= Desc("32位无符号数")]] std::uint32_t small_value;
    [[= Desc("64位无符号数")]] std::uint64_t big_value;
};

TEST_CASE("assign_from_json<WithUnsigned> negative rejected")
{
    auto r = agent::assign_from_json<WithUnsigned>({
        {"small_value", -1}, {"big_value", 1}
    });
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::InvalidArgs);
}

TEST_CASE("assign_from_json<WithUnsigned> max uint64 accepted")
{
    auto r = agent::assign_from_json<WithUnsigned>({
        {"small_value", 7}, {"big_value", 18446744073709551615ull}
    });
    REQUIRE(r.has_value());
    CHECK(r->small_value == 7);
    CHECK(r->big_value == 18446744073709551615ull);
}

TEST_CASE("assign_from_json<WithUnsigned> uint32 overflow rejected")
{
    auto r = agent::assign_from_json<WithUnsigned>({
        {"small_value", 4294967296ull}, {"big_value", 1}
    });
    CHECK(!r.has_value());
}

TEST_CASE("assign_from_json integer accepts whole-number float")
{
    // LLM 偶尔把整数写成 3.0，数值为整数即接受
    auto r = agent::assign_from_json<PrimitiveTypes>({
        {"name", "x"}, {"count", 3.0}, {"ratio", 1.0}, {"enabled", true}
    });
    REQUIRE(r.has_value());
    CHECK(r->count == 3);
}

TEST_CASE("assign_from_json integer rejects fractional float")
{
    auto r = agent::assign_from_json<PrimitiveTypes>({
        {"name", "x"}, {"count", 3.5}, {"ratio", 1.0}, {"enabled", true}
    });
    CHECK(!r.has_value());
}

TEST_CASE("assign_from_json root not object")
{
    auto r = agent::assign_from_json<PrimitiveTypes>(nlohmann::json::array());
    CHECK(!r.has_value());
    if (!r.has_value()) CHECK(r.error().code == Errc::InvalidArgs);
}

TEST_CASE("Tools::exec schema validation accepts whole-number float as integer")
{
    auto fn = [](nlohmann::json args) -> Result<std::string> {
        return std::to_string(args["n"].template get<std::int64_t>());
    };
    auto registered = Tools::reg(ToolInfo{"int_probe", "整数探针",
        R"({"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]})"_json},
        std::move(fn));
    REQUIRE(registered.has_value());

    auto whole = Tools::exec("int_probe", {{"n", 3.0}});
    REQUIRE(whole.has_value());
    CHECK(*whole == "3");

    auto fractional = Tools::exec("int_probe", {{"n", 3.5}});
    CHECK(!fractional.has_value());
    if (!fractional.has_value()) CHECK(fractional.error().code == Errc::InvalidArgs);
}

// ═══════════════════════════════════════════════════════ //
//  深层容器内 optional<Struct> 的 null 类型传播           //
// ═══════════════════════════════════════════════════════ //

struct WithDeepOptStruct
{
    [[= Desc("深网格")]] std::vector<std::vector<std::optional<Coordinate>>> grid;
};

TEST_CASE("schema_of WithDeepOptStruct keeps null in deeply nested optional object")
{
    auto j = agent::schema_of<WithDeepOptStruct>();
    CHECK(j["properties"]["grid"]["type"] == "array");
    CHECK(j["properties"]["grid"]["items"]["type"] == "array");
    CHECK(j["properties"]["grid"]["items"]["items"]["type"].is_array());
    CHECK(j["properties"]["grid"]["items"]["items"]["type"][0] == "object");
    CHECK(j["properties"]["grid"]["items"]["items"]["type"][1] == "null");
}

TEST_CASE("assign_from_json<WithDeepOptStruct> null elements in nested grid")
{
    auto r = agent::assign_from_json<WithDeepOptStruct>({
        {"grid", nlohmann::json::array({
            nlohmann::json::array({
                nlohmann::json{{"x", 1.0}, {"y", 2.0}},
                nullptr
            })
        })}
    });
    REQUIRE(r.has_value());
    REQUIRE(r->grid.size() == 1);
    REQUIRE(r->grid[0].size() == 2);
    REQUIRE(r->grid[0][0].has_value());
    CHECK(r->grid[0][0]->x == doctest::Approx(1.0));
    CHECK(!r->grid[0][1].has_value());
}

// ═══════════════════════════════════════════════════════ //
//  ArgsCheck::Tool：跳过 exec 预校验，工具自校验          //
// ═══════════════════════════════════════════════════════ //

TEST_CASE("Tools::reg ArgsCheck::Tool skips schema pre-validation")
{
    auto fn = [](nlohmann::json args) -> Result<std::string> {
        // 自校验工具：错误信息带独特标记，证明调用真的到达了 fn
        if (!args.contains("must"))
            return std::unexpected(Error{Errc::InvalidArgs, "checked by tool itself"});
        return "tool ok";
    };
    auto registered = Tools::reg(
        ToolInfo{"self_check", "自校验工具", R"({
            "type":"object",
            "properties":{"must":{"type":"string"}},
            "required":["must"]
        })"_json},
        std::move(fn),
        ArgsCheck::Tool);
    REQUIRE(registered.has_value());

    // 缺 required 字段：Schema 模式会在 exec 拦下，Tool 模式应到达 fn
    auto missing = Tools::exec("self_check", nlohmann::json::object());
    CHECK(!missing.has_value());
    if (!missing.has_value())
        CHECK(missing.error().message == "checked by tool itself");

    auto ok = Tools::exec("self_check", {{"must", "x"}});
    REQUIRE(ok.has_value());
    CHECK(*ok == "tool ok");
}

// ═══════════════════════════════════════════════════════ //
//  list 排序确定性 + 跨 TU 静态注册                       //
// ═══════════════════════════════════════════════════════ //

TEST_CASE("Tools::list sorted by name")
{
    auto tools = Tools::list();
    REQUIRE(tools.size() >= 2);
    for (std::size_t i = 1; i < tools.size(); ++i)
        CHECK(tools[i - 1].name < tools[i].name);
}

TEST_CASE("cross-TU explicit instantiation registers tools")
{
    // MultiTuAlpha / MultiTuBeta 定义在 test_multi_tu_tool_a/b.cpp，
    // 本 TU 没有任何符号引用它们——注册仍应生效
    auto alpha = Tools::exec("MultiTuAlpha", {{"text", "hi"}});
    REQUIRE(alpha.has_value());
    CHECK(*alpha == "alpha:hi");

    auto beta = Tools::exec("MultiTuBeta", {{"text", "yo"}});
    REQUIRE(beta.has_value());
    CHECK(*beta == "beta:yo");
}

// ═══════════════════════════════════════════════════════ //
//  运行时注册：复杂嵌套 schema 良构校验 + 深层参数校验    //
// ═══════════════════════════════════════════════════════ //

TEST_CASE("Tools::reg runtime complex nested schema end-to-end")
{
    // 手写深嵌套 schema：object → array<object>（长度约束）→ 嵌套 object
    // 覆盖 enum、["string","null"]、多层 required
    auto fn = [](nlohmann::json args) -> Result<std::string> {
        return std::to_string(args["orders"].size());
    };
    auto registered = Tools::reg(ToolInfo{"submit_orders", "提交订单批次", R"({
        "type": "object",
        "properties": {
            "batch_name": {"type": "string"},
            "orders": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "status": {"type": "string", "enum": ["pending", "shipped"]},
                        "note": {"type": ["string", "null"]},
                        "dimensions": {
                            "type": "object",
                            "properties": {
                                "width": {"type": "number"},
                                "height": {"type": "number"}
                            },
                            "required": ["width", "height"]
                        }
                    },
                    "required": ["id", "status", "dimensions"]
                }
            }
        },
        "required": ["batch_name", "orders"]
    })"_json}, std::move(fn));
    REQUIRE(registered.has_value());

    // ── 合法：深层全部正确（含 null note、integer 充当 number）── //
    auto ok = Tools::exec("submit_orders", R"({
        "batch_name": "b1",
        "orders": [
            {"id": 1, "status": "pending", "note": null,
             "dimensions": {"width": 1.5, "height": 2.0}},
            {"id": 2, "status": "shipped", "note": "快",
             "dimensions": {"width": 3, "height": 4}}
        ]
    })"_json);
    REQUIRE(ok.has_value());
    CHECK(*ok == "2");

    // ── 深层类型错：orders[0].dimensions.width 是字符串 ── //
    auto bad_deep_type = Tools::exec("submit_orders", R"({
        "batch_name": "b", "orders": [
            {"id": 1, "status": "pending",
             "dimensions": {"width": "wide", "height": 2.0}}
        ]})"_json);
    CHECK(!bad_deep_type.has_value());
    if (!bad_deep_type.has_value()) {
        CHECK(bad_deep_type.error().code == Errc::InvalidArgs);
        // 错误信息应定位到深层路径
        CHECK(bad_deep_type.error().message.find("width") != std::string::npos);
    }

    // ── 深层缺 required：dimensions 缺 height ── //
    auto missing_deep = Tools::exec("submit_orders", R"({
        "batch_name": "b", "orders": [
            {"id": 1, "status": "pending", "dimensions": {"width": 1.0}}
        ]})"_json);
    CHECK(!missing_deep.has_value());
    if (!missing_deep.has_value()) CHECK(missing_deep.error().code == Errc::InvalidArgs);

    // ── 枚举值非法：status 不在 enum 列表 ── //
    auto bad_enum = Tools::exec("submit_orders", R"({
        "batch_name": "b", "orders": [
            {"id": 1, "status": "lost",
             "dimensions": {"width": 1.0, "height": 2.0}}
        ]})"_json);
    CHECK(!bad_enum.has_value());
    if (!bad_enum.has_value()) CHECK(bad_enum.error().code == Errc::InvalidArgs);

    // ── 数组长度违规：空数组（minItems 1）与 4 个元素（maxItems 3）── //
    auto empty_orders = Tools::exec("submit_orders",
        R"({"batch_name": "b", "orders": []})"_json);
    CHECK(!empty_orders.has_value());

    nlohmann::json too_many = R"({"batch_name": "b", "orders": []})"_json;
    for (int i = 0; i < 4; ++i)
        too_many["orders"].push_back(R"({"id": 1, "status": "pending",
            "dimensions": {"width": 1.0, "height": 2.0}})"_json);
    auto overflow = Tools::exec("submit_orders", std::move(too_many));
    CHECK(!overflow.has_value());

    // ── 数组元素类型错：字符串代替 object ── //
    auto bad_element = Tools::exec("submit_orders",
        R"({"batch_name": "b", "orders": ["not an order"]})"_json);
    CHECK(!bad_element.has_value());

    // ── 深层 null 于不允许 null 的字段：dimensions 为 null ── //
    auto null_deep = Tools::exec("submit_orders", R"({
        "batch_name": "b", "orders": [
            {"id": 1, "status": "pending", "dimensions": null}
        ]})"_json);
    CHECK(!null_deep.has_value());

    // ── nullable 字段真的接受两种形态：null 与字符串（合法例已含）── //
    auto note_string = Tools::exec("submit_orders", R"({
        "batch_name": "b", "orders": [
            {"id": 1, "status": "pending", "note": "备注",
             "dimensions": {"width": 1.0, "height": 2.0}}
        ]})"_json);
    REQUIRE(note_string.has_value());
    // note 类型错（数字）仍要拦
    auto note_number = Tools::exec("submit_orders", R"({
        "batch_name": "b", "orders": [
            {"id": 1, "status": "pending", "note": 42,
             "dimensions": {"width": 1.0, "height": 2.0}}
        ]})"_json);
    CHECK(!note_number.has_value());
}

TEST_CASE("Tools::reg rejects deeply nested malformed schemas")
{
    auto fn = [](nlohmann::json) -> Result<std::string> { return ""; };

    // 深层（items.properties 内）type 是非法类型名
    auto r1 = Tools::reg(ToolInfo{"deep_bad1", "", R"({
        "type": "object",
        "properties": {
            "a": {"type": "array", "items": {
                "type": "object",
                "properties": {"b": {"type": "not_a_type"}}
            }}
        }})"_json}, fn);
    CHECK(!r1.has_value());
    if (!r1.has_value()) CHECK(r1.error().code == Errc::InvalidArgs);

    // 深层 required 引用未声明字段
    auto r2 = Tools::reg(ToolInfo{"deep_bad2", "", R"({
        "type": "object",
        "properties": {
            "a": {"type": "object",
                  "properties": {"x": {"type": "string"}},
                  "required": ["misspelled"]}
        }})"_json}, fn);
    CHECK(!r2.has_value());

    // 深层 type 数组里混入非字符串
    auto r3 = Tools::reg(ToolInfo{"deep_bad3", "", R"({
        "type": "object",
        "properties": {
            "a": {"type": "array", "items": {"type": ["string", 7]}}
        }})"_json}, fn);
    CHECK(!r3.has_value());

    // 深层 minItems 为负数
    auto r4 = Tools::reg(ToolInfo{"deep_bad4", "", R"({
        "type": "object",
        "properties": {
            "a": {"type": "array", "minItems": -2, "items": {"type": "integer"}}
        }})"_json}, fn);
    CHECK(!r4.has_value());

    // 拒绝的工具不应出现在注册表
    for (auto const* bad_name : {"deep_bad1", "deep_bad2", "deep_bad3", "deep_bad4"})
        CHECK(!Tools::get(bad_name).has_value());
}

// ═══════════════════════════════════════════════════════ //
//  无注解参数 / 多注解按类型过滤                          //
// ═══════════════════════════════════════════════════════ //

struct NoDescInput
{
    std::string          plain_field;             // 无注解
    [[= Desc("有描述")]] int described_field = 5; // 有注解
    std::optional<bool>  flag;                    // 无注解 optional
};

TEST_CASE("schema_of allows fields without Desc annotation")
{
    auto j = agent::schema_of<NoDescInput>();
    CHECK(j["properties"]["plain_field"]["type"] == "string");
    CHECK(!j["properties"]["plain_field"].contains("description"));
    CHECK(j["properties"]["described_field"]["description"] == "有描述");
    CHECK(j["properties"]["described_field"]["default"] == 5);
    CHECK(!j["properties"]["flag"].contains("description"));
    CHECK(j["required"].size() == 1);
    CHECK(j["required"][0] == "plain_field");
}

TEST_CASE("assign_from_json works without annotations")
{
    auto r = agent::assign_from_json<NoDescInput>({{"plain_field", "x"}});
    REQUIRE(r.has_value());
    CHECK(r->plain_field == "x");
    CHECK(r->described_field == 5);
    CHECK(!r->flag.has_value());
}

// 无 Desc 注解的工具 struct：描述为空串，注册照常
struct NoDescTool
{
    using params_type = NoDescInput;
    static Result<std::string> invoke(params_type const& p)
    {
        return p.plain_field;
    }
};
template struct agent::ToolBase<NoDescTool>;

TEST_CASE("ToolBase registers tool without Desc annotation")
{
    auto info = Tools::get("NoDescTool");
    REQUIRE(info.has_value());
    CHECK(info->description.empty());

    auto r = Tools::exec("NoDescTool", {{"plain_field", "raw"}});
    REQUIRE(r.has_value());
    CHECK(*r == "raw");
}

// 多注解且 Desc 不在第一位：find_desc 按类型过滤必须找到它
// （盲取 annots[0] 的实现会在这里 extract<DescArg> 到错误类型）
struct OtherTag { int value; };

struct MixedAnnotInput
{
    [[= OtherTag{1}]] [[= Desc("次序在后")]] std::string field;
};

TEST_CASE("find_desc picks Desc among multiple annotations")
{
    auto j = agent::schema_of<MixedAnnotInput>();
    CHECK(j["properties"]["field"]["description"] == "次序在后");
}

