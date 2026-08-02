# Examples：链接 agent_core 静态库。
# 工具定义（ToolBase<T> 显式实例化）必须留在示例 .cpp 这个 TU、直接编译进
# 可执行——放进静态库会被链接器丢弃（没有任何符号引用它的 .o），静态注册不触发。
# 参考 Botcraft 的 Examples 组织：宏统一设置 + 每示例 add_subdirectory。

# 示例 target 需要 -freflection（工具反射定义在示例 TU）
add_executable(deepseek_chat
    deepseek_chat.cpp
)

target_link_libraries(deepseek_chat PRIVATE agent_core)
target_compile_options(deepseek_chat PRIVATE -freflection)

set_target_properties(deepseek_chat PROPERTIES FOLDER Examples)

add_executable(agnes_chat
    agnes_chat.cpp
)

target_link_libraries(agnes_chat PRIVATE agent_core)
target_compile_options(agnes_chat PRIVATE -freflection)

set_target_properties(agnes_chat PROPERTIES FOLDER Examples)
