# 核心库（agent_core，不含 main）：LLM 调用层 + 工具系统 + 传输层。
# agent 可执行与 examples 均链接此库。
# 注意（PLAN.md 局限）：静态库会丢弃「链接器看不到符号引用」的 .o——
# 所以工具定义（ToolBase<T> 显式实例化）必须留在调用方 TU（main.cpp / 示例 .cpp），
# 不能放进本库，否则静态注册不触发。
add_library(agent_core STATIC
    tools/tools.cpp
    core/http_client.cpp
    llm/model_registry.cpp
    llm/engine/openai_completions.cpp
    llm/engine/image_support.cpp
    llm/providers/openai.cpp
    llm/providers/deepseek.cpp
    llm/providers/agnes.cpp
    llm/providers/gemini.cpp
    llm/providers/anthropic.cpp
)

target_include_directories(agent_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(agent_core PRIVATE -freflection)

target_link_libraries(agent_core
    PUBLIC
        asio
        nlohmann_json
    PRIVATE
        CURL::libcurl
        Threads::Threads
        OpenSSL::SSL
        OpenSSL::Crypto
        ZLIB::ZLIB
        brotlidec
        brotlicommon
)

if(WIN32)
    target_link_libraries(agent_core PRIVATE ws2_32)
endif()

# 主可执行（工具演示：Greet 等工具定义在 main.cpp TU，直接编译进本可执行）
add_executable(agent
    main.cpp
)

target_link_libraries(agent PRIVATE agent_core)
target_compile_options(agent PRIVATE -freflection)
