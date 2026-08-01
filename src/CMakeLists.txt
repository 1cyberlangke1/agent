add_executable(agent
    main.cpp
    tools/tools.cpp
    core/http_client.cpp
    llm/model_registry.cpp
    llm/engine/openai_completions.cpp
    llm/providers/openai.cpp
    llm/providers/deepseek.cpp
    llm/providers/gemini.cpp
)

target_include_directories(agent
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(agent PRIVATE -freflection)

target_link_libraries(agent
    PRIVATE
        asio
        CURL::libcurl
        nlohmann_json
        Threads::Threads
        OpenSSL::SSL
        OpenSSL::Crypto
        ZLIB::ZLIB
        brotlidec
        brotlicommon
)
