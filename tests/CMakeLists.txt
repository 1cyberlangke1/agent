if(AGENT_BUILD_TESTS)
    enable_testing()

    # T2 契约对比：dump build_params 供 compare_requests.py 与官方 SDK golden 对比
    option(AGENT_CONTRACT "dump build_params for T2 contract compare" OFF)

    # doctest 头文件路径
    set(DOCTEST_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/doctest)

    add_executable(test_agent
        main.cpp
        core/test_result.cpp
        core/test_http_client.cpp
        tools/test_tools.cpp
        tools/test_tools_concurrency.cpp
        tools/test_multi_tu_tool_a.cpp
        tools/test_multi_tu_tool_b.cpp
        llm/test_content.cpp
        llm/test_types.cpp
        llm/test_model.cpp
        llm/test_stream.cpp
        llm/test_openai.cpp
        llm/test_gemini.cpp
        llm/test_anthropic.cpp
        ${CMAKE_SOURCE_DIR}/src/tools/tools.cpp
        ${CMAKE_SOURCE_DIR}/src/llm/model_registry.cpp
        ${CMAKE_SOURCE_DIR}/src/core/http_client.cpp
        ${CMAKE_SOURCE_DIR}/src/llm/engine/openai_completions.cpp
        ${CMAKE_SOURCE_DIR}/src/llm/providers/openai.cpp
        ${CMAKE_SOURCE_DIR}/src/llm/providers/deepseek.cpp
        ${CMAKE_SOURCE_DIR}/src/llm/providers/gemini.cpp
        ${CMAKE_SOURCE_DIR}/src/llm/providers/anthropic.cpp
    )

    target_include_directories(test_agent
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${DOCTEST_INCLUDE_DIR}
    )

    target_link_libraries(test_agent
        PRIVATE
            asio
            CURL::libcurl
            nlohmann_json
            Threads::Threads
    )
    if(WIN32)
        target_link_libraries(test_agent PRIVATE ws2_32)
    endif()

    target_compile_options(test_agent PRIVATE -freflection)
    target_compile_definitions(test_agent PRIVATE
        AGENT_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
    )
    if(AGENT_CONTRACT)
        target_compile_definitions(test_agent PRIVATE
            AGENT_CONTRACT_OUT="${CMAKE_BINARY_DIR}/contract/out"
        )
    endif()
    add_test(NAME agent COMMAND test_agent)

    # 编译期拒绝测试：此文件应编译失败
    add_library(test_recursive_reject EXCLUDE_FROM_ALL tools/test_recursive_reject.cpp)
    target_include_directories(test_recursive_reject PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${DOCTEST_INCLUDE_DIR}
    )
    target_compile_options(test_recursive_reject PRIVATE -freflection)
    add_test(NAME recursive_reject
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
            --target test_recursive_reject --config $<CONFIG>
    )
    set_tests_properties(recursive_reject PROPERTIES WILL_FAIL TRUE)

    # T2 契约层：官方 SDK 镜像对照（.venv 缺失 → SKIP_RETURN_CODE=2 跳过）
    if(EXISTS ${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe)
        set(CONTRACT_PYTHON ${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe)
    elseif(EXISTS ${CMAKE_SOURCE_DIR}/.venv/bin/python)
        set(CONTRACT_PYTHON ${CMAKE_SOURCE_DIR}/.venv/bin/python)
    else()
        set(CONTRACT_PYTHON python)
    endif()
    add_test(NAME contract_mirror
        COMMAND ${CONTRACT_PYTHON} ${CMAKE_SOURCE_DIR}/tests/contract/run_mirror.py)
    set_tests_properties(contract_mirror PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME contract_compare
        COMMAND ${CONTRACT_PYTHON} ${CMAKE_SOURCE_DIR}/tests/contract/compare_requests.py
            ${CMAKE_BINARY_DIR}/contract/out ${CMAKE_BINARY_DIR}/contract/golden)
    set_tests_properties(contract_compare PROPERTIES SKIP_RETURN_CODE 2)
endif()
