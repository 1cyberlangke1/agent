if(AGENT_BUILD_TESTS)
    enable_testing()

    # doctest 头文件路径
    set(DOCTEST_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/doctest)

    add_executable(test_agent
        main.cpp
        test_result.cpp
        test_content.cpp
        test_tools.cpp
        test_tools_concurrency.cpp
        test_multi_tu_tool_a.cpp
        test_multi_tu_tool_b.cpp
        ${CMAKE_SOURCE_DIR}/src/tools.cpp
    )

    target_include_directories(test_agent
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${DOCTEST_INCLUDE_DIR}
    )

    target_link_libraries(test_agent
        PRIVATE
            nlohmann_json
            Threads::Threads
    )

    target_compile_options(test_agent PRIVATE -freflection)
    add_test(NAME agent COMMAND test_agent)

    # 编译期拒绝测试：此文件应编译失败
    add_library(test_recursive_reject EXCLUDE_FROM_ALL test_recursive_reject.cpp)
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
endif()
