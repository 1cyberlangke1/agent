cmake_minimum_required(VERSION 3.26)
project(agent VERSION 0.1.0 LANGUAGES CXX)

cmake_policy(SET CMP0074 NEW)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AGENT_BUILD_EXAMPLES "Build example targets" ON)
option(AGENT_BUILD_TESTS "Build tests" OFF)

# 文档：doxygen 存在时提供 `--target doc` 生成 API 文档（build/docs/html）。
# WARN_AS_ERROR=YES（Doxyfile）强制全部公共注释，缺注释即构建失败。
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "生成 Doxygen 文档 → build/docs/html/index.html"
    )
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/asio.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/openssl.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/zlib.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/brotli.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/nlohmann_json.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/curl.cmake")

find_package(Threads REQUIRED)

add_subdirectory(src)

if(AGENT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(AGENT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
