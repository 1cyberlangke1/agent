# libcurl —— HTTP 传输层（静态库，仅 HTTP/HTTPS）
#
# TLS 后端：Windows 用 schannel（系统 TLS + 系统证书库，免 OpenSSL 证书
# 导入坑）；其他平台用 OpenSSL。
# 压缩：zlib + brotli 接入（自动解压 gzip/deflate/br；引擎侧
# CURLOPT_ACCEPT_ENCODING="" 声明全部支持的编码）。
# 必须在 cmake/zlib.cmake、cmake/brotli.cmake 之后 include——
# 复用它们的 find 结果 / 本地构建产物。

file(GLOB RESULT "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/curl/CMakeLists.txt")
list(LENGTH RESULT RES_LEN)
if(RES_LEN EQUAL 0)
    message(STATUS "curl not found, cloning it...")
    execute_process(COMMAND git submodule update --init -- 3rdparty/curl
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

set(BUILD_CURL_EXE OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_LIBCURL_DOCS OFF CACHE BOOL "" FORCE)
set(BUILD_MISC_DOCS OFF CACHE BOOL "" FORCE)
set(ENABLE_CURL_MANUAL OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HTTP_ONLY ON CACHE BOOL "" FORCE)
set(CURL_DISABLE_INSTALL ON CACHE BOOL "" FORCE)
set(CURL_USE_LIBPSL OFF CACHE BOOL "" FORCE)
set(CURL_USE_LIBSSH2 OFF CACHE BOOL "" FORCE)
set(USE_NGHTTP2 OFF CACHE BOOL "" FORCE)
set(USE_LIBIDN2 OFF CACHE BOOL "" FORCE)

# zlib：cmake/zlib.cmake 已 find（cache 命中）或本地预构建，
# curl 内部 find_package(ZLIB) 直接复用同一结果。
set(CURL_ZLIB ON CACHE STRING "" FORCE)

# brotli：curl 的 FindBrotli 三变量都已定义时跳过 pkg-config/find。
# cmake/brotli.cmake 本地构建分支创建了 brotlidec IMPORTED target，
# 产物在 build/3rdparty/brotli/install；pkg-config 命中系统 brotli 时
# 无需 hint（curl 的 FindBrotli 同样能 pkg-config 命中）。
set(CURL_BROTLI ON CACHE BOOL "" FORCE)
if(TARGET brotlidec)
    set(AGENT_BROTLI_INSTALL "${CMAKE_CURRENT_BINARY_DIR}/3rdparty/brotli/install")
    file(GLOB AGENT_BROTLIDEC_LIB "${AGENT_BROTLI_INSTALL}/lib/*brotlidec*")
    file(GLOB AGENT_BROTLICOMMON_LIB "${AGENT_BROTLI_INSTALL}/lib/*brotlicommon*")
    set(BROTLI_INCLUDE_DIR "${AGENT_BROTLI_INSTALL}/include" CACHE PATH "" FORCE)
    set(BROTLIDEC_LIBRARY "${AGENT_BROTLIDEC_LIB}" CACHE FILEPATH "" FORCE)
    set(BROTLICOMMON_LIBRARY "${AGENT_BROTLICOMMON_LIB}" CACHE FILEPATH "" FORCE)
endif()

set(CURL_ZSTD OFF CACHE BOOL "" FORCE)
if(WIN32)
    set(CURL_USE_SCHANNEL ON CACHE BOOL "" FORCE)
else()
    set(CURL_USE_OPENSSL ON CACHE BOOL "" FORCE)
endif()

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/curl" EXCLUDE_FROM_ALL)
