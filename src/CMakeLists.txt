add_executable(agent main.cpp tools.cpp)

target_include_directories(agent
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(agent PRIVATE -freflection)

target_link_libraries(agent
    PRIVATE
        asio
        beast
        nlohmann_json
        Threads::Threads
        OpenSSL::SSL
        OpenSSL::Crypto
        ZLIB::ZLIB
        brotlidec
        brotlicommon
)
