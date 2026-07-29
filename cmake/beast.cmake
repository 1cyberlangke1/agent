add_library(beast INTERFACE IMPORTED)
set_property(TARGET beast PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/beast/include")
target_link_libraries(beast INTERFACE asio)
