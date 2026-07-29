add_library(nlohmann_json INTERFACE IMPORTED)
set_property(TARGET nlohmann_json PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/nlohmann_json/single_include")
