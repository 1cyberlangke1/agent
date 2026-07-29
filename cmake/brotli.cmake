find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(BROTLI libbrotlidec QUIET)
endif()

if(NOT BROTLI_FOUND)
    set(BROTLI_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/brotli/")
    set(BROTLI_BUILD_PATH "${CMAKE_CURRENT_BINARY_DIR}/3rdparty/brotli")

    file(GLOB RESULT "${BROTLI_BUILD_PATH}/install")
    list(LENGTH RESULT RES_LEN)
    if(RES_LEN EQUAL 0)
        message(STATUS "Can't find Brotli, building it locally...")

        file(GLOB RESULT "${BROTLI_SRC_PATH}")
        list(LENGTH RESULT RES_LEN)
        if(RES_LEN EQUAL 0)
            execute_process(COMMAND git submodule update --init -- 3rdparty/brotli WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
        endif()

        file(MAKE_DIRECTORY "${BROTLI_BUILD_PATH}")

        execute_process(
            COMMAND "${CMAKE_COMMAND}" "${BROTLI_SRC_PATH}" "-G" "${CMAKE_GENERATOR}" "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX=install" "-DBUILD_SHARED_LIBS=OFF" "-DBROTLI_BUILD_TOOLS=OFF" "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
            WORKING_DIRECTORY "${BROTLI_BUILD_PATH}")

        execute_process(COMMAND "${CMAKE_COMMAND}" "--build" "." "--target" "install" "--parallel" "2" "--config" "Release" WORKING_DIRECTORY "${BROTLI_BUILD_PATH}")
    endif()

    file(GLOB brotlidec_lib "${BROTLI_BUILD_PATH}/install/lib/*brotlidec*")
    file(GLOB brotlicommon_lib "${BROTLI_BUILD_PATH}/install/lib/*brotlicommon*")
    add_library(brotlidec STATIC IMPORTED)
    set_property(TARGET brotlidec PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${BROTLI_BUILD_PATH}/install/include")
    set_target_properties(brotlidec PROPERTIES IMPORTED_LOCATION "${brotlidec_lib}")
    add_library(brotlicommon STATIC IMPORTED)
    set_target_properties(brotlicommon PROPERTIES IMPORTED_LOCATION "${brotlicommon_lib}")
endif()
