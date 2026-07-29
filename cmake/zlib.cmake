find_package(ZLIB QUIET)

if(NOT TARGET ZLIB::ZLIB)
    set(ZLIB_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/zlib/")
    set(ZLIB_BUILD_PATH "${CMAKE_CURRENT_BINARY_DIR}/3rdparty/zlib")

    file(GLOB RESULT "${ZLIB_BUILD_PATH}/install")
    list(LENGTH RESULT RES_LEN)
    if(RES_LEN EQUAL 0)
        message(STATUS "Can't find Zlib, cloning and building it from sources")

        file(GLOB RESULT "${ZLIB_SRC_PATH}")
        list(LENGTH RESULT RES_LEN)
        if(RES_LEN EQUAL 0)
            execute_process(COMMAND git submodule update --init -- 3rdparty/zlib WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
        endif()

        file(MAKE_DIRECTORY "${ZLIB_BUILD_PATH}")

        execute_process(
            COMMAND "${CMAKE_COMMAND}" "${ZLIB_SRC_PATH}" "-G" "${CMAKE_GENERATOR}" "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX=install" "-DCMAKE_POSITION_INDEPENDENT_CODE=ON" "-DZLIB_BUILD_SHARED=OFF"
            WORKING_DIRECTORY "${ZLIB_BUILD_PATH}")

        execute_process(COMMAND "${CMAKE_COMMAND}" "--build" "." "--target" "install" "--parallel" "2" "--config" "Release" WORKING_DIRECTORY "${ZLIB_BUILD_PATH}")
    endif()

    set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

    find_package(ZLIB QUIET NO_DEFAULT_PATH PATHS "${ZLIB_BUILD_PATH}/install")

    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()
