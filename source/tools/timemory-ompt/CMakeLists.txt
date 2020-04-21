cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory")
    if(NOT TIMEMORY_USE_OMPT)
        return()
    endif()
endif()

project(timemory-ompt-tool)

# set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME ompt)

set(LIB_TYPES)
if(BUILD_SHARED_LIBS AND NOT TIMEMORY_SKIP_BUILD)
    list(APPEND LIB_TYPES SHARED)
endif()

if(BUILD_STATIC_LIBS AND NOT TIMEMORY_SKIP_BUILD)
    list(APPEND LIB_TYPES STATIC)
endif()

foreach(_TYPE ${LIB_TYPES})
    string(TOLOWER "${_TYPE}" _LC_TYPE)
    if(NOT TARGET timemory-cxx-${_LC_TYPE})
        continue()
    endif()

    set(TARGET_NAME timemory-ompt-${_LC_TYPE})

    add_library(${TARGET_NAME} ${_TYPE} ${PROJECT_SOURCE_DIR}/timemory-ompt.cpp)

    # public link targets
    target_link_libraries(${TARGET_NAME}
        PUBLIC
            timemory-headers
            timemory-cxx-shared
            timemory-ompt
            timemory-cxx-${_LC_TYPE})

    # private link targets
    target_link_libraries(${TARGET_NAME}
        PRIVATE
            timemory-compile-options
            timemory-arch
            timemory-vector)

    # use rpath
    set_target_properties(${TARGET_NAME} PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH ON
        OUTPUT_NAME timemory-ompt)

    # installation
    install(TARGETS ${TARGET_NAME} DESTINATION ${CMAKE_INSTALL_LIBDIR})

endforeach()