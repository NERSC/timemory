cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# this is for internal use
if(NOT TIMEMORY_BUILD_OMPT_LIBRARY OR NOT TIMEMORY_USE_OMPT OR TIMEMORY_SKIP_BUILD)
    return()
endif()

project(timemory-ompt-tool)

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
    add_library(timemory::${TARGET_NAME} ALIAS ${TARGET_NAME})

    # public link targets
    target_link_libraries(${TARGET_NAME} PUBLIC
        timemory-headers
        timemory-ompt
        timemory-cxx-${_LC_TYPE})

    # private link targets
    target_link_libraries(${TARGET_NAME} PRIVATE
        timemory::timemory-default-visibility
        timemory::timemory-compile-options)

    # use rpath
    set_target_properties(${TARGET_NAME} PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH ON
        OUTPUT_NAME     timemory-ompt
        VERSION         ${timemory_VERSION}
        SOVERSION       ${timemory_VERSION_MAJOR}.${timemory_VERSION_MINOR})

    # installation
    install(TARGETS ${TARGET_NAME} DESTINATION ${CMAKE_INSTALL_LIBDIR})

endforeach()
