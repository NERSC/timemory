cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

if(NOT TIMEMORY_BUILD_MALLOCP_LIBRARY OR NOT TIMEMORY_USE_GOTCHA OR
    NOT TARGET timemory::timemory-cxx-shared OR TIMEMORY_SKIP_BUILD)
    return()
endif()

project(timemory-mallocp-tool)

add_library(timemory-mallocp-shared SHARED ${PROJECT_SOURCE_DIR}/timemory-mallocp.cpp)
add_library(timemory::timemory-mallocp-shared ALIAS timemory-mallocp-shared)

# public link targets
target_link_libraries(timemory-mallocp-shared PUBLIC
    timemory::timemory-headers
    timemory::timemory-cxx-shared
    timemory::timemory-gotcha)

# private link targets
target_link_libraries(timemory-mallocp-shared PRIVATE
    timemory::timemory-default-visibility
    timemory::timemory-compile-options)

# use rpath
set_target_properties(timemory-mallocp-shared PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH ON
    OUTPUT_NAME     timemory-mallocp
    VERSION         ${timemory_VERSION}
    SOVERSION       ${timemory_VERSION_MAJOR}.${timemory_VERSION_MINOR})

# installation
install(TARGETS timemory-mallocp-shared
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    OPTIONAL)
