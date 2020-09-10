cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

if(NOT TIMEMORY_BUILD_MPIP_LIBRARY OR NOT TIMEMORY_USE_MPI OR NOT TIMEMORY_USE_GOTCHA OR
    NOT TARGET timemory-cxx-shared OR TIMEMORY_SKIP_BUILD)
    return()
endif()

project(timemory-mpip-tool)

add_library(timemory-mpip-shared SHARED ${PROJECT_SOURCE_DIR}/timemory-mpip.cpp)
add_library(timemory::timemory-mpip-shared ALIAS timemory-mpip-shared)

# public link targets
target_link_libraries(timemory-mpip-shared PUBLIC
    timemory-headers
    timemory-cxx-shared
    timemory-mpi
    timemory-gotcha)

# private link targets
target_link_libraries(timemory-mpip-shared PRIVATE
    timemory::timemory-default-visibility
    timemory::timemory-compile-options)

# use rpath
set_target_properties(timemory-mpip-shared PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH ON
    OUTPUT_NAME     timemory-mpip
    VERSION         ${timemory_VERSION}
    SOVERSION       ${timemory_VERSION_MAJOR}.${timemory_VERSION_MINOR})

# installation
install(TARGETS timemory-mpip-shared DESTINATION ${CMAKE_INSTALL_LIBDIR})
