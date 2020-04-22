cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

if(NOT TIMEMORY_BUILD_MPIP OR NOT TIMEMORY_USE_MPI OR NOT TIMEMORY_USE_GOTCHA OR
    NOT TARGET timemory-cxx-shared OR TIMEMORY_SKIP_BUILD)
    return()
endif()

project(timemory-mpip-tool)

add_library(timemory-mpip SHARED ${PROJECT_SOURCE_DIR}/timemory-mpip.cpp)

# public link targets
target_link_libraries(timemory-mpip
    PUBLIC
        timemory-headers
        timemory-cxx-shared
        timemory-mpi
        timemory-gotcha)

# private link targets
target_link_libraries(timemory-mpip
    PRIVATE
        timemory-compile-options
        timemory-arch
        timemory-vector)

# use rpath
set_target_properties(timemory-mpip PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

# installation
install(TARGETS timemory-mpip DESTINATION ${CMAKE_INSTALL_LIBDIR})
