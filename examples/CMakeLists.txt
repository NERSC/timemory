cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0048 NEW)

#----------------------------------------------------------------------------------------#
#   TiMemory examples project
#
project(TiMemory-Examples C CXX)

#----------------------------------------------------------------------------------------#
#   Determine if standalone
#
set(STANDALONE ON)
if(NOT "${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set(STANDALONE OFF)
endif()

#----------------------------------------------------------------------------------------#
#   If we are building as a standalone project or as part of source/build tree
#
if(STANDALONE)
    set(TIMEMORY_SKIP_INTERFACE_LIBRARY ON)
    find_package(TiMemory REQUIRED)
    unset(TIMEMORY_SKIP_INTERFACE_LIBRARY)
    option(TIMEMORY_USE_MPI "Enable MPI usage" ON)
    if(TIMEMORY_USE_MPI)
        find_package(MPI)
    endif()
else()
    if(NOT TIMEMORY_BUILD_EXAMPLES)
        return()
    endif()
    find_package(MPI QUIET)
endif()

#----------------------------------------------------------------------------------------#
#   Add the examples
#
add_subdirectory(ex-optional)
if(TIMEMORY_BUILD_C OR TARGET timemory-c-shared)
    add_subdirectory(ex-c)
endif()
add_subdirectory(ex-cxx-tuple)
add_subdirectory(ex-cxx-overhead)
add_subdirectory(ex-cpu-roofline)

get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
if("CUDA" IN_LIST LANGUAGES AND CMAKE_CUDA_COMPILER)
    if(NOT DEFINED TIMEMORY_USE_CUDA OR TIMEMORY_USE_CUDA)
        add_subdirectory(ex-cuda-event)
    endif()
endif()
