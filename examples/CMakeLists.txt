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
if(TIMEMORY_BUILD_C OR TIMEMORY_C_LIBRARY)
    add_subdirectory(ex-c)
endif()
add_subdirectory(ex-cxx-tuple)
add_subdirectory(ex-cxx-overhead)
add_subdirectory(ex-roofline)

get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
find_package(CUDA QUIET)
if("CUDA" IN_LIST LANGUAGES AND CUDA_FOUND)
    if(NOT DEFINED TIMEMORY_USE_CUDA OR TIMEMORY_USE_CUDA)
        add_subdirectory(ex-cuda-event)
    endif()
endif()

#add_subdirectory(ex-cxx-graph)
#add_subdirectory(ex-cxx)

#if(TIMEMORY_USE_MPI AND MPI_FOUND)
#    add_subdirectory(ex-cxx-mpi)
#endif()
