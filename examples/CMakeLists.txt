cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0048 NEW)

#------------------------------------------------------------------------------#
#   TiMemory examples project
#
project(TiMemory-Examples C CXX)

#------------------------------------------------------------------------------#
#   Determine if standalone
#
set(STANDALONE ON)
if(NOT "${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set(STANDALONE OFF)
endif()

#------------------------------------------------------------------------------#
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
endif()

#------------------------------------------------------------------------------#
#   Add the examples
#
add_subdirectory(ex-tuple)
#add_subdirectory(ex-c)
#add_subdirectory(ex-cxx)
#add_subdirectory(ex-cxx-total)
#add_subdirectory(ex-cxx-graph)
add_subdirectory(ex-cxx-overhead)
if(TIMEMORY_USE_MPI AND MPI_FOUND)
#    add_subdirectory(ex-cxx-mpi)
endif(TIMEMORY_USE_MPI AND MPI_FOUND)
