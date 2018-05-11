cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

if(CMAKE_VERSION VERSION_LESS 3.1.3)
    set(GOOD_CMAKE OFF)
else(CMAKE_VERSION VERSION_LESS 3.1.3)
    set(GOOD_CMAKE ON)
    cmake_policy(SET CMP0048 NEW)
endif(CMAKE_VERSION VERSION_LESS 3.1.3)
cmake_policy(SET CMP0042 NEW)

#------------------------------------------------------------------------------#
#   TiMemory examples project
#
project(TiMemory-Examples C CXX)

if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set(STANDALONE ON)
else("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set(STANDALONE OFF)
endif("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")

#------------------------------------------------------------------------------#
#   If we are building as a standalone project or as part of source/build tree
#
if(STANDALONE)
    option(TIMEMORY_USE_MPI "Enable MPI usage" ON)
    if(TIMEMORY_USE_MPI)
        find_package(MPI)
    endif(TIMEMORY_USE_MPI)
else(STANDALONE)
    if(NOT TIMEMORY_BUILD_EXAMPLES)
        return()
    endif(NOT TIMEMORY_BUILD_EXAMPLES)
endif(STANDALONE)

#------------------------------------------------------------------------------#
#   Add the examples
#
add_subdirectory(ex-c)
add_subdirectory(ex-cxx)
add_subdirectory(ex-cxx-overhead)
add_subdirectory(ex-cxx-total)
if(TIMEMORY_USE_MPI AND MPI_FOUND)
    add_subdirectory(ex-cxx-mpi)
endif(TIMEMORY_USE_MPI AND MPI_FOUND)


#------------------------------------------------------------------------------#
#   Configuration for build tree
#
if(NOT STANDALONE)
#    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory
#        ${CMAKE_BINARY_DIR}/examples
#        ERROR_QUIET)

#    set(example_dirs c cxx cxx-mpi cxx-overhead cxx-total)
#    foreach(example_dir ${example_dirs})
#        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
#            ${CMAKE_CURRENT_LIST_DIR}/ex-${example_dir}
#            ${CMAKE_BINARY_DIR}/examples/ex-${example_dir}
#            ERROR_QUIET)
#    endforeach(example_dir ${example_dirs})

#    set_property(GLOBAL APPEND PROPERTY
#        BUILDTREE_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR})

endif(NOT STANDALONE)


