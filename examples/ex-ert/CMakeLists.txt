cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory")
    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT "CUDA" IN_LIST LANGUAGES OR NOT CMAKE_CUDA_COMPILER OR NOT TIMEMORY_USE_CUDA)
        return()
    endif()
endif()

project(timemory-ERT-Example LANGUAGES CXX CUDA)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-ert-example)
set(COMPONENTS cuda compile-options arch vector mpi)

set_source_files_properties(test_ert.cpp PROPERTIES
    LANGUAGE CUDA
    LINKER_LANGUAGE CUDA)

find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})
add_executable(test_ert test_ert.cpp)
target_link_libraries(test_ert timemory-ert-example)
#set_target_properties(test_ert PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)
install(TARGETS test_ert DESTINATION bin)
