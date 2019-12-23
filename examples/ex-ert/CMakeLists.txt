cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

include(CheckLanguage)

# travis has limited resources so running ERT takes a while
# if("$ENV{USER}" STREQUAL "travis" AND "$ENV{CONTINUOUS_INTEGRATION}" STREQUAL "true")
#    return()
# endif()

set(_USE_CUDA ON)
# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory")
    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT "CUDA" IN_LIST LANGUAGES OR NOT CMAKE_CUDA_COMPILER OR NOT TIMEMORY_USE_CUDA)
        set(_USE_CUDA OFF)
    endif()
endif()

project(timemory-ERT-Example LANGUAGES CXX)

option(USE_CUDA "Enable CUDA" ${_USE_CUDA})

set(timemory_FIND_COMPONENTS_INTERFACE timemory-ert-example)
set(COMPONENTS cxx compile-options vector analysis-tools OPTIONAL_COMPONENTS arch mpi)

if(USE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        set(_USE_CUDA OFF)
    endif()
endif()

if(USE_CUDA)
    list(APPEND COMPONENTS cuda)
endif()

find_package(timemory REQUIRED COMPONENTS ${COMPONENTS} OPTIONAL_COMPONENTS cupti)

add_executable(ex_ert ex_ert.cpp)
target_link_libraries(ex_ert timemory-ert-example)
install(TARGETS ex_ert DESTINATION bin)

if(USE_CUDA)
    set_source_files_properties(ex_ert.cpp PROPERTIES
        LANGUAGE CUDA
        LINKER_LANGUAGE CUDA)
    target_compile_definitions(ex_ert PRIVATE USE_CUDA)
endif()
