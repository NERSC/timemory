cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory")
    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT "CUDA" IN_LIST LANGUAGES OR NOT CMAKE_CUDA_COMPILER OR NOT TIMEMORY_USE_CUDA)
        return()
    endif()
endif()

project(timemory-CUDA-Examples LANGUAGES C CXX CUDA)

option(USE_PAPI "Enable PAPI" OFF)
option(USE_CUPTI "Enable CUPTI" OFF)
option(USE_LOGGING "Enable logging in debug mode" OFF)

add_library(logging INTERFACE)
target_compile_definitions(logging INTERFACE
    $<IF:$<BOOL:${USE_LOGGING}>,$<$<CONFIG:Debug>:DEBUG>,>)

set(COMPONENTS)
if(USE_CUPTI)
    list(APPEND COMPONENTS cupti)
endif()
if(USE_PAPI)
    list(APPEND COMPONENTS papi)
endif()

set(timemory_FIND_COMPONENTS_INTERFACE timemory-cuda-event-example)
find_package(timemory REQUIRED
    COMPONENTS cuda cuda-compiler threading compile-options analysis-tools cudart
    OPTIONAL_COMPONENTS
    nvtx ${COMPONENTS})

add_executable(ex_cuda_event ex_cuda_event.cu)
target_link_libraries(ex_cuda_event PRIVATE timemory-cuda-event-example logging)
install(TARGETS ex_cuda_event DESTINATION bin OPTIONAL)

add_executable(ex_kernel_instrument ex_kernel_instrument.cu)
target_link_libraries(ex_kernel_instrument PRIVATE
    timemory::timemory-headers
    timemory::timemory-compile-options
    timemory::timemory-cuda
    timemory::timemory-cudart
    timemory::timemory-cuda-compiler
    timemory::timemory-cxx)
install(TARGETS ex_kernel_instrument DESTINATION bin OPTIONAL)

add_executable(ex_kernel_instrument_v2 ex_kernel_instrument_v2.cu)
target_link_libraries(ex_kernel_instrument_v2 PRIVATE
    timemory::timemory-headers
    timemory::timemory-compile-options
    timemory::timemory-cuda
    timemory::timemory-cudart
    timemory::timemory-cuda-compiler
    timemory::timemory-cxx)
install(TARGETS ex_kernel_instrument_v2 DESTINATION bin OPTIONAL)
