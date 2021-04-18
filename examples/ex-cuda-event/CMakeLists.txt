cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory")
    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT "CUDA" IN_LIST LANGUAGES OR NOT CMAKE_CUDA_COMPILER OR NOT TIMEMORY_USE_CUDA)
        return()
    endif()
endif()

project(timemory-CUDA-Event-Example LANGUAGES C CXX CUDA)

set(EXE_NAME ex_cuda_event)
set(COMPONENTS cuda threading compile-options analysis-tools cudart OPTIONAL_COMPONENTS nvtx)

option(USE_CUPTI "Enable CUPTI" OFF)
if(USE_CUPTI)
    list(APPEND COMPONENTS cupti)
endif()

option(USE_PAPI "Enable PAPI" OFF)
if(USE_PAPI)
    list(APPEND COMPONENTS papi)
endif()

option(USE_LOGGING "Enable logging in debug mode" OFF)
add_library(logging INTERFACE)
if(USE_LOGGING)
    target_compile_definitions(logging INTERFACE $<$<CONFIG:Debug>:DEBUG>)
endif()

set(timemory_FIND_COMPONENTS_INTERFACE timemory-cuda-event-example)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

add_executable(${EXE_NAME} ${EXE_NAME}.cu)
target_link_libraries(${EXE_NAME} timemory-cuda-event-example logging)
if("${CUDA_VERSION}" VERSION_LESS 11.0)
    target_compile_options(${EXE_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:$<$<CUDA_COMPILER_ID:NVIDIA>:--expt-extended-lambda>>)
else()
    target_compile_options(${EXE_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:$<$<CUDA_COMPILER_ID:NVIDIA>:--extended-lambda>>)
endif()

install(TARGETS ${EXE_NAME} DESTINATION bin OPTIONAL)
