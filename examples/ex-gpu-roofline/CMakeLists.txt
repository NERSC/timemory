cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory")
    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT "CUDA" IN_LIST LANGUAGES OR NOT CMAKE_CUDA_COMPILER OR
        NOT (TIMEMORY_USE_CUDA AND TIMEMORY_USE_CUPTI))
        return()
    endif()
endif()

project(timemory-GPU-Roofline-Example LANGUAGES C CXX CUDA)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

option(BUILD_COMBINED_ROOFLINE "Build a combined roofline example" ON)

add_library(gpu-roofline-compile-options INTERFACE)
target_compile_options(gpu-roofline-compile-options INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)

set(EXE_NAME ex_gpu_roofline)
set(COMPONENTS compile-options analysis-tools roofline)
set(timemory_FIND_COMPONENTS_INTERFACE timemory-gpu-roofline-example)
set_source_files_properties(${EXE_NAME}.cpp PROPERTIES
    LANGUAGE CUDA
    LINKER_LANGUAGE CUDA)

find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

add_library(gpu-fp-half   INTERFACE)
add_library(gpu-fp-single INTERFACE)
add_library(gpu-fp-double INTERFACE)

target_compile_definitions(gpu-fp-half   INTERFACE ROOFLINE_FP_BYTES=2)
target_compile_definitions(gpu-fp-single INTERFACE ROOFLINE_FP_BYTES=4)
target_compile_definitions(gpu-fp-double INTERFACE ROOFLINE_FP_BYTES=8)

target_link_libraries(gpu-fp-half   INTERFACE gpu-roofline-compile-options)
target_link_libraries(gpu-fp-single INTERFACE gpu-roofline-compile-options)
target_link_libraries(gpu-fp-double INTERFACE gpu-roofline-compile-options)

# using half-precision
if(TIMEMORY_USE_CUDA_HALF)
    add_executable(ex-gpu-roofline-half            ${EXE_NAME}.cpp)
    target_link_libraries(ex-gpu-roofline-half     timemory-gpu-roofline-example gpu-fp-half)
    set_target_properties(ex-gpu-roofline-half     PROPERTIES OUTPUT_NAME ${EXE_NAME}.hp)
endif()

# using single-precision
add_executable(ex-gpu-roofline-single          ${EXE_NAME}.cpp)
target_link_libraries(ex-gpu-roofline-single   timemory-gpu-roofline-example gpu-fp-single)
set_target_properties(ex-gpu-roofline-single   PROPERTIES OUTPUT_NAME ${EXE_NAME}.sp)

# using double-precision
add_executable(ex-gpu-roofline-double          ${EXE_NAME}.cpp)
target_link_libraries(ex-gpu-roofline-double   timemory-gpu-roofline-example gpu-fp-double)
set_target_properties(ex-gpu-roofline-double   PROPERTIES OUTPUT_NAME ${EXE_NAME}.dp)

# using all-precision
if(BUILD_COMBINED_ROOFLINE)
    add_executable(ex-gpu-roofline                 ${EXE_NAME}.cpp)
    target_link_libraries(ex-gpu-roofline          timemory-gpu-roofline-example)
    set_target_properties(ex-gpu-roofline          PROPERTIES OUTPUT_NAME ${EXE_NAME})
endif()

install(
    TARGETS
        ex-gpu-roofline-single
        ex-gpu-roofline-double
    DESTINATION
        bin
    COMPONENT examples
    OPTIONAL)

if(TARGET ex-gpu-roofline)
    install(TARGETS ex-gpu-roofline DESTINATION bin COMPONENT examples OPTIONAL)
endif()

if(TARGET ex-gpu-roofline-half)
    install(TARGETS ex-gpu-roofline-half DESTINATION bin COMPONENT examples OPTIONAL)
endif()
