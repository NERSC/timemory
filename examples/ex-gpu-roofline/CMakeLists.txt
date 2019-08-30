cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "TiMemory")
    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT "CUDA" IN_LIST LANGUAGES OR NOT CMAKE_CUDA_COMPILER OR
        NOT (TIMEMORY_USE_CUDA AND TIMEMORY_USE_CUPTI))
        return()
    endif()
endif()

project(TiMemory-GPU-Roofline-Example LANGUAGES C CXX CUDA)

add_library(gpu-roofline-compile-options INTERFACE)
target_compile_options(gpu-roofline-compile-options INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)

set(EXE_NAME test_gpu_roofline)
set(COMPONENTS cuda cupti compile-options analysis-tools arch papi)
set(TiMemory_FIND_COMPONENTS_INTERFACE timemory-gpu-roofline-example)
set_source_files_properties(${EXE_NAME}.cpp PROPERTIES
    LANGUAGE CUDA
    LINKER_LANGUAGE CUDA)

find_package(TiMemory REQUIRED COMPONENTS ${COMPONENTS})

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
add_executable(test-gpu-roofline-half            ${EXE_NAME}.cpp)
target_link_libraries(test-gpu-roofline-half     timemory-gpu-roofline-example gpu-fp-half)
set_target_properties(test-gpu-roofline-half     PROPERTIES OUTPUT_NAME ${EXE_NAME}.hp)

# using single-precision
add_executable(test-gpu-roofline-single          ${EXE_NAME}.cpp)
target_link_libraries(test-gpu-roofline-single   timemory-gpu-roofline-example gpu-fp-single)
set_target_properties(test-gpu-roofline-single   PROPERTIES OUTPUT_NAME ${EXE_NAME}.sp)

# using double-precision
add_executable(test-gpu-roofline-double          ${EXE_NAME}.cpp)
target_link_libraries(test-gpu-roofline-double   timemory-gpu-roofline-example gpu-fp-double)
set_target_properties(test-gpu-roofline-double   PROPERTIES OUTPUT_NAME ${EXE_NAME}.dp)

# using double-precision
add_executable(test-gpu-roofline                 ${EXE_NAME}.cpp)
target_link_libraries(test-gpu-roofline          timemory-gpu-roofline-example)
set_target_properties(test-gpu-roofline          PROPERTIES OUTPUT_NAME ${EXE_NAME})

install(
    TARGETS
        test-gpu-roofline
        test-gpu-roofline-half
        test-gpu-roofline-single
        test-gpu-roofline-double
    DESTINATION
        bin)
