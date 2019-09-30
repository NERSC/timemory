cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(timemory-CPU-Roofline-Example LANGUAGES C CXX)

set(EXE_NAME test_cpu_roofline)
set(COMPONENTS papi threading compile-options analysis-tools arch)

option(USE_EXTERN_TEMPLATES "Enable extern templates" OFF)
if(USE_EXTERN_TEMPLATES)
    list(APPEND COMPONENTS cxx shared extern-templates)
endif()

find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

add_library(cpu-fp-single INTERFACE)
add_library(cpu-fp-double INTERFACE)

target_compile_definitions(cpu-fp-single INTERFACE ROOFLINE_FP_BYTES=4)
target_compile_definitions(cpu-fp-double INTERFACE ROOFLINE_FP_BYTES=8)

# create the fibonacci example
add_executable(test-cpu-roofline-fib-single          ${EXE_NAME}.cpp)
target_link_libraries(test-cpu-roofline-fib-single   timemory cpu-fp-single)
set_target_properties(test-cpu-roofline-fib-single   PROPERTIES OUTPUT_NAME ${EXE_NAME}.sp)

add_executable(test-cpu-roofline-fib-double          ${EXE_NAME}.cpp)
target_link_libraries(test-cpu-roofline-fib-double   timemory cpu-fp-double)
set_target_properties(test-cpu-roofline-fib-double   PROPERTIES OUTPUT_NAME ${EXE_NAME})

# install the targets
install(TARGETS test-cpu-roofline-fib-single test-cpu-roofline-fib-double DESTINATION bin)

if(TIMEMORY_USE_PAPI)
    set(sources test_cpu_roofline_ert.cpp kernel.cpp kernel.hpp rep.h)

    add_library(ert-definitions INTERFACE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        target_compile_definitions(ert-definitions INTERFACE ERT_INTEL)
    endif()
    set(ERT_ALIGN 0 CACHE STRING "Alignment")
    set(ERT_FLOP 0 CACHE STRING "Flops")
    set(ERT_TRIALS_MIN 0 CACHE STRING "Trials")
    set(ERT_WORKING_SET_MIN 0 CACHE STRING "Working set")
    set(ERT_MEMORY_MAX 0 CACHE STRING "Memory size")

    foreach(_FIELD ERT_ALIGN ERT_FLOP ERT_TRIALS_MIN ERT_WORKING_SET_MIN ERT_MEMORY_MAX)
        target_compile_definitions(ert-definitions INTERFACE ${_FIELD}=${${_FIELD}})
    endforeach()

    # create an executable that uses the legacy ERT macro system for expansion
    add_executable(test-cpu-roofline-macros ${sources})
    target_link_libraries(test-cpu-roofline-macros timemory)
    target_compile_definitions(test-cpu-roofline-macros PRIVATE MACROS)
    set_target_properties(test-cpu-roofline-macros
        PROPERTIES OUTPUT_NAME ${EXE_NAME}_ert_with_macros)

    # create an executable that uses a template unrolling method for expansion
    add_executable(test-cpu-roofline-templates ${sources})
    target_link_libraries(test-cpu-roofline-templates timemory)
    target_compile_definitions(test-cpu-roofline-templates PRIVATE TEMPLATES)
    set_target_properties(test-cpu-roofline-templates
        PROPERTIES OUTPUT_NAME ${EXE_NAME}_ert_with_templates)

    # install the targets
    install(TARGETS test-cpu-roofline-templates test-cpu-roofline-macros DESTINATION bin)
endif()
