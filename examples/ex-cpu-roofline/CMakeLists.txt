cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory" AND NOT TIMEMORY_USE_PAPI)
    return()
endif()

project(timemory-CPU-Roofline-Example LANGUAGES C CXX)

set(EXE_NAME ex_cpu_roofline)
set(COMPONENTS compile-options analysis-tools cpu-roofline ert)

find_package(timemory REQUIRED COMPONENTS ${COMPONENTS} OPTIONAL_COMPONENTS mpi)

add_library(cpu-fp-single INTERFACE)
add_library(cpu-fp-double INTERFACE)

target_compile_definitions(cpu-fp-single INTERFACE ROOFLINE_FP_BYTES=4)
target_compile_definitions(cpu-fp-double INTERFACE ROOFLINE_FP_BYTES=8)

# create the fibonacci example
add_executable(ex-cpu-roofline-fib-single          ${EXE_NAME}.cpp)
target_link_libraries(ex-cpu-roofline-fib-single   timemory cpu-fp-single)
set_target_properties(ex-cpu-roofline-fib-single   PROPERTIES OUTPUT_NAME ${EXE_NAME}.sp)

add_executable(ex-cpu-roofline-fib-double          ${EXE_NAME}.cpp)
target_link_libraries(ex-cpu-roofline-fib-double   timemory cpu-fp-double)
set_target_properties(ex-cpu-roofline-fib-double   PROPERTIES OUTPUT_NAME ${EXE_NAME})

# install the targets
install(TARGETS ex-cpu-roofline-fib-single ex-cpu-roofline-fib-double DESTINATION bin
    COMPONENT examples OPTIONAL)
