cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(timemory-CXX-Overhead-Example LANGUAGES C CXX)

set(EXE_NAME ex_cxx_overhead)
set(COMPONENTS compile-options analysis-tools OPTIONAL_COMPONENTS cxx)

option(USE_PAPI "Enable papi" OFF)
option(USE_CALIPER "Enable caliper" OFF)
option(USE_MPI "Enable MPI" OFF)
option(USE_TAU "Enable TAU" OFF)

set(USE_PAPI_COMPONENTS papi)
set(USE_CALIPER_COMPONENTS caliper)
set(USE_MPI_COMPONENTS mpi)
set(USE_TAU_COMPONENTS tau)

foreach(_OPT PAPI CALIPER MPI TAU)
    message(STATUS "${PROJECT_NAME}: USE_${_OPT} = ${USE_${_OPT}}")
    if(USE_${_OPT})
        list(APPEND COMPONENTS ${USE_${_OPT}_COMPONENTS})
    endif()
endforeach()

set(timemory_FIND_COMPONENTS_INTERFACE timemory-cxx-overhead-example)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

# this tend to cause the CI to exceed the time-limit
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
target_link_libraries(${EXE_NAME} timemory-cxx-overhead-example)
install(TARGETS ${EXE_NAME} DESTINATION bin OPTIONAL)
