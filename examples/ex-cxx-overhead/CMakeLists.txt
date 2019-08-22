cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(TiMemory-CXX-Overhead-Example LANGUAGES C CXX)

set(EXE_NAME test_cxx_overhead)
set(COMPONENTS compile-options analysis-tools)

option(USE_PAPI "Enable papi" OFF)
option(USE_CALIPER "Enable caliper" OFF)
option(USE_EXTERN_TEMPLATES "Enable extern templates" OFF)

set(USE_PAPI_COMPONENTS papi)
set(USE_CALIPER_COMPONENTS caliper)
set(USE_EXTERN_TEMPLATES_COMPONENTS cxx extern-templates)

foreach(_OPT PAPI CALIPER EXTERN_TEMPLATES)
    message(STATUS "${PROJECT_NAME}: USE_${_OPT} = ${USE_${_OPT}}")
    if(USE_${_OPT})
        list(APPEND COMPONENTS ${USE_${_OPT}_COMPONENTS})
    endif()
endforeach()

set(TiMemory_FIND_COMPONENTS_INTERFACE timemory-cxx-overhead-example)
find_package(TiMemory REQUIRED COMPONENTS ${COMPONENTS})

add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
target_link_libraries(${EXE_NAME} timemory-cxx-overhead-example)
install(TARGETS ${EXE_NAME} DESTINATION bin)
