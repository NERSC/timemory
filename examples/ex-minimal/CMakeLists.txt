cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(timemory-Minimal-Example LANGUAGES CXX)

include(CheckCCompilerFlag)
set(timemory_FIND_COMPONENTS_INTERFACE timemory-minimal-example)
set(COMPONENTS compile-options analysis-tools papi caliper)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

set(TIMEMORY_C_LIBRARY )
if(TARGET timemory-c-shared)
    set(TIMEMORY_C_LIBRARY timemory-c-shared)
elseif(TARGET timemory-c-static)
    set(TIMEMORY_C_LIBRARY timemory-c-static)
endif()

if(TIMEMORY_C_LIBRARY)
    check_c_compiler_flag("-Wno-unused-parameter" "no_unused_parameter")
    add_executable(test_c_minimal test_minimal.c)
    target_link_libraries(test_c_minimal timemory-minimal-example ${TIMEMORY_C_LIBRARY})
    install(TARGETS test_c_minimal DESTINATION bin)
    if(no_unused_parameter)
        target_compile_options(test_c_minimal PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wno-unused-parameter>)
    endif()
endif()

add_executable(test_cxx_minimal test_minimal.cpp)
target_link_libraries(test_cxx_minimal timemory-minimal-example)
install(TARGETS test_cxx_minimal DESTINATION bin)

if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "/usr/bin/env python")
endif()

configure_file(${PROJECT_SOURCE_DIR}/test_python_minimal.py
    ${CMAKE_BINARY_DIR}/test_python_minimal @ONLY)

install(
    FILES ${CMAKE_BINARY_DIR}/test_python_minimal
    DESTINATION bin
    PERMISSIONS
        OWNER_EXECUTE OWNER_READ OWNER_WRITE
        GROUP_EXECUTE GROUP_READ
        WORLD_EXECUTE WORLD_READ)
