cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(timemory-Minimal-Example LANGUAGES CXX)

include(CheckCCompilerFlag)
set(timemory_FIND_COMPONENTS_INTERFACE timemory-minimal-example)
set(COMPONENTS compile-options analysis-tools)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

set(TIMEMORY_C_LIBRARY )
if(TARGET timemory-c-shared)
    set(TIMEMORY_C_LIBRARY timemory-c-shared)
elseif(TARGET timemory-c-static)
    set(TIMEMORY_C_LIBRARY timemory-c-static)
endif()

if(TIMEMORY_C_LIBRARY)
    check_c_compiler_flag("-Wno-unused-parameter" "no_unused_parameter")
    if(no_unused_parameter)
        # use this method so flag is provided after timemory-compile-options
        add_library(test-c-minimal-compile-options INTERFACE)
        target_compile_options(test-c-minimal-compile-options INTERFACE
            $<$<COMPILE_LANGUAGE:C>:-Wno-unused-parameter>)
        list(APPEND TIMEMORY_C_LIBRARY test-c-minimal-compile-options)
    endif()

    add_executable(test_c_minimal test_minimal.c)
    target_link_libraries(test_c_minimal timemory-minimal-example ${TIMEMORY_C_LIBRARY})
    install(TARGETS test_c_minimal DESTINATION bin)

    add_executable(test_c_minimal_library test_minimal_library.c)
    target_link_libraries(test_c_minimal_library timemory-minimal-example
        timemory-cxx-shared ${TIMEMORY_C_LIBRARY})
    install(TARGETS test_c_minimal_library DESTINATION bin)
endif()

add_executable(test_cxx_minimal test_minimal.cpp)
target_link_libraries(test_cxx_minimal timemory-minimal-example)
install(TARGETS test_cxx_minimal DESTINATION bin)

add_executable(test_cxx_minimal_library test_minimal_library.cpp)
target_link_libraries(test_cxx_minimal_library timemory-minimal-example timemory-cxx-shared)
install(TARGETS test_cxx_minimal_library DESTINATION bin)

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
