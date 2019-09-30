cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(timemory-Minimal-Example LANGUAGES CXX)

include(CheckCCompilerFlag)
set(timemory_FIND_COMPONENTS_INTERFACE timemory-minimal-example)
set(COMPONENTS compile-options analysis-tools)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

set(TIMEMORY_C_LIBRARY )
if("C" IN_LIST timemory_ENABLED_LINK_LANGUAGES)
    if("SHARED" IN_LIST timemory_ENABLED_LINK_LIBRARIES)
        set(TIMEMORY_C_LIBRARY timemory-c-shared)
    elseif("STATIC" IN_LIST timemory_ENABLED_LINK_LIBRARIES)
        set(TIMEMORY_C_LIBRARY timemory-c-static)
    endif()
endif()

set(TIMEMORY_CXX_LIBRARY )
if("CXX" IN_LIST timemory_ENABLED_LINK_LANGUAGES)
    if("SHARED" IN_LIST timemory_ENABLED_LINK_LIBRARIES)
        set(TIMEMORY_CXX_LIBRARY timemory-cxx-shared)
    elseif("STATIC" IN_LIST timemory_ENABLED_LINK_LIBRARIES)
        set(TIMEMORY_CXX_LIBRARY timemory-cxx-static)
    endif()
endif()

# if C library is available
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

endif()

# if C stubs via CXX library are available
if(TIMEMORY_CXX_LIBRARY)
    add_executable(test_c_minimal_library test_minimal_library.c)
    target_link_libraries(test_c_minimal_library timemory-minimal-example
        ${TIMEMORY_CXX_LIBRARY} ${TIMEMORY_C_LIBRARY})
    install(TARGETS test_c_minimal_library DESTINATION bin)
endif()

# headers are always available
add_executable(test_cxx_minimal test_minimal.cpp)
target_link_libraries(test_cxx_minimal timemory-minimal-example)
install(TARGETS test_cxx_minimal DESTINATION bin)

if(TIMEMORY_CXX_LIBRARY)
    add_executable(test_cxx_minimal_library test_minimal_library.cpp)
    target_link_libraries(test_cxx_minimal_library timemory-minimal-example ${TIMEMORY_CXX_LIBRARY})
    install(TARGETS test_cxx_minimal_library DESTINATION bin)
endif()

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
