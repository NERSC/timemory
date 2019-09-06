cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(timemory-gotcha-Example LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(timemory_FIND_COMPONENTS_INTERFACE timemory::gotcha-example)
set(COMPONENTS compile-options arch vector mpi gotcha)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

if(NOT "timemory-mpi" IN_LIST timemory_EMPTY_INTERFACE_LIBRARIES)
    add_library(test_gotcha_lib SHARED test_gotcha_lib.hpp test_gotcha_lib.cpp)
    target_link_libraries(test_gotcha_lib PUBLIC timemory::gotcha-example)

    add_executable(test_gotcha test_gotcha.cpp)
    target_link_libraries(test_gotcha test_gotcha_lib)

    install(TARGETS test_gotcha DESTINATION bin)
    install(TARGETS test_gotcha_lib DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(NOT "timemory-gotcha" IN_LIST timemory_EMPTY_INTERFACE_LIBRARIES)
    add_executable(test_gotcha_native test_gotcha_native.cpp)
    target_link_libraries(test_gotcha_native timemory::gotcha-example)
    install(TARGETS test_gotcha_native DESTINATION bin)
endif()
