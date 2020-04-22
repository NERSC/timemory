cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# this is for internal use
if(WIN32 OR ("${CMAKE_PROJECT_NAME}" STREQUAL "timemory" AND NOT TIMEMORY_USE_GOTCHA))
    return()
endif()

project(timemory-gotcha-Example LANGUAGES C CXX)

include(GNUInstallDirs)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-gotcha-example)
set(COMPONENTS compile-options vector gotcha analysis-tools OPTIONAL_COMPONENTS arch)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

add_library(ex_gotcha_lib SHARED ex_gotcha_lib.hpp ex_gotcha_lib.cpp)
target_link_libraries(ex_gotcha_lib PUBLIC timemory-gotcha-example)

add_executable(ex_gotcha ex_gotcha.cpp)
target_link_libraries(ex_gotcha ex_gotcha_lib)

add_executable(ex_gotcha_replacement ex_gotcha_replacement.cpp)
target_link_libraries(ex_gotcha_replacement ex_gotcha_lib)

add_library(ex_gotcha_lib_mpi SHARED ex_gotcha_lib.hpp ex_gotcha_lib.cpp)
target_link_libraries(ex_gotcha_lib_mpi PUBLIC timemory-gotcha-example timemory-mpi)

add_executable(ex_gotcha_mpi ex_gotcha.cpp)
target_link_libraries(ex_gotcha_mpi ex_gotcha_lib_mpi timemory-mpi)

install(TARGETS ex_gotcha ex_gotcha_mpi ex_gotcha_replacement DESTINATION bin)
install(TARGETS ex_gotcha_lib             DESTINATION ${CMAKE_INSTALL_LIBDIR})
