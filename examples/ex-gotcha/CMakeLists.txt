cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# this is for internal use
if(WIN32 OR ("${CMAKE_PROJECT_NAME}" STREQUAL "timemory" AND NOT TIMEMORY_USE_GOTCHA))
    return()
endif()

project(timemory-gotcha-Example LANGUAGES C CXX)

include(GNUInstallDirs)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-gotcha-example)
set(COMPONENTS compile-options arch vector gotcha)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

add_library(ex_gotcha_lib SHARED ex_gotcha_lib.hpp ex_gotcha_lib.cpp)
target_link_libraries(ex_gotcha_lib PUBLIC timemory-gotcha-example)

add_executable(ex_gotcha ex_gotcha.cpp)
target_link_libraries(ex_gotcha ex_gotcha_lib)

add_executable(ex_gotcha_mpi ex_gotcha.cpp)
target_link_libraries(ex_gotcha_mpi ex_gotcha_lib timemory-mpi)

install(TARGETS ex_gotcha ex_gotcha_mpi DESTINATION bin)
install(TARGETS ex_gotcha_lib             DESTINATION ${CMAKE_INSTALL_LIBDIR})
