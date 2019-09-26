cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

if(WIN32)
	return()
endif()

project(timemory-gotcha-Example LANGUAGES C CXX)

include(GNUInstallDirs)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-gotcha-example)
set(COMPONENTS compile-options arch vector gotcha)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

add_library(test_gotcha_lib SHARED test_gotcha_lib.hpp test_gotcha_lib.cpp)
target_link_libraries(test_gotcha_lib PUBLIC timemory-gotcha-example)

add_executable(test_gotcha test_gotcha.cpp)
target_link_libraries(test_gotcha test_gotcha_lib)

add_executable(test_gotcha_mpi test_gotcha.cpp)
target_link_libraries(test_gotcha_mpi test_gotcha_lib timemory-mpi)

install(TARGETS test_gotcha test_gotcha_mpi DESTINATION bin)
install(TARGETS test_gotcha_lib             DESTINATION ${CMAKE_INSTALL_LIBDIR})

# if(NOT "timemory-gotcha" IN_LIST timemory_EMPTY_INTERFACE_LIBRARIES)
#    add_executable(test_gotcha_native test_gotcha_native.cpp)
#    target_link_libraries(test_gotcha_native timemory-gotcha-example)
#    add_executable(test_gotcha_native_mpi test_gotcha_native.cpp)
#    target_link_libraries(test_gotcha_native_mpi timemory-gotcha-example timemory-mpi)
#    install(TARGETS test_gotcha_native test_gotcha_native_mpi DESTINATION bin)
# endif()
