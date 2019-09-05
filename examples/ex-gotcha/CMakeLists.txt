cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(timemory-gotcha-Example LANGUAGES CXX CUDA)

set(timemory_FIND_COMPONENTS_INTERFACE timemory::gotcha-example)
set(COMPONENTS compile-options arch vector mpi gotcha)

find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})
add_executable(test_gotcha test_gotcha.cpp)
add_executable(test_gotcha_native test_gotcha_native.cpp)
target_link_libraries(test_gotcha timemory::gotcha-example)
target_link_libraries(test_gotcha_native timemory::gotcha-example)
install(TARGETS test_gotcha DESTINATION bin)
