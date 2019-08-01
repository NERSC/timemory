cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(TiMemory-CXX-Basic-Example LANGUAGES CXX)

set(TiMemory_FIND_COMPONENTS_INTERFACE timemory-cxx-basic-example)
find_package(TiMemory REQUIRED COMPONENTS compile-options analysis-tools)

add_executable(test_cxx_basic test_cxx_basic.cpp)
target_link_libraries(test_cxx_basic timemory-cxx-basic-example)
install(TARGETS test_cxx_basic DESTINATION bin)
