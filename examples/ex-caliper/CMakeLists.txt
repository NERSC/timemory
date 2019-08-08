cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(TiMemory-Caliper-Example LANGUAGES CXX)

set(TiMemory_FIND_COMPONENTS_INTERFACE timemory-caliper-example)
set(COMPONENTS compile-options analysis-tools caliper)
find_package(TiMemory REQUIRED COMPONENTS ${COMPONENTS})

add_executable(test_caliper test_caliper.cpp)
target_link_libraries(test_caliper timemory-caliper-example)
install(TARGETS test_caliper DESTINATION bin)
