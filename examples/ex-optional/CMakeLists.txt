cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(TiMemory-Optional-Example LANGUAGES C CXX)

find_package(TiMemory REQUIRED COMPONENTS compile-options analysis-tools papi caliper)

file(GLOB sources ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.hpp)

add_executable(test_optional_on ${sources})
add_executable(test_optional_off ${sources})

target_link_libraries(test_optional_on timemory)
target_compile_definitions(test_optional_on PRIVATE USE_TIMEMORY)

install(TARGETS test_optional_on test_optional_off DESTINATION bin)
