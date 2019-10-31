cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(timemory-Optional-Example LANGUAGES C CXX)

find_package(timemory REQUIRED COMPONENTS compile-options analysis-tools papi caliper)

file(GLOB sources ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.hpp)

add_executable(ex_optional_on ${sources})
add_executable(ex_optional_off ${sources})

target_link_libraries(ex_optional_on timemory)
target_compile_definitions(ex_optional_on PRIVATE USE_TIMEMORY)

install(TARGETS ex_optional_on ex_optional_off DESTINATION bin)
