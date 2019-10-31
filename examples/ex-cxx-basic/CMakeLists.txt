cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(timemory-CXX-Basic-Example LANGUAGES CXX)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-cxx-basic-example)
set(COMPONENTS compile-options analysis-tools papi)
option(USE_CALIPER "Enable timemory Caliper" ON)
if(USE_CALIPER)
    list(APPEND COMPONENTS caliper)
endif()

find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})
add_executable(ex_cxx_basic ex_cxx_basic.cpp)
target_link_libraries(ex_cxx_basic timemory-cxx-basic-example)
install(TARGETS ex_cxx_basic DESTINATION bin)
