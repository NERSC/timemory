cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(TiMemory-CXX-Overhead-Example LANGUAGES C CXX)

set(EXE_NAME test_cxx_overhead)
set(COMPONENTS cxx compile-options papi threading analysis-tools)

option(USE_EXTERN_TEMPLATES "Enable extern templates" OFF)
if(USE_EXTERN_TEMPLATES)
    list(APPEND COMPONENTS extern-templates)
endif()

set(TiMemory_FIND_COMPONENTS_INTERFACE timemory-cxx-overhead-example)
find_package(TiMemory REQUIRED COMPONENTS ${COMPONENTS})

add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
target_link_libraries(${EXE_NAME} timemory-cxx-overhead-example)
install(TARGETS ${EXE_NAME} DESTINATION bin)
