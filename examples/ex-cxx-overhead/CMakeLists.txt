cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(TiMemory-CXX-Overhead-Example LANGUAGES C CXX)

set(EXE_NAME test_cxx_overhead)
set(COMPONENTS cxx compile-options)

option(USE_EXTERN_TEMPLATES "Enable extern templates" OFF)
if(USE_EXTERN_TEMPLATES)
    list(APPEND COMPONENTS extern-templates)
endif()

find_package(TiMemory REQUIRED COMPONENTS ${COMPONENTS})
add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
target_link_libraries(${EXE_NAME} timemory)

if(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
    install(TARGETS ${EXE_NAME} DESTINATION bin)
endif()
