cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0048 NEW)

#------------------------------------------------------------------------------#
project(TiMemory-CXX-Overhead-Example LANGUAGES C CXX)
message(STATUS "Project: ${PROJECT_NAME}")

set(EXE_NAME test_cxx_overhead)

find_package(TiMemory REQUIRED)

add_executable(${EXE_NAME} ${EXE_NAME}.cpp)

target_link_libraries(${EXE_NAME} timemory-cxx-library timemory-cxx-compile-options)

option(USE_EXTERN_TEMPLATES "Enable extern templates" OFF)
if(USE_EXTERN_TEMPLATES)
    message(STATUS "Using extern templates from TiMemory...")
    target_link_libraries(${EXE_NAME} timemory-extern-templates)
endif()

if(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
    install(TARGETS ${EXE_NAME} DESTINATION bin)
endif()
