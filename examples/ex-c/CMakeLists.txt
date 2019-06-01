cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0048 NEW)

#------------------------------------------------------------------------------#
project(TiMemory-C-Example LANGUAGES C)
message(STATUS "Project: ${PROJECT_NAME}")

set(EXE_NAME test_c_timing)

find_package(TiMemory REQUIRED)

add_executable(${EXE_NAME} ${EXE_NAME}.c)

target_link_libraries(${EXE_NAME} timemory-c-library-static timemory-c-compile-options)

if(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
    install(TARGETS ${EXE_NAME} DESTINATION bin)
endif()
