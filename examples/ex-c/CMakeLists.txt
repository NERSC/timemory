cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory" AND NOT TIMEMORY_BUILD_C)
    return()
endif()

project(timemory-C-Example LANGUAGES C)

set(EXE_NAME test_c_timing)
set(COMPONENTS c shared compile-options analysis-tools)

option(USE_EXTERN_TEMPLATES "Enable extern templates" OFF)
if(USE_EXTERN_TEMPLATES)
    list(APPEND COMPONENTS extern-templates)
endif()

set(timemory_FIND_COMPONENTS_INTERFACE timemory::c-example)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

add_executable(${EXE_NAME} ${EXE_NAME}.c)
target_link_libraries(${EXE_NAME} timemory::c-example)
install(TARGETS ${EXE_NAME} DESTINATION bin)
