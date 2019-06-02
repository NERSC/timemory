cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(TiMemory-C-Example LANGUAGES C)

set(EXE_NAME test_c_timing)
set(COMPONENTS c static compile-options extensions)

option(USE_EXTERN_TEMPLATES "Enable extern templates" OFF)
if(USE_EXTERN_TEMPLATES)
    list(APPEND COMPONENTS extern-templates)
endif()

find_package(TiMemory REQUIRED COMPONENTS ${COMPONENTS})
add_executable(${EXE_NAME} ${EXE_NAME}.c)
target_link_libraries(${EXE_NAME} timemory)

if(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
    install(TARGETS ${EXE_NAME} DESTINATION bin)
endif()
