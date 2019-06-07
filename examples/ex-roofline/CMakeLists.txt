cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(TiMemory-CXX-Roofline-Example LANGUAGES C CXX)

set(EXE_NAME test_cxx_roofline)
set(COMPONENTS cxx papi threading compile-options analysis-tools)

option(USE_EXTERN_TEMPLATES "Enable extern templates" OFF)
if(USE_EXTERN_TEMPLATES)
    list(APPEND COMPONENTS extern-templates)
endif()

find_package(TiMemory REQUIRED COMPONENTS ${COMPONENTS})

add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
target_link_libraries(${EXE_NAME} timemory)
install(TARGETS ${EXE_NAME} DESTINATION bin)
