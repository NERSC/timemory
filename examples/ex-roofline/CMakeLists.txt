cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(TiMemory-CXX-Roofline-Example LANGUAGES C CXX)

set(EXE_NAME test_cxx_roofline)
set(COMPONENTS cxx papi threading compile-options analysis-tools arch)

option(USE_EXTERN_TEMPLATES "Enable extern templates" OFF)
if(USE_EXTERN_TEMPLATES)
    list(APPEND COMPONENTS extern-templates)
endif()

find_package(TiMemory REQUIRED COMPONENTS ${COMPONENTS})

set(sources test_cxx_roofline.cpp kernel.cpp kernel.hpp rep.h)

add_executable(${EXE_NAME}-macros ${sources})
add_executable(${EXE_NAME}-templates ${sources})
add_executable(${EXE_NAME}-fib ${EXE_NAME}_fib.cpp)

target_link_libraries(${EXE_NAME}-macros timemory)
install(TARGETS ${EXE_NAME}-macros DESTINATION bin)
target_compile_definitions(${EXE_NAME}-macros PRIVATE MACROS)
set_target_properties(${EXE_NAME}-macros PROPERTIES
  OUTPUT_NAME ${EXE_NAME}-ert-with-macros)

target_link_libraries(${EXE_NAME}-templates timemory)
install(TARGETS ${EXE_NAME}-templates DESTINATION bin)
target_compile_definitions(${EXE_NAME}-templates PRIVATE TEMPLATES)
set_target_properties(${EXE_NAME}-templates PROPERTIES
  OUTPUT_NAME ${EXE_NAME}-ert-with-templates)

target_link_libraries(${EXE_NAME}-fib timemory)
install(TARGETS ${EXE_NAME}-fib DESTINATION bin)
set_target_properties(${EXE_NAME}-fib PROPERTIES
  OUTPUT_NAME ${EXE_NAME})
