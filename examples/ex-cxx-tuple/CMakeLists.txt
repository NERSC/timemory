cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(TiMemory-CXX-Tuple-Example LANGUAGES C CXX)

set(EXE_NAME test_cxx_tuple)
set(COMPONENTS cxx compile-options extern-templates)

find_package(TiMemory REQUIRED COMPONENTS ${COMPONENTS})
add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
target_link_libraries(${EXE_NAME} timemory)

if(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
    install(TARGETS ${EXE_NAME} DESTINATION bin)
endif()
