cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(TiMemory-CXX-Tuple-Example LANGUAGES C CXX)

set(EXE_NAME test_cxx_tuple)
set(COMPONENTS cxx compile-options extensions extern-templates)

set(TiMemory_FIND_COMPONENTS_INTERFACE timemory-cxx-tuple-example)
find_package(TiMemory REQUIRED COMPONENTS ${COMPONENTS})
add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
target_link_libraries(${EXE_NAME} timemory-cxx-tuple-example)

if(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
    install(TARGETS ${EXE_NAME} DESTINATION bin)
endif()
