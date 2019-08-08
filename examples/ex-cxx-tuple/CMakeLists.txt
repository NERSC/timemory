cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(TiMemory-CXX-Tuple-Example LANGUAGES C CXX)

set(EXE_NAME test_cxx_tuple)
set(COMPONENTS cxx compile-options papi threading analysis-tools extern-templates)

option(USE_CALIPER "Enable caliper" ON)
if(USE_CALIPER)
    list(APPEND COMPONENTS caliper)
endif()

set(TiMemory_FIND_COMPONENTS_INTERFACE timemory-cxx-tuple-example)
find_package(TiMemory REQUIRED COMPONENTS ${COMPONENTS})

add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
target_link_libraries(${EXE_NAME} timemory-cxx-tuple-example)
install(TARGETS ${EXE_NAME} DESTINATION bin)
