cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(timemory-CXX-Tuple-Example LANGUAGES C CXX)

set(EXE_NAME ex_cxx_tuple)
set(COMPONENTS compile-options papi threading analysis-tools)

option(USE_MPI "Enable MPI" ON)
if(USE_MPI)
    list(APPEND COMPONENTS mpi)
endif()

option(USE_CALIPER "Enable caliper" ON)
if(USE_CALIPER)
    list(APPEND COMPONENTS caliper)
endif()

set(timemory_FIND_COMPONENTS_INTERFACE timemory-cxx-tuple-example)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
target_link_libraries(${EXE_NAME} timemory-cxx-tuple-example)
install(TARGETS ${EXE_NAME} DESTINATION bin)
