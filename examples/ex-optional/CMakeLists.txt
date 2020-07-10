cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(timemory-Optional-Example LANGUAGES C CXX)

find_package(timemory REQUIRED COMPONENTS compile-options analysis-tools
    OPTIONAL_COMPONENTS cxx shared)

option(USE_OPENMP "Enable OpenMP in timemory example" ON)
option(USE_MPI "Enable MPI in timemory example" ON)

find_package(OpenMP QUIET)
find_package(MPI QUIET)

if(OpenMP_FOUND AND USE_OPENMP)
    list(APPEND _OMP OpenMP::OpenMP_CXX)
endif()

if(USE_MPI)
    if(TARGET timemory-mpi)
        set(_MPI timemory-mpi)
    elseif(TARGET timemory::timemory-mpi)
        set(_MPI timemory::timemory-mpi)
    endif()

    if(MPI_FOUND)
        list(APPEND _MPI MPI::MPI_CXX)
        target_compile_definitions(MPI::MPI_CXX INTERFACE TIMEMORY_USE_MPI)
    endif()
endif()

file(GLOB sources ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.hpp)

add_executable(ex_optional_on ${sources})
add_executable(ex_optional_off ${sources})

target_link_libraries(ex_optional_on timemory ${_OMP})
target_link_libraries(ex_optional_off timemory-compile-options ${_OMP})
target_compile_definitions(ex_optional_on PRIVATE USE_TIMEMORY)

install(TARGETS ex_optional_on ex_optional_off DESTINATION bin OPTIONAL)

if(_MPI)
    add_executable(ex_optional_on.mpi ${sources})
    add_executable(ex_optional_off.mpi ${sources})

    target_link_libraries(ex_optional_on.mpi timemory ${_MPI} ${_OMP})
    target_link_libraries(ex_optional_off.mpi timemory-compile-options ${_MPI} ${_OMP})
    target_compile_definitions(ex_optional_on.mpi PRIVATE USE_TIMEMORY)

    install(TARGETS ex_optional_on.mpi ex_optional_off.mpi DESTINATION bin OPTIONAL)
endif()
