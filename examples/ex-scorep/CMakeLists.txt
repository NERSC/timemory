cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory" AND NOT TIMEMORY_USE_SCOREP)
    return()
endif()

project(timemory-ScoreP-Example LANGUAGES C CXX)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-scorep-example)
set(COMPONENTS headers compile-options analysis-tools scorep-component)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

option(USE_OPENMP "Enable OpenMP in Score-P example" ON)
option(USE_MPI "Enable MPI in Score-P example" ON)

find_package(OpenMP QUIET)
find_package(MPI QUIET)

if(OpenMP_FOUND AND USE_OPENMP)
    list(APPEND _OMP OpenMP::OpenMP_CXX)
endif()

if(USE_MPI AND TIMEMORY_USE_MPI AND TIMEMORY_USE_MPI_INIT)
    if(TARGET timemory-mpi)
        set(_MPI timemory-mpi)
    elseif(TARGET timemory::timemory-mpi)
        set(_MPI timemory::timemory-mpi)
    endif()

    if(MPI_FOUND)
        list(APPEND _MPI MPI::MPI_CXX)
    endif()
endif()

# Add the simple example
add_executable(ex_scorep ex_scorep.cpp)
target_link_libraries(ex_scorep timemory-scorep-example)
install(TARGETS ex_scorep DESTINATION bin
    COMPONENT examples OPTIONAL)

# Add the OpenMP enabled example
if (_OMP)
    add_executable(ex_scorep.omp ex_scorep.cpp)
    target_compile_definitions(ex_scorep.omp PRIVATE USE_OPENMP)
    target_link_libraries(ex_scorep.omp timemory-scorep-example ${_OMP})
    install(TARGETS ex_scorep.omp DESTINATION bin
        COMPONENT examples OPTIONAL)
endif()

# Add the MPI enabled example
if(_MPI)
    add_executable(ex_scorep.mpi ex_scorep.cpp)
    target_compile_definitions(ex_scorep.mpi PRIVATE USE_MPI)
    if (_OMP)
        target_compile_definitions(ex_scorep.mpi PRIVATE USE_OPENMP)
    endif()

    target_link_libraries(ex_scorep.mpi timemory-scorep-example ${_MPI})
    install(TARGETS ex_scorep.mpi DESTINATION DESTINATION bin
        COMPONENT examples OPTIONAL)
endif()
