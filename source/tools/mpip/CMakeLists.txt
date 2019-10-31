cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# this is for internal use
if(NOT TIMEMORY_BUILD_MPIP OR (NOT TIMEMORY_USE_MPI AND NOT TIMEMORY_USE_GOTCHA))
    return()
endif()

project(timemory-mpip-tool LANGUAGES CXX)

# find MPI
find_package(MPI QUIET)
if(NOT MPI_FOUND)
    return()
endif()

set(_MPI_DIRS
    ${MPI_INCLUDE_DIRS}
    ${MPI_C_HEADER_DIR}
    ${MPI_CXX_HEADER_DIR}
    ${MPI_ROOT_DIR})

find_file(MPI_HEADER
    NAMES mpi.h
    HINTS ${_MPI_DIRS}
    PATHS ${_MPI_DIRS})

if(NOT MPI_HEADER)
    message(WARNING "${PROJECT_NAME} could not find mpi.h in ${_MPI_DIRS}")
    return()
endif()

configure_file(${PROJECT_SOURCE_DIR}/create_mpi_gotcha.sh
    ${PROJECT_BINARY_DIR}/create_mpi_gotcha.sh @ONLY)

execute_process(
    COMMAND ${PROJECT_BINARY_DIR}/create_mpi_gotcha.sh
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    OUTPUT_FILE ${PROJECT_BINARY_DIR}/_timemory-mpip.cpp
    ERROR_VARIABLE ERR_MSG
    RESULT_VARIABLE RET
)

# ensure we have a fresh file
#execute_process(COMMAND
#    ${CMAKE_COMMAND} -E remove -f ${PROJECT_SOURCE_DIR}/timemory-mpip.cpp)

# will only copy if different
configure_file(${PROJECT_BINARY_DIR}/_timemory-mpip.cpp
    ${PROJECT_BINARY_DIR}/timemory-mpip.cpp COPYONLY)

if(NOT EXISTS ${PROJECT_BINARY_DIR}/timemory-mpip.cpp)
    message(STATUS "Missing file '${PROJECT_BINARY_DIR}/timemory-mpip.cpp'...")
    return()
endif()

if(NOT RET GREATER 0)
    add_library(timemory-mpip SHARED ${PROJECT_BINARY_DIR}/timemory-mpip.cpp)
    target_link_libraries(timemory-mpip
        PRIVATE
            timemory-headers
            timemory-cxx-shared
            timemory-compile-options
            timemory-mpi
            timemory-arch
            timemory-vector
            timemory-gotcha)
    set_target_properties(timemory-mpip PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH ON)
    install(TARGETS timemory-mpip DESTINATION ${CMAKE_INSTALL_LIBDIR})
else()
    message(WARNING "Executing '${PROJECT_BINARY_DIR}/create_mpi_gotcha.sh' returned ${RET}")
endif()
