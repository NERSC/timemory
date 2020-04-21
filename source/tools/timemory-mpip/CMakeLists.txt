cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

# this is for internal use
if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory")
    if(NOT TIMEMORY_BUILD_MPIP OR NOT TIMEMORY_USE_MPI OR NOT TIMEMORY_USE_GOTCHA)
        return()
    endif()
endif()

project(timemory-mpip-tool)

# find MPI
find_package(MPI QUIET)
if(NOT MPI_FOUND)
    return()
endif()

# set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME mpip)

option(TIMEMORY_GENERATE_MPIP "Generate the mpip via parsing mpi.h for PMPI functions" OFF)
mark_as_advanced(TIMEMORY_GENERATE_MPIP)

set(_MPI_DIRS
    ${MPI_INCLUDE_DIRS}
    ${MPI_C_HEADER_DIR}
    ${MPI_CXX_HEADER_DIR}
    ${MPI_ROOT_DIR})

find_file(MPI_HEADER
    NAMES mpi.h
    HINTS ${_MPI_DIRS}
    PATHS ${_MPI_DIRS})

if(NOT MPI_HEADER AND TIMEMORY_GENERATE_MPIP)
    message(WARNING "${PROJECT_NAME} could not find mpi.h in ${_MPI_DIRS}. Skipping generation...")
    set(TIMEMORY_GENERATE_MPIP OFF)
endif()

if(TIMEMORY_GENERATE_MPIP)

    configure_file(${PROJECT_SOURCE_DIR}/create_mpi_gotcha.sh
        ${PROJECT_BINARY_DIR}/create_mpi_gotcha.sh @ONLY)

    execute_process(
        COMMAND ${PROJECT_BINARY_DIR}/create_mpi_gotcha.sh
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        OUTPUT_FILE ${PROJECT_BINARY_DIR}/_timemory-mpip.cpp
        ERROR_VARIABLE ERR_MSG
        RESULT_VARIABLE RET
    )

    # will only copy if different
    configure_file(${PROJECT_BINARY_DIR}/_timemory-mpip.cpp
        ${PROJECT_BINARY_DIR}/timemory-mpip.cpp COPYONLY)

    if(NOT EXISTS ${PROJECT_BINARY_DIR}/timemory-mpip.cpp)
        message(STATUS "Missing file '${PROJECT_BINARY_DIR}/timemory-mpip.cpp'...")
        return()
    endif()

    if(NOT RET GREATER 0)
        add_library(timemory-mpip SHARED ${PROJECT_BINARY_DIR}/timemory-mpip.cpp)
    else()
        message(WARNING "Executing '${PROJECT_BINARY_DIR}/create_mpi_gotcha.sh' returned ${RET}")
    endif()

else()

    add_library(timemory-mpip SHARED ${PROJECT_SOURCE_DIR}/timemory-mpip.cpp)

endif()

if(TARGET timemory-mpip)
    # public link targets
    target_link_libraries(timemory-mpip
        PUBLIC
            timemory-headers
            timemory-cxx-shared
            timemory-mpi
            timemory-gotcha)
    # private link targets
    target_link_libraries(timemory-mpip
        PRIVATE
            timemory-compile-options
            timemory-arch
            timemory-vector)
    # use rpath
    set_target_properties(timemory-mpip PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
    # installation
    install(TARGETS timemory-mpip DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
