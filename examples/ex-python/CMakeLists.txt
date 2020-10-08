cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(timemory-Python-Example LANGUAGES C CXX)

if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory")
    set(TARGET_BINARY_DIR ${CMAKE_BINARY_DIR})
    if(NOT TIMEMORY_USE_PYTHON)
        return()
    endif()
else()
    set(TARGET_BINARY_DIR ${PROJECT_BINARY_DIR})
endif()

if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "/usr/bin/env python")
endif()

foreach(_TYPE sample tracer profiler general builtin external)
    set(FILENAME ex_python_${_TYPE})
    configure_file(
        ${PROJECT_SOURCE_DIR}/ex_${_TYPE}.py
        ${TARGET_BINARY_DIR}/ex_python_${_TYPE}
        @ONLY)
    install(FILES   ${TARGET_BINARY_DIR}/ex_python_${_TYPE}
        DESTINATION bin
        PERMISSIONS
            OWNER_EXECUTE OWNER_READ OWNER_WRITE
            GROUP_EXECUTE GROUP_READ
            WORLD_EXECUTE WORLD_READ
        COMPONENT examples
        OPTIONAL)
endforeach()

if(TIMEMORY_BUILD_MINIMAL_TESTING)
    return()
endif()

if(NOT TARGET pybind11::module)
    find_package(pybind11 QUIET)
endif()

if(NOT TARGET pybind11::module)
    return()
endif()

find_package(MPI QUIET)
if(NOT MPI_FOUND)
    return()
endif()

set(_OMP)
find_package(OpenMP QUIET)
if(OpenMP_FOUND)
    set(_OMP OpenMP::OpenMP_CXX)
endif()

set(TIMEMORY_INTERFACE_LIBRARY timemory-ex-python-bindings)
find_package(timemory COMPONENTS headers OPTIONAL_COMPONENTS mpi cxx shared)

pybind11_add_module(libex_python_bindings ${PROJECT_SOURCE_DIR}/libex_bindings.cpp)

set(_COMPILE_OPTS) # for -Wno-attributes if available by the compiler
if(TARGET timemory-compile-options)
    set(_COMPILE_OPTS timemory-compile-options)
elseif(TARGET timemory::timemory-compile-options)
    set(_COMPILE_OPTS timemory::timemory-compile-options)
endif()

target_compile_definitions(libex_python_bindings PRIVATE USE_MPI)
target_link_libraries(libex_python_bindings PRIVATE
    MPI::MPI_C MPI::MPI_CXX ${_OMP} ${_COMPILE_OPTS})

configure_file(${PROJECT_SOURCE_DIR}/ex_bindings.py
    ${TARGET_BINARY_DIR}/ex_python_bindings @ONLY)

install(TARGETS libex_python_bindings DESTINATION bin
    COMPONENT examples OPTIONAL)

install(
    FILES
        ${TARGET_BINARY_DIR}/ex_python_bindings
    DESTINATION bin
    PERMISSIONS
        OWNER_EXECUTE OWNER_READ OWNER_WRITE
        GROUP_EXECUTE GROUP_READ
        WORLD_EXECUTE WORLD_READ
    COMPONENT examples
    OPTIONAL)
