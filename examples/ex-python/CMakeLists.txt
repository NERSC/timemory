cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(timemory-Python-Example LANGUAGES C CXX)

if("${CMAKE_PROJECT_NAME}" STREQUAL "timemory")
    if(NOT TIMEMORY_USE_PYTHON)
        return()
    endif()
endif()

if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "/usr/bin/env python")
endif()

configure_file(${PROJECT_SOURCE_DIR}/ex_sample.py
    ${CMAKE_BINARY_DIR}/ex_python_sample @ONLY)

  configure_file(${PROJECT_SOURCE_DIR}/ex_profiler.py
    ${CMAKE_BINARY_DIR}/ex_python_profiler @ONLY)
  
install(
    FILES
        ${CMAKE_BINARY_DIR}/ex_python_sample
        ${CMAKE_BINARY_DIR}/ex_python_profiler
    DESTINATION bin
    PERMISSIONS
        OWNER_EXECUTE OWNER_READ OWNER_WRITE
        GROUP_EXECUTE GROUP_READ
        WORLD_EXECUTE WORLD_READ)
