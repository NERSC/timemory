cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(timemory-Caliper-Example LANGUAGES CXX)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-caliper-example)
set(COMPONENTS compile-options analysis-tools caliper)
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

add_executable(test_caliper test_caliper.cpp)
target_link_libraries(test_caliper timemory-caliper-example)
install(TARGETS test_caliper DESTINATION bin)

if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "/usr/bin/env python")
endif()

configure_file(${PROJECT_SOURCE_DIR}/test_caliper.py
    ${CMAKE_BINARY_DIR}/test_python_caliper @ONLY)

install(
    FILES ${CMAKE_BINARY_DIR}/test_python_caliper
    DESTINATION bin
    PERMISSIONS
        OWNER_EXECUTE OWNER_READ OWNER_WRITE
        GROUP_EXECUTE GROUP_READ
        WORLD_EXECUTE WORLD_READ)
