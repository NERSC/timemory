cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

if(CMAKE_VERSION VERSION_LESS 3.1.3)
    set(GOOD_CMAKE OFF)
else(CMAKE_VERSION VERSION_LESS 3.1.3)
    set(GOOD_CMAKE ON)
endif(CMAKE_VERSION VERSION_LESS 3.1.3)

if(GOOD_CMAKE)
    cmake_policy(SET CMP0048 NEW)
endif(GOOD_CMAKE)
cmake_policy(SET CMP0042 NEW)

#----------------------------------------------------------------------------
project(TiMemory-Example1)

if(NOT "${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_SOURCE_DIR}")
    include_directories(${CMAKE_SOURCE_DIR}/source
        ${CMAKE_SOURCE_DIR}/source/cereal/include)
    set(TiMemory_LIBRARIES timemory-shared ${EXTERNAL_LIBRARIES})
else()
    find_package(TiMemory REQUIRED)
    foreach(_DIR ${TiMemory_INCLUDE_DIRS})
        include_directories(SYSTEM ${_DIR})
    endforeach(_DIR ${TiMemory_INCLUDE_DIRS})
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TiMemory_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TiMemory_CXX_FLAGS}")

#----------------------------------------------------------------------------
# Add the executable, and link it to the TiMemory libraries
#
add_executable(test_timing test_timing.cpp)
target_link_libraries(test_timing ${TiMemory_LIBRARIES})

if(NOT DEFINED SETUP_PY OR NOT SETUP_PY)
    install(TARGETS test_timing DESTINATION bin)
else(NOT DEFINED SETUP_PY OR NOT SETUP_PY)
endif(NOT DEFINED SETUP_PY OR NOT SETUP_PY)
