cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)

#----------------------------------------------------------------------------
project(TiMemory-Example1)

if("${CMAKE_PROJECT_NAME}" STREQUAL "TiMemory")
    include_directories(${CMAKE_SOURCE_DIR}/source)
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
add_executable(timemory_test_timing test_timing.cpp)
target_link_libraries(timemory_test_timing ${TiMemory_LIBRARIES})

install(TARGETS timemory_test_timing DESTINATION bin)
