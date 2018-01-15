cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)

#----------------------------------------------------------------------------
project(TiMemory-Example1)

find_package(TiMemory REQUIRED)
foreach(_DIR ${TiMemory_INCLUDE_DIRS})
    include_directories(SYSTEM ${_DIR})
endforeach(_DIR ${TiMemory_INCLUDE_DIRS})


#----------------------------------------------------------------------------
# Add the executable, and link it to the TiMemory libraries
#
add_executable(timemory_test_timing test_timing.cpp)
target_link_libraries(timemory_test_timing ${TiMemory_LIBRARIES})
