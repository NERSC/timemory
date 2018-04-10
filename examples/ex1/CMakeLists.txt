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

set(EXE_NAME test_timing)

if(NOT "${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    include_directories(${CMAKE_SOURCE_DIR}/source
        ${CMAKE_SOURCE_DIR}/source/cereal/include)
    set_property(TARGET timemory-static PROPERTY
        IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/TiMemoryBuild.cmake")
    set_property(TARGET timemory-shared
        PROPERTY IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/TiMemoryBuild.cmake")
    set(TiMemory_LIBRARIES timemory-shared ${EXTERNAL_LIBRARIES})
    set(PROJECT_DEPENDS timemory-static timemory-shared)
else()
    find_package(TiMemory REQUIRED)
    foreach(_DIR ${TiMemory_INCLUDE_DIRS})
        include_directories(SYSTEM ${_DIR})
    endforeach(_DIR ${TiMemory_INCLUDE_DIRS})
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
endif(NOT DEFINED CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TiMemory_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TiMemory_CXX_FLAGS}")

#----------------------------------------------------------------------------
# Add the executable, and link it to the TiMemory libraries
#
add_executable(${EXE_NAME} ${EXE_NAME}.cpp)
target_link_libraries(${EXE_NAME} ${TiMemory_LIBRARIES} ${EXTERNAL_LIBRARIES})

if(DEFINED PROJECT_DEPENDS)
    add_dependencies(${EXE_NAME} ${PROJECT_DEPENDS})
endif(DEFINED PROJECT_DEPENDS)

if(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
    install(TARGETS ${EXE_NAME} DESTINATION bin)
endif(NOT DEFINED TIMEMORY_SETUP_PY OR NOT TIMEMORY_SETUP_PY)
