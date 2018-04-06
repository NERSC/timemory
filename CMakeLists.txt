cmake_minimum_required(VERSION 2.8.12)

################################################################################
#
#   CMake settings
#
################################################################################

if(CMAKE_VERSION VERSION_LESS 3.1.3)
    set(GOOD_CMAKE OFF CACHE BOOL "Old version of CMake not support CMAKE_CXX_STANDARD" FORCE)
else(CMAKE_VERSION VERSION_LESS 3.1.3)
    set(GOOD_CMAKE ON CACHE BOOL "Old version of CMake not support CMAKE_CXX_STANDARD" FORCE)
endif(CMAKE_VERSION VERSION_LESS 3.1.3)

file(READ "${CMAKE_CURRENT_LIST_DIR}/.version.txt" VERSION_STRING
    LIMIT_COUNT 1)
set(TIMEMORY_VERSION 1.3.0 CACHE STRING "Version of TiMemory project" FORCE)
message(STATUS "TiMemory version ${TIMEMORY_VERSION}")
mark_as_advanced(TIMEMORY_VERSION)

if(GOOD_CMAKE)
    cmake_policy(SET CMP0048 NEW)
    cmake_policy(SET CMP0042 NEW)
    project(TiMemory LANGUAGES C CXX VERSION ${TIMEMORY_VERSION})
else(GOOD_CMAKE)
    project(TiMemory C CXX)
    set(PROJECT_VERSION ${TIMEMORY_VERSION})
    string(REPLACE "." ";" "${PROJECT_VERSION}" _VERSION_LIST)
    list(GET _VERSION_LIST 0 PROJECT_VERSION_MAJOR)
    list(GET _VERSION_LIST 1 PROJECT_VERSION_MINOR)
    list(GET _VERSION_LIST 2 PROJECT_VERSION_PATCH)
endif(GOOD_CMAKE)

foreach(_TYPE MAJOR MINOR PATCH)
    set(TIMEMORY_VERSION_${_TYPE} ${PROJECT_VERSION_${_TYPE}} CACHE STRING "Timemory ${_TYPE} version" FORCE)
    mark_as_advanced(TIMEMORY_VERSION_${_TYPE})
endforeach(_TYPE MAJOR MINOR PATCH)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules
    ${CMAKE_MODULE_PATH})

if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set(SUBPROJECT OFF)
else("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set(SUBPROJECT ON)
endif("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME development)

################################################################################

include(MacroUtilities)
include(Compilers)
include(GNUInstallDirs)
include(ProjectSettings)
include(BuildSettings)
include(Packages)

################################################################################
#   TiMemory source
################################################################################

add_subdirectory(source)
add_subdirectory(examples)

################################################################################
#   Package setup
################################################################################

include(PackageConfigure)
