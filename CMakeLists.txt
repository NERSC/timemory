cmake_minimum_required(VERSION 2.8.12)

################################################################################
#
#   CMake settings
#
################################################################################

file(READ "${CMAKE_CURRENT_LIST_DIR}/.version.txt" VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" VERSION_STRING "${VERSION_STRING}")
string(REGEX REPLACE "[A-Za-z].*" "" VERSION_STRING "${VERSION_STRING}")
set(TIMEMORY_VERSION "${VERSION_STRING}" CACHE STRING "Version of TiMemory project" FORCE)
message(STATUS "TiMemory version ${TIMEMORY_VERSION}")

if(CMAKE_VERSION VERSION_LESS 3.1.3)
    set(GOOD_CMAKE OFF CACHE INTERNAL "CMake does not support CMAKE_CXX_STANDARD" FORCE)
else(CMAKE_VERSION VERSION_LESS 3.1.3)
    set(GOOD_CMAKE ON CACHE INTERNAL "CMake supports CMAKE_CXX_STANDARD" FORCE)
endif(CMAKE_VERSION VERSION_LESS 3.1.3)

if(GOOD_CMAKE)
    cmake_policy(SET CMP0048 NEW)
    cmake_policy(SET CMP0042 NEW)
    #cmake_policy(SET CMP0054 NEW)
    project(TiMemory LANGUAGES C CXX VERSION ${TIMEMORY_VERSION})
else(GOOD_CMAKE)
    project(TiMemory C CXX)
    set(PROJECT_VERSION ${TIMEMORY_VERSION})
    string(REPLACE "." ";" "${PROJECT_VERSION}" _VERSION_LIST)
    list(GET _VERSION_LIST 0 PROJECT_VERSION_MAJOR)
    list(GET _VERSION_LIST 1 PROJECT_VERSION_MINOR)
    list(GET _VERSION_LIST 2 PROJECT_VERSION_PATCH)
endif(GOOD_CMAKE)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})

foreach(_TYPE MAJOR MINOR PATCH)
    set(TIMEMORY_VERSION_${_TYPE} ${PROJECT_VERSION_${_TYPE}} CACHE STRING "Timemory ${_TYPE} version" FORCE)
    mark_as_advanced(TIMEMORY_VERSION_${_TYPE})
endforeach(_TYPE MAJOR MINOR PATCH)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME development)
mark_as_advanced(TIMEMORY_VERSION)

################################################################################

include(MacroUtilities)
include(Options)
include(Compilers)
include(GNUInstallDirs)
include(ProjectSettings)
include(BuildSettings)
include(Packages)

################################################################################
#   TiMemory source
################################################################################

add_subdirectory(source)

################################################################################
#   Package setup
################################################################################

include(PackageConfigure)

################################################################################
#   Examples and Testing
################################################################################

set(TiMemory_DIR "${CMAKE_BINARY_DIR}" CACHE STRING 
    "TiMemory install directory" FORCE)
add_subdirectory(examples)
include(Testing)
