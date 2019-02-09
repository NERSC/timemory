cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

################################################################################
#
#   CMake settings
#
################################################################################

file(READ "${CMAKE_CURRENT_LIST_DIR}/VERSION" VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" VERSION_STRING "${VERSION_STRING}")
string(REGEX REPLACE "[A-Za-z].*" "" VERSION_STRING "${VERSION_STRING}")
set(TIMEMORY_VERSION "${VERSION_STRING}" CACHE STRING "Version of TiMemory project" FORCE)
message(STATUS "TiMemory version ${TIMEMORY_VERSION}")

cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0042 NEW)
#cmake_policy(SET CMP0054 NEW)
project(TiMemory LANGUAGES C CXX VERSION ${TIMEMORY_VERSION})

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules
    ${CMAKE_CURRENT_LIST_DIR}/source/python/pybind11/tools
    ${CMAKE_MODULE_PATH})

foreach(_TYPE MAJOR MINOR PATCH)
    set(TIMEMORY_VERSION_${_TYPE} ${PROJECT_VERSION_${_TYPE}} CACHE STRING "Timemory ${_TYPE} version" FORCE)
    mark_as_advanced(TIMEMORY_VERSION_${_TYPE})
endforeach(_TYPE MAJOR MINOR PATCH)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME development)
mark_as_advanced(TIMEMORY_VERSION)

if(SKBUILD)
    set(TIMEMORY_SETUP_PY OFF CACHE BOOL "Built with scikit-build" FORCE)
endif()

################################################################################

include(MacroUtilities)
include(Options)
include(Compilers)
include(GNUInstallDirs)
include(ProjectSettings)
include(BuildSettings)
include(Packages)
include(ClangFormat)

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
