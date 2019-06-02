##########################################################################################
#       CMake settings
##########################################################################################

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(WARNING "In-source build")
endif()

cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0042 NEW)

#----------------------------------------------------------------------------------------#
#   versioning
#----------------------------------------------------------------------------------------#

file(READ "${CMAKE_CURRENT_LIST_DIR}/VERSION" VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" VERSION_STRING "${VERSION_STRING}")
string(REGEX REPLACE "[A-Za-z].*" "" VERSION_STRING "${VERSION_STRING}")
set(TIMEMORY_VERSION "${VERSION_STRING}" CACHE STRING "Version of TiMemory project" FORCE)
message(STATUS "TiMemory version ${TIMEMORY_VERSION}")

#----------------------------------------------------------------------------------------#
#   project
#----------------------------------------------------------------------------------------#

project(TiMemory LANGUAGES C CXX VERSION ${TIMEMORY_VERSION})

# Check if project is being used directly or via add_subdirectory
set(${PROJECT_NAME}_MASTER_PROJECT ON)
if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(${PROJECT_NAME}_MASTER_PROJECT OFF)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules
    ${CMAKE_CURRENT_LIST_DIR}/source/python/pybind11/tools
    ${CMAKE_MODULE_PATH})

foreach(_TYPE MAJOR MINOR PATCH)
    set(TIMEMORY_VERSION_${_TYPE} ${PROJECT_VERSION_${_TYPE}} CACHE STRING "${PROJECT_NAME} ${_TYPE} version" FORCE)
    mark_as_advanced(TIMEMORY_VERSION_${_TYPE})
endforeach(_TYPE MAJOR MINOR PATCH)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME development)
mark_as_advanced(TIMEMORY_VERSION)

if(SKBUILD)
    set(TIMEMORY_SETUP_PY OFF CACHE BOOL "Built with scikit-build" FORCE)
endif()

set(CMAKE_DIRECTORY_LABELS "${PROJECT_NAME}")
set(LIBNAME timemory)

#----------------------------------------------------------------------------------------#
#   configuration
#----------------------------------------------------------------------------------------#

include(MacroUtilities)
include(Options)
include(Compilers)
include(GNUInstallDirs)
include(ProjectSettings)
include(BuildSettings)
include(Packages)
include(ClangFormat)

#----------------------------------------------------------------------------------------#
#   TiMemory source
#----------------------------------------------------------------------------------------#

add_subdirectory(source)

#----------------------------------------------------------------------------------------#
#   Package setup
#----------------------------------------------------------------------------------------#

include(PackageConfigure)

#----------------------------------------------------------------------------------------#
#   Examples and Testing
#----------------------------------------------------------------------------------------#

set(TiMemory_DIR "${CMAKE_BINARY_DIR}" CACHE STRING "TiMemory install directory" FORCE)
add_subdirectory(examples)
enable_testing()
