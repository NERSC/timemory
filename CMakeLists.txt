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
if(NOT CMAKE_VERSION VERSION_LESS 3.13)
    cmake_policy(SET CMP0079 NEW)
endif()
# this gets annoying
if(TIMEMORY_BUILD_GTEST)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL
        "Suppress Warnings that are meant for the author of the CMakeLists.txt files")
endif()

#----------------------------------------------------------------------------------------#
#   versioning
#----------------------------------------------------------------------------------------#

file(READ "${CMAKE_CURRENT_LIST_DIR}/VERSION" FULL_VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" FULL_VERSION_STRING "${FULL_VERSION_STRING}")
string(REGEX REPLACE "[A-Za-z].*" "" VERSION_STRING "${FULL_VERSION_STRING}")
set(TIMEMORY_VERSION "${VERSION_STRING}")
message(STATUS "timemory version ${TIMEMORY_VERSION} (${FULL_VERSION_STRING})")
set(TIMEMORY_VERSION_STRING "${FULL_VERSION_STRING}")
string(REPLACE "." ";" VERSION_LIST "${VERSION_STRING}")
LIST(GET VERSION_LIST 0 TIMEMORY_VERSION_MAJOR)
LIST(GET VERSION_LIST 1 TIMEMORY_VERSION_MINOR)
LIST(GET VERSION_LIST 2 TIMEMORY_VERSION_PATCH)

math(EXPR TIMEMORY_VERSION_CODE
    "${TIMEMORY_VERSION_MAJOR} * 10000 + ${TIMEMORY_VERSION_MINOR} * 100 + ${TIMEMORY_VERSION_PATCH}")

#----------------------------------------------------------------------------------------#
#   setup.py
#----------------------------------------------------------------------------------------#

if(SKBUILD)
    set(CMAKE_INSTALL_LIBDIR lib)
endif()

#----------------------------------------------------------------------------------------#
#   project
#----------------------------------------------------------------------------------------#

project(timemory LANGUAGES C CXX VERSION ${TIMEMORY_VERSION})

configure_file(${PROJECT_SOURCE_DIR}/source/timemory/version.h.in
    ${PROJECT_SOURCE_DIR}/source/timemory/version.h)

# Check if project is being used directly or via add_subdirectory
set(${PROJECT_NAME}_MASTER_PROJECT ON)
if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(${PROJECT_NAME}_MASTER_PROJECT OFF)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules
    ${CMAKE_CURRENT_LIST_DIR}/external/pybind11/tools
    ${CMAKE_MODULE_PATH})
set(CMAKE_DIRECTORY_LABELS "${PROJECT_NAME}")
set(CMAKE_INSTALL_MESSAGE LAZY)

foreach(_TYPE MAJOR MINOR PATCH)
    set(TIMEMORY_VERSION_${_TYPE} ${PROJECT_VERSION_${_TYPE}})
endforeach(_TYPE MAJOR MINOR PATCH)
set(LIBNAME timemory)

#----------------------------------------------------------------------------------------#
# set the output directory (critical for Windows and Xcode)
#
foreach(_TYPE ARCHIVE LIBRARY RUNTIME)
    set(_BIN_DIR ${CMAKE_BINARY_DIR})
    if(NOT ${PROJECT_NAME}_MASTER_PROJECT)
        set(_BIN_DIR ${PROJECT_BINARY_DIR})
    endif()
    if(WIN32)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${_BIN_DIR}/outputs/runtime)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${_BIN_DIR}/outputs/library)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_BIN_DIR}/outputs/archive)
    else()
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${_BIN_DIR})
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${_BIN_DIR})
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${_BIN_DIR})
    endif()
endforeach()

#----------------------------------------------------------------------------------------#
#   configuration
#----------------------------------------------------------------------------------------#

include(MacroUtilities)
include(Options)

#----------------------------------------------------------------------------------------#
#   handle some additional configuration before other includes
#----------------------------------------------------------------------------------------#

set(_BUILD_SHARED_CXX ${BUILD_SHARED_LIBS})
set(_BUILD_STATIC_CXX ${BUILD_STATIC_LIBS})

option(TIMEMORY_BUILD_QUIET "Suppress author warnings without -Wno-dev" OFF)
mark_as_advanced(TIMEMORY_BUILD_QUIET)

if(SKBUILD)
    set(_BUILD_SHARED_CXX ON)
else()
    if(TIMEMORY_BUILD_PYTHON AND NOT BUILD_SHARED_LIBS)
        if(NOT TIMEMORY_BUILD_QUIET)
            message(AUTHOR_WARNING "BUILD_SHARED_LIBS=OFF --> disabling TIMEMORY_BUILD_PYTHON...")
            set(TIMEMORY_BUILD_PYTHON OFF)
        endif()
    endif()
endif()

#----------------------------------------------------------------------------------------#
#   thread-local static settings.
#   NOTE: if building Python, the thread-local storage MUST be global-dynamic
#----------------------------------------------------------------------------------------#

set(_TLS_DESCRIPT "Thread-local static model: 'global-dynamic', 'local-dynamic', 'initial-exec', 'local-exec'")
set(_TLS_OPTIONS "global-dynamic" "local-dynamic" "initial-exec" "local-exec")

if(SKBUILD OR TIMEMORY_BUILD_PYTHON)
    set(TIMEMORY_TLS_MODEL "global-dynamic" CACHE STRING "${_TLS_DESCRIPT}")
    # ensure local override
    set(TIMEMORY_TLS_MODEL "global-dynamic")
else()
    set(TIMEMORY_TLS_MODEL "initial-exec" CACHE STRING "${_TLS_DESCRIPT}")
endif()

set_property(CACHE TIMEMORY_TLS_MODEL PROPERTY STRINGS "${_TLS_OPTIONS}")
if(NOT "${TIMEMORY_TLS_MODEL}" IN_LIST _TLS_OPTIONS)
    message(FATAL_ERROR "TIMEMORY_TLS_MODEL must be one of: \"${_TLS_OPTIONS}\"")
endif()

add_feature(TIMEMORY_TLS_MODEL "${_TLS_DESCRIPT}")
unset(_TLS_DESCRIPT)
unset(_TLS_OPTIONS)

#----------------------------------------------------------------------------------------#
#   rest of configuration
#----------------------------------------------------------------------------------------#

include(ClangFormat)
include(Compilers)
include(GNUInstallDirs)
include(ProjectSettings)
include(BuildSettings)
include(Packages)
include(CMakeParseArguments)

#----------------------------------------------------------------------------------------#
#   timemory exports
#----------------------------------------------------------------------------------------#

install(TARGETS ${TIMEMORY_INTERFACE_LIBRARIES}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    EXPORT ${PROJECT_NAME}-library-depends)

if(TIMEMORY_BUILD_CALIPER)
    set(_CALIPER_TARGETS)
    foreach(CALIPER_TARGET caliper caliper-serial caliper-tools-util caliper-mpi)
        if(TARGET ${CALIPER_TARGET})
            list(APPEND TIMEMORY_ADDITIONAL_EXPORT_TARGETS ${CALIPER_TARGET})
        endif()
    endforeach()
    list(APPEND TIMEMORY_ADDITIONAL_EXPORT_TARGETS ${_CALIPER_TARGETS})
    install(TARGETS ${_CALIPER_TARGETS}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        EXPORT ${PROJECT_NAME}-library-depends)
endif()

#----------------------------------------------------------------------------------------#
#   timemory source
#----------------------------------------------------------------------------------------#
# activate clang-tidy if enabled
_timemory_activate_clang_tidy()

add_subdirectory(source)

# Install the export set for use with the install-tree
install(EXPORT ${PROJECT_NAME}-library-depends
    DESTINATION ${CMAKE_INSTALL_CONFIGDIR}
    NAMESPACE ${PROJECT_NAME}::)

#----------------------------------------------------------------------------------------#
#   Examples and Testing
#----------------------------------------------------------------------------------------#

if(TIMEMORY_BUILD_EXAMPLES)
    set(timemory_DIR "${CMAKE_BINARY_DIR}" CACHE INTERNAL "timemory install directory")
    add_subdirectory(examples)
endif()

#----------------------------------------------------------------------------------------#
#   Documentation
#----------------------------------------------------------------------------------------#

if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")

    include(Documentation)

    if(TIMEMORY_BUILD_DOCS)
        SET(CMAKE_INSTALL_MESSAGE NEVER)
        Generate_Documentation(Doxyfile.${PROJECT_NAME})
        SET(CMAKE_INSTALL_MESSAGE LAZY)
    endif()

    print_features()
endif()
