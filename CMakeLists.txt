
##########################################################################################
#       CMake settings
##########################################################################################

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(WARNING "In-source build")
endif()

option(TIMEMORY_CCACHE_BUILD "Enable ccache build" OFF)
mark_as_advanced(TIMEMORY_CCACHE_BUILD)
if(TIMEMORY_CCACHE_BUILD)
    find_program(TIMEMORY_CCACHE_EXE ccache PATH_SUFFIXES bin)
    if(TIMEMORY_CCACHE_EXE)
        if(NOT EXISTS "${TIMEMORY_CCACHE_EXE}")
            message(WARNING
                "TIMEMORY_CCACHE_BUILD is ON but TIMEMORY_CCACHE_EXE (${TIMEMORY_CCACHE_EXE}) does not exist!")
        else()
            set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${TIMEMORY_CCACHE_EXE}")
        endif()
    endif()
endif()

cmake_policy(SET CMP0010 NEW)
cmake_policy(SET CMP0022 NEW)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0053 NEW)
cmake_policy(SET CMP0063 NEW)
if(NOT CMAKE_VERSION VERSION_LESS 3.13)
    cmake_policy(SET CMP0077 NEW)
    cmake_policy(SET CMP0079 NEW)
endif()
if(NOT CMAKE_VERSION VERSION_LESS 3.14)
    cmake_policy(SET CMP0082 NEW)
endif()
if(NOT CMAKE_VERSION VERSION_LESS 3.18)
    cmake_policy(SET CMP0104 OLD)
endif()
# make sure testing enabled
if(TIMEMORY_BUILD_MINIMAL_TESTING)
    set(TIMEMORY_BUILD_TESTING ON)
endif()
# this gets annoying
if(TIMEMORY_BUILD_GOOGLE_TEST OR TIMEMORY_BUILD_TESTING)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL
        "Suppress Warnings that are meant for the author of the CMakeLists.txt files")
endif()
# override any cache settings
if(TIMEMORY_BUILD_TESTING)
    set(TIMEMORY_BUILD_GOOGLE_TEST ON)
    if(NOT TIMEMORY_BUILD_MINIMAL_TESTING)
        set(TIMEMORY_BUILD_EXAMPLES ON)
    endif()
else()
    if(TIMEMORY_BUILD_MINIMAL_TESTING)
        set(TIMEMORY_BUILD_GOOGLE_TEST ON)
        set(TIMEMORY_BUILD_EXAMPLES OFF)
    endif()
endif()
#
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
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
set(TIMEMORY_VERSION
    "${TIMEMORY_VERSION_MAJOR}.${TIMEMORY_VERSION_MINOR}.${TIMEMORY_VERSION_PATCH}")

math(EXPR TIMEMORY_VERSION_CODE
    "${TIMEMORY_VERSION_MAJOR} * 10000 + ${TIMEMORY_VERSION_MINOR} * 100 + ${TIMEMORY_VERSION_PATCH}")

#----------------------------------------------------------------------------------------#
#   setup.py
#----------------------------------------------------------------------------------------#

if(SKBUILD)
    set(CMAKE_INSTALL_LIBDIR lib)
    set(TIMEMORY_USE_PYTHON ON CACHE BOOL "" FORCE)
    set(TIMEMORY_BUILD_PYTHON ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
endif()

#----------------------------------------------------------------------------------------#
#   project
#----------------------------------------------------------------------------------------#

project(timemory LANGUAGES C CXX VERSION ${TIMEMORY_VERSION})

# Check if project is being used directly or via add_subdirectory
set(${PROJECT_NAME}_MASTER_PROJECT ON)
if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(${PROJECT_NAME}_MASTER_PROJECT OFF)
endif()

# set these as the defaults
if(timemory_MASTER_PROJECT)
    set(CMAKE_ENABLE_EXPORTS ON CACHE BOOL "Executable exports symbols for loadable modules")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON CACHE BOOL
        "Append directories in the linker search path")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Build position independent code")
    mark_as_advanced(CMAKE_ENABLE_EXPORTS)
else()
    set(CMAKE_ENABLE_EXPORTS ON)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

find_package(Git QUIET)
if(Git_FOUND)
    execute_process(
        COMMAND             ${GIT_EXECUTABLE} describe --tags
        WORKING_DIRECTORY   ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE     TIMEMORY_GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
        COMMAND             ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY   ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE     TIMEMORY_GIT_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

# install directories
include(GNUInstallDirs)
# cmake installation folder -- change CMAKE_INSTALL_DATAROOTDIR to tweak this
set(CMAKE_INSTALL_CONFIGDIR  ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME})

if(UNIX AND NOT APPLE)
    set(CMAKE_INSTALL_RPATH
        "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}:\$ORIGIN/timemory:\$ORIGIN:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/timemory")
endif()

# create the full path version and generic path versions
foreach(_TYPE DATAROOT CMAKE INCLUDE LIB BIN MAN DOC)
    # generic "PROJECT_INSTALL_" variables (used by documentation)"
    set(PROJECT_INSTALL_${_TYPE}DIR ${CMAKE_INSTALL_${TYPE}DIR})
endforeach()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules
    ${CMAKE_CURRENT_LIST_DIR}/external/pybind11/tools
    ${CMAKE_MODULE_PATH})
set(CMAKE_DIRECTORY_LABELS "${PROJECT_NAME}")
set(CMAKE_INSTALL_MESSAGE LAZY)

foreach(_TYPE MAJOR MINOR PATCH)
    set(TIMEMORY_VERSION_${_TYPE} ${PROJECT_VERSION_${_TYPE}})
    set(${PROJECT_NAME}_VERSION_${_TYPE} ${PROJECT_VERSION_${_TYPE}})
endforeach()
set(timemory_VERSION ${PROJECT_VERSION})
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
    if((TIMEMORY_BUILD_PYTHON OR TIMEMORY_USE_PYTHON) AND NOT BUILD_SHARED_LIBS)
        if(NOT TIMEMORY_BUILD_QUIET)
            message(AUTHOR_WARNING "BUILD_SHARED_LIBS=OFF --> disabling TIMEMORY_BUILD_PYTHON...")
        endif()
        set(TIMEMORY_BUILD_PYTHON OFF)
        set(TIMEMORY_USE_PYTHON OFF)
    endif()
endif()

#----------------------------------------------------------------------------------------#
#   thread-local static settings.
#   NOTE: if building Python, the thread-local storage MUST be global-dynamic
#----------------------------------------------------------------------------------------#

set(_TLS_DESCRIPT "Thread-local static model: 'global-dynamic', 'local-dynamic', 'initial-exec', 'local-exec'")
set(_TLS_OPTIONS "global-dynamic" "local-dynamic" "initial-exec" "local-exec")

if(SKBUILD OR TIMEMORY_BUILD_PYTHON OR TIMEMORY_USE_PYTHON OR TIMEMORY_USE_DYNINST)
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

include(CMakeParseArguments)

message(STATUS "Adding clang-format...")
include(ClangFormat)

message(STATUS "Configuring compilers...")
include(Compilers)

message(STATUS "Configuring build settings...")
include(BuildSettings)

message(STATUS "Configuring external packages and interface libraries...")
include(Packages)

# only after submodules have been added
if(TIMEMORY_USE_COVERAGE)
    set(CMAKE_UNITY_BUILD OFF)
    set(TIMEMORY_UNITY_BUILD OFF CACHE BOOL "" FORCE)
else()
    set(CMAKE_UNITY_BUILD ${TIMEMORY_UNITY_BUILD})
endif()

# set these as the defaults
if(timemory_MASTER_PROJECT)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF CACHE BOOL
        "Add compile flag to hide symbols of inline functions")
    set(CMAKE_C_VISIBILITY_PRESET "default" CACHE STRING "Default visibility")
    set(CMAKE_CXX_VISIBILITY_PRESET "default" CACHE STRING "Default visibility")
else()
    set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
    set(CMAKE_C_VISIBILITY_PRESET "default")
    set(CMAKE_CXX_VISIBILITY_PRESET "default")
endif()

#----------------------------------------------------------------------------------------#
#   timemory exports
#----------------------------------------------------------------------------------------#

#message(STATUS "Exporting interface libraries...")
#install(TARGETS ${TIMEMORY_INTERFACE_LIBRARIES}
#    DESTINATION ${CMAKE_INSTALL_LIBDIR}
#    EXPORT ${PROJECT_NAME}-library-depends)

#----------------------------------------------------------------------------------------#
#   timemory source
#----------------------------------------------------------------------------------------#
if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    # activate clang-tidy if enabled
    _timemory_activate_clang_tidy()
endif()

set(timemory_DIR ${PROJECT_BINARY_DIR} CACHE PATH "Path to timemory" FORCE)

message(STATUS "Adding timemory (python)...")
add_subdirectory(timemory)

message(STATUS "Adding source...")
add_subdirectory(source)

message(STATUS "Adding external...")
add_subdirectory(external)

# Install the export set for use with the install-tree
message(STATUS "Exporting library dependencies...")
if(TIMEMORY_INSTALL_CONFIG)
    install(EXPORT ${PROJECT_NAME}-library-depends
        DESTINATION ${CMAKE_INSTALL_CONFIGDIR}
        NAMESPACE ${PROJECT_NAME}::
        OPTIONAL)
endif()

#----------------------------------------------------------------------------------------#
#   Examples and Testing
#----------------------------------------------------------------------------------------#

if(TIMEMORY_BUILD_EXAMPLES)
    message(STATUS "Adding examples...")
    add_subdirectory(examples)
elseif(TIMEMORY_USE_PYTHON AND TIMEMORY_CI)
    message(STATUS "Adding examples/ex-python...")
    # copies over some python scripts
    function(CONFIGURE_PYTHON_SCRIPT)
        foreach(_TYPE ${ARGN})
            set(FILENAME ex_python_${_TYPE})
            configure_file(
                ${PROJECT_SOURCE_DIR}/examples/ex-python/ex_${_TYPE}.py
                ${PROJECT_BINARY_DIR}/ex_python_${_TYPE}
                @ONLY)
        endforeach()
    endfunction()
    configure_python_script(sample tracer profiler general builtin external)
endif()

#----------------------------------------------------------------------------------------#
#   Documentation
#----------------------------------------------------------------------------------------#

if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")

    include(Documentation)

    if(TIMEMORY_BUILD_DOCS)
        message(STATUS "Adding documentation...")
        Generate_Documentation(Doxyfile.${PROJECT_NAME})
    endif()

    option(TIMEMORY_PRINT_FEATURES
        "Print the list of enabled/disabled options and interface libraries" ON)
    mark_as_advanced(TIMEMORY_PRINT_FEATURES)

    if(TIMEMORY_PRINT_FEATURES)
        print_features()
    endif()

endif()

#----------------------------------------------------------------------------------------#
#   Makefile.timemory.inc
#----------------------------------------------------------------------------------------#

option(TIMEMORY_GENERATE_MAKEFILE_INCLUDE
    "Generate a Makefile.timemory.inc for Makefile build systems" ${TIMEMORY_INSTALL_CONFIG})
mark_as_advanced(TIMEMORY_GENERATE_MAKEFILE_INCLUDE)

if(TIMEMORY_GENERATE_MAKEFILE_INCLUDE)
    include(GenerateMakefile)
endif()

# remove testing configure warnings:
#   "Manually-specified variables were not used by the project: ... etc. ..."
#
foreach(_EXAMPLE_OPTION MPI TAU CUDA PAPI CUPTI KOKKOS CALIPER)
    set(_IGNORE ${USE_${_EXAMPLE_OPTION}})
endforeach()
