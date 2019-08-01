# include guard
include_guard(DIRECTORY)

##########################################################################################
#
#        TiMemory Options
#
##########################################################################################


include(MacroUtilities)
include(CheckLanguage)

set(_FEATURE )
if(NOT ${PROJECT_NAME}_MASTER_PROJECT)
    set(_FEATURE NO_FEATURE)
endif()

set(SANITIZER_TYPE leak CACHE STRING "Sanitizer type")
set(_USE_PAPI OFF)
if(UNIX AND NOT APPLE)
    set(_USE_PAPI ON)
endif()

set(_USE_COVERAGE OFF)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(_USE_COVERAGE ON)
endif()

# Check if CUDA can be enabled
if(NOT DEFINED TIMEMORY_USE_CUDA OR TIMEMORY_USE_CUDA)
    set(_USE_CUDA ON)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
        else()
            message(STATUS "No CUDA support")
            set(_USE_CUDA OFF)
        endif()
    else()
        set(_USE_CUDA OFF)
    endif()
else()
    set(_USE_CUDA OFF)
endif()

set(_TLS_DESCRIPT "Thread-local static model: 'global-dynamic', 'local-dynamic', 'initial-exec', 'local-exec'")
set(TIMEMORY_TLS_MODEL "initial-exec" CACHE STRING "${_TLS_DESCRIPT}")

# CMake options
add_feature(CMAKE_BUILD_TYPE "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)")
add_feature(CMAKE_INSTALL_PREFIX "Installation prefix")
add_feature(CMAKE_C_STANDARD "C language standard")
add_feature(CMAKE_CXX_STANDARD "C++ language standard")
add_feature(CMAKE_CUDA_STANDARD "CUDA language standard")

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")
set(BUILD_STATIC_LIBS ON CACHE BOOL "Build static libraries")

add_feature(BUILD_SHARED_LIBS "Build shared libraries")
add_feature(BUILD_STATIC_LIBS "Build static libraries")

if(${PROJECT_NAME}_MASTER_PROJECT OR TIMEMORY_LANGUAGE_STANDARDS)
    # standard
    set(CMAKE_C_STANDARD 11 CACHE STRING "C language standard")
    set(CMAKE_CXX_STANDARD 11 CACHE STRING "CXX language standard")
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD} CACHE STRING "CUDA language standard")

    # standard required
    add_option(CMAKE_C_STANDARD_REQUIRED "Require C language standard" ON)
    add_option(CMAKE_CXX_STANDARD_REQUIRED "Require C++ language standard" ON)
    add_option(CMAKE_CUDA_STANDARD_REQUIRED "Require C++ language standard" ON)

    # extensions
    add_option(CMAKE_C_EXTENSIONS "C language standard extensions (e.g. gnu11)" OFF)
    add_option(CMAKE_CXX_EXTENSIONS "C++ language standard (e.g. gnu++11)" OFF)
    add_option(CMAKE_CUDA_EXTENSIONS "CUDA language standard (e.g. gnu++11)" OFF)

else()
    add_feature(CMAKE_C_STANDARD_REQUIRED "Require C language standard")
    add_feature(CMAKE_CXX_STANDARD_REQUIRED "Require C++ language standard")
    add_feature(CMAKE_CUDA_STANDARD_REQUIRED "Require C++ language standard")
    add_feature(CMAKE_C_EXTENSIONS "C language standard extensions (e.g. gnu11)")
    add_feature(CMAKE_CXX_EXTENSIONS "C++ language standard (e.g. gnu++11)")
    add_feature(CMAKE_CUDA_EXTENSIONS "CUDA language standard (e.g. gnu++11)")
endif()

add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Embed RPATH using link path" ON)

# Build settings
add_option(TIMEMORY_DEVELOPER_INSTALL
    "Python developer installation from setup.py"  OFF ${_FEATURE})
add_option(TIMEMORY_DOXYGEN_DOCS
    "Make a `doc` make target"  OFF ${_FEATURE})
add_option(TIMEMORY_BUILD_EXAMPLES
    "Build the examples"  OFF)
add_option(TIMEMORY_BUILD_C
    "Build the C compatible library"  ${${PROJECT_NAME}_MASTER_PROJECT})
add_option(TIMEMORY_BUILD_PYTHON
    "Build Python binds for ${PROJECT_NAME}" ${${PROJECT_NAME}_MASTER_PROJECT})
add_option(TIMEMORY_BUILD_LTO
    "Enable link-time optimizations in build" OFF)
add_option(TIMEMORY_BUILD_TOOLS
    "Enable building tools"  ${${PROJECT_NAME}_MASTER_PROJECT})
add_option(TIMEMORY_BUILD_EXTERN_TEMPLATES
    "Pre-compile list of templates for extern" ${${PROJECT_NAME}_MASTER_PROJECT})
add_option(TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS
    "Add extra optimization flags" OFF)
add_option(TIMEMORY_BUILD_GTEST
    "Enable GoogleTest" OFF)

# Features

if(${PROJECT_NAME}_MASTER_PROJECT)
    add_feature(${PROJECT_NAME}_C_FLAGS "C compiler flags")
    add_feature(${PROJECT_NAME}_CXX_FLAGS "C++ compiler flags")
    add_feature(TIMEMORY_INSTALL_PREFIX "${PROJECT_NAME} installation")
endif()

# TiMemory options
add_option(TIMEMORY_USE_EXCEPTIONS
    "Signal handler throws exceptions (default: exit)" OFF  ${_FEATURE})
add_option(TIMEMORY_USE_EXTERN_INIT
    "Do initialization in library instead of headers" OFF)
add_option(TIMEMORY_USE_MPI "Enable MPI usage"
    ON ${_FEATURE})
add_option(TIMEMORY_USE_SANITIZER
    "Enable -fsanitize flag (=${SANITIZER_TYPE})" OFF ${_FEATURE})
add_option(TIMEMORY_USE_PAPI
    "Enable PAPI" ${_USE_PAPI} ${_FEATURE})
add_option(TIMEMORY_USE_CLANG_TIDY
    "Enable running clang-tidy" OFF ${_FEATURE})
add_option(TIMEMORY_USE_COVERAGE
    "Enable code-coverage" ${_USE_COVERAGE} ${_FEATURE})
add_option(TIMEMORY_USE_GPERF
    "Enable gperf-tools" OFF)
add_option(TIMEMORY_USE_ARCH
    "Enable architecture flags" OFF ${_FEATURE})
add_option(TIMEMORY_USE_CUDA
    "Enable CUDA option for GPU measurements" ${_USE_CUDA} ${_FEATURE})
add_option(TIMEMORY_USE_CUPTI
    "Enable CUPTI profiling for NVIDIA GPUs" ${_USE_CUDA} ${_FEATURE})

# disable these for Debug builds
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(TIMEMORY_BUILD_LTO OFF)
    set(TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS OFF)
    set(TIMEMORY_USE_ARCH OFF)
endif()

add_feature(TIMEMORY_TLS_MODEL "${_TLS_DESCRIPT}")
unset(_TLS_DESCRIPT)

# cereal options
add_option(WITH_WERROR "Compile with '-Werror' C++ compiler flag" OFF NO_FEATURE)
add_option(THREAD_SAFE "Compile Cereal with THREAD_SAFE option" ON NO_FEATURE)
add_option(JUST_INSTALL_CEREAL "Skip testing of Cereal" ON NO_FEATURE)
add_option(SKIP_PORTABILITY_TEST "Skip Cereal portability test" ON NO_FEATURE)

if(TIMEMORY_DOXYGEN_DOCS)
    add_option(TIMEMORY_BUILD_DOXYGEN "Include `doc` make target in all" OFF NO_FEATURE)
    mark_as_advanced(TIMEMORY_BUILD_DOXYGEN)
endif()

if(TIMEMORY_BUILD_PYTHON)
    set(PYBIND11_INSTALL OFF CACHE BOOL "Don't install Pybind11")
endif()

# clang-tidy
if(TIMEMORY_USE_CLANG_TIDY)
    find_program(CLANG_TIDY_COMMAND NAMES clang-tidy)
    add_feature(CLANG_TIDY_COMMAND "Path to clang-tidy command")
    if(NOT CLANG_TIDY_COMMAND)
        message(WARNING "TIMEMORY_USE_CLANG_TIDY is ON but clang-tidy is not found!")
        set(TIMEMORY_USE_CLANG_TIDY OFF)
    else()
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")

        # Create a preprocessor definition that depends on .clang-tidy content so
        # the compile command will change when .clang-tidy changes.  This ensures
        # that a subsequent build re-runs clang-tidy on all sources even if they
        # do not otherwise need to be recompiled.  Nothing actually uses this
        # definition.  We add it to targets on which we run clang-tidy just to
        # get the build dependency on the .clang-tidy file.
        file(SHA1 ${PROJECT_SOURCE_DIR}/.clang-tidy clang_tidy_sha1)
        set(CLANG_TIDY_DEFINITIONS "CLANG_TIDY_SHA1=${clang_tidy_sha1}")
        unset(clang_tidy_sha1)
    endif()
    configure_file(${PROJECT_SOURCE_DIR}/.clang-tidy ${PROJECT_SOURCE_DIR}/.clang-tidy COPYONLY)
endif()
