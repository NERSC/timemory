
################################################################################
#
#        TiMemory Options
#
################################################################################

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

set(_TLS_DESCRIPT "Thread-local static model: 'global-dynamic', 'local-dynamic', 'initial-exec', 'local-exec'")
set(TIMEMORY_TLS_MODEL "initial-exec" CACHE STRING "${_TLS_DESCRIPT}")

# CMake options
add_option(CMAKE_C_STANDARD_REQUIRED "Require C standard" ON)
add_option(CMAKE_CXX_STANDARD_REQUIRED "Require C++ standard" ON)
add_option(CMAKE_CXX_EXTENSIONS "Build with CXX extensions (e.g. gnu++11)" OFF)
add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Embed RPATH using link path" ON)
add_option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Build settings
add_option(TIMEMORY_DEVELOPER_INSTALL "Python developer installation from setup.py" OFF ${_FEATURE})
add_option(TIMEMORY_DOXYGEN_DOCS "Make a `doc` make target" OFF ${_FEATURE})
add_option(TIMEMORY_BUILD_EXAMPLES "Build the examples" OFF ${_FEATURE})
add_option(TIMEMORY_BUILD_C "Build the C compatible library" ON ${_FEATURE})
add_option(TIMEMORY_BUILD_PYTHON "Build Python binds for ${PROJECT_NAME}" ON ${_FEATURE})
add_option(TIMEMORY_BUILD_LTO "Enable link-time optimizations in build" OFF ${_FEATURE})
add_option(TIMEMORY_BUILD_TOOLS "Enable building tools" ON ${_FEATURE})

# Features
add_feature(CMAKE_C_STANDARD "C language standard")
add_feature(CMAKE_CXX_STANDARD "C++ language standard")
add_feature(CMAKE_BUILD_TYPE "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)")
add_feature(CMAKE_INSTALL_PREFIX "Installation prefix")
add_feature(${PROJECT_NAME}_DEFINITIONS "${PROJECT_NAME} compile definitions")
if(${PROJECT_NAME}_MASTER_PROJECT)
    add_feature(${PROJECT_NAME}_C_FLAGS "C compiler flags")
    add_feature(${PROJECT_NAME}_CXX_FLAGS "C++ compiler flags")
    add_feature(TIMEMORY_INSTALL_PREFIX "${PROJECT_NAME} installation")
endif()

# TiMemory options
add_option(TIMEMORY_USE_EXCEPTIONS "Signal handler throws exceptions (default: exit)" OFF ${_FEATURE})
add_option(TIMEMORY_USE_MPI "Enable MPI usage" ON ${_FEATURE})
add_option(TIMEMORY_USE_SANITIZER "Enable -fsanitize flag (=${SANITIZER_TYPE})" OFF)
add_option(TIMEMORY_USE_PAPI "Enable PAPI" ${_USE_PAPI})
add_option(TIMEMORY_USE_CLANG_TIDY "Enable running clang-tidy" OFF)
add_option(TIMEMORY_USE_COVERAGE "Enable code-coverage" ${_USE_COVERAGE} ${_FEATURE})
add_option(TIMEMORY_USE_GPERF "Enable gperf-tools" OFF)
add_option(TIMEMORY_USE_CUDA "Enable CUDA option for GPU measurements" ${_USE_CUDA})
add_option(TIMEMORY_USE_CUPTI "Enable CUPTI profiling for NVIDIA GPUs" ${_USE_CUDA})
if(TIMEMORY_USE_CUDA AND ${PROJECT_NAME}_MASTER_PROJECT)
    add_feature(CMAKE_CUDA_STANDARD "CUDA STL standard")
    add_feature(${PROJECT_NAME}_CUDA_FLAGS "CUDA NVCC compiler flags")
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

if(TIMEMORY_USE_EXCEPTIONS)
    list(APPEND ${PROJECT_NAME}_DEFINITIONS TIMEMORY_EXCEPTIONS)
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
