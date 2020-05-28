# include guard
include_guard(DIRECTORY)

##########################################################################################
#
#        timemory Options
#
##########################################################################################

include(MacroUtilities)
include(CheckLanguage)

function(DEFINE_DEFAULT_OPTION VAR VAL)
    if(TIMEMORY_REQUIRE_PACKAGES)
        set(${VAR} OFF PARENT_SCOPE)
    else()
        set(${VAR} ${VAL} PARENT_SCOPE)
    endif()
    set_property(GLOBAL APPEND PROPERTY DEFAULT_OPTION_VARIABLES ${VAR})
endfunction()

set(_FEATURE )
set(_USE_PAPI OFF)
set(_USE_COVERAGE OFF)
set(_BUILD_OPT OFF)
set(_BUILD_CALIPER ON)
set(_NON_APPLE_UNIX OFF)
set(_DEFAULT_BUILD_SHARED ON)
set(_DEFAULT_BUILD_STATIC ON)

set(SANITIZER_TYPE leak CACHE STRING "Sanitizer type")
set(TIMEMORY_gperftools_COMPONENTS "profiler" CACHE STRING "gperftools components")
set(TIMEMORY_gperftools_COMPONENTS_OPTIONS
    "profiler;tcmalloc;tcmalloc_and_profiler;tcmalloc_debug;tcmalloc_minimal;tcmalloc_minimal_debug")
set_property(CACHE TIMEMORY_gperftools_COMPONENTS PROPERTY STRINGS
    "${TIMEMORY_gperftools_COMPONENTS_OPTIONS}")

if(NOT ${PROJECT_NAME}_MASTER_PROJECT)
    set(_FEATURE NO_FEATURE)
endif()

if(UNIX AND NOT APPLE)
    set(_NON_APPLE_UNIX ON)
    set(_USE_PAPI ON)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(_BUILD_OPT ON)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(_USE_COVERAGE ON)
endif()

if(WIN32)
    set(_BUILD_CALIPER OFF)
endif()

# Check if CUDA can be enabled
if(NOT DEFINED TIMEMORY_USE_CUDA OR TIMEMORY_USE_CUDA)
    set(_USE_CUDA ON)
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

# if already defined, set default for shared to OFF
if(DEFINED BUILD_STATIC_LIBS AND BUILD_STATIC_LIBS)
    set(_DEFAULT_BUILD_STATIC ON)
    set(_DEFAULT_BUILD_SHARED OFF)
endif()

# if already defined, set default for shared to OFF
if(DEFINED BUILD_SHARED_LIBS AND BUILD_SHARED_LIBS)
    set(_DEFAULT_BUILD_STATIC OFF)
    set(_DEFAULT_BUILD_SHARED ON)
endif()

# something got messed up, so reset
if(NOT _DEFAULT_BUILD_SHARED AND NOT _DEFAULT_BUILD_STATIC)
    set(_DEFAULT_BUILD_SHARED ON)
    set(_DEFAULT_BUILD_STATIC ON)
endif()

# except is setup.py, always default static to off
if(SKBUILD)
    set(_DEFAULT_BUILD_SHARED ON)
    set(_DEFAULT_BUILD_STATIC OFF)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" _CONFIG)

# CMake options
add_feature(CMAKE_BUILD_TYPE "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)")
add_feature(CMAKE_INSTALL_PREFIX "Installation prefix")
add_feature(CMAKE_C_STANDARD "C language standard")
add_feature(CMAKE_CXX_STANDARD "C++ language standard")
add_feature(CMAKE_CUDA_STANDARD "CUDA language standard")
add_feature(CMAKE_C_FLAGS "C build flags")
add_feature(CMAKE_CXX_FLAGS "C++ build flags")
add_feature(CMAKE_CUDA_FLAGS "CUDA build flags")
add_feature(CMAKE_C_FLAGS_${_CONFIG} "C optimization type build flags")
add_feature(CMAKE_CXX_FLAGS_${_CONFIG} "C++ optimization type build flags")
add_feature(CMAKE_CUDA_FLAGS_${_CONFIG} "CUDA optimization type build flags")

add_option(BUILD_SHARED_LIBS "Build shared libraries" ${_DEFAULT_BUILD_SHARED})
add_option(BUILD_STATIC_LIBS "Build static libraries" ${_DEFAULT_BUILD_STATIC})
add_option(TIMEMORY_SKIP_BUILD "Disable building any libraries" OFF)

if(TIMEMORY_SKIP_BUILD)
    # local override
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_STATIC_LIBS OFF)
endif()

if(NOT BUILD_SHARED_LIBS AND NOT BUILD_STATIC_LIBS)
    # local override
    set(TIMEMORY_BUILD_C OFF)
    set(TIMEMORY_BUILD_PYTHON OFF)
    set(TIMEMORY_USE_PYTHON OFF)
    set(TIMEMORY_BUILD_KOKKOS_TOOLS OFF)
    set(TIMEMORY_BUILD_DYNINST_TOOLS OFF)
    set(TIMEMORY_BUILD_MPIP_LIBRARY OFF)
    set(TIMEMORY_BUILD_OMPT_LIBRARY OFF)
endif()

if(NOT BUILD_SHARED_LIBS AND NOT BUILD_STATIC_LIBS AND NOT TIMEMORY_SKIP_BUILD)
    message(STATUS "")
    message(STATUS "Set TIMEMORY_SKIP_BUILD=ON instead of BUILD_SHARED_LIBS=OFF and BUILD_STATIC_LIBS=OFF")
    message(STATUS "")
    message(FATAL_ERROR "Confusing settings")
endif()

# if(BUILD_STATIC_LIBS AND NOT BUILD_SHARED_LIBS)
#    set(CMAKE_FIND_LIBRARY_SUFFIXES .a .so .dylib)
# endif()

if(${PROJECT_NAME}_MASTER_PROJECT OR TIMEMORY_LANGUAGE_STANDARDS)
    if("${CMAKE_CXX_STANDARD}" LESS 14)
        unset(CMAKE_CXX_STANDARD CACHE)
    endif()

    if("${CMAKE_CUDA_STANDARD}" LESS 14)
        unset(CMAKE_CUDA_STANDARD CACHE)
    endif()
endif()

if(${PROJECT_NAME}_MASTER_PROJECT OR TIMEMORY_LANGUAGE_STANDARDS)
    # standard
    set(CMAKE_C_STANDARD 11 CACHE STRING "C language standard")
    set(CMAKE_CXX_STANDARD 14 CACHE STRING "CXX language standard")
    set(CMAKE_CUDA_STANDARD 14 CACHE STRING "CUDA language standard")

    # standard required
    add_option(CMAKE_C_STANDARD_REQUIRED "Require C language standard" ON)
    add_option(CMAKE_CXX_STANDARD_REQUIRED "Require C++ language standard" ON)
    add_option(CMAKE_CUDA_STANDARD_REQUIRED "Require C++ language standard" ON)

    # extensions
    add_option(CMAKE_C_EXTENSIONS "C language standard extensions (e.g. gnu11)" OFF)
    add_option(CMAKE_CXX_EXTENSIONS "C++ language standard (e.g. gnu++14)" OFF)
    add_option(CMAKE_CUDA_EXTENSIONS "CUDA language standard (e.g. gnu++14)" OFF)
else()
    add_feature(CMAKE_C_STANDARD_REQUIRED "Require C language standard")
    add_feature(CMAKE_CXX_STANDARD_REQUIRED "Require C++ language standard")
    add_feature(CMAKE_CUDA_STANDARD_REQUIRED "Require C++ language standard")
    add_feature(CMAKE_C_EXTENSIONS "C language standard extensions (e.g. gnu11)")
    add_feature(CMAKE_CXX_EXTENSIONS "C++ language standard (e.g. gnu++14)")
    add_feature(CMAKE_CUDA_EXTENSIONS "CUDA language standard (e.g. gnu++14)")
endif()

add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Embed RPATH using link path" ON)

# Build settings
add_option(TIMEMORY_BUILD_DOCS
    "Make a `doc` make target"  OFF)
add_option(TIMEMORY_BUILD_TESTING
    "Enable testing" OFF)
add_option(TIMEMORY_BUILD_GOOGLE_TEST
    "Enable GoogleTest" ${TIMEMORY_BUILD_TESTING})
add_option(TIMEMORY_BUILD_EXAMPLES
    "Build the examples"  ${TIMEMORY_BUILD_TESTING})
add_option(TIMEMORY_BUILD_C
    "Build the C compatible library" ${${PROJECT_NAME}_MASTER_PROJECT})
add_option(TIMEMORY_BUILD_PYTHON
    "Build Python binds for ${PROJECT_NAME}" ${${PROJECT_NAME}_MASTER_PROJECT})
add_option(TIMEMORY_BUILD_LTO
    "Enable link-time optimizations in build" OFF)
add_option(TIMEMORY_BUILD_TOOLS
    "Enable building tools" ${${PROJECT_NAME}_MASTER_PROJECT})
add_option(TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS
    "Add extra optimization flags" ${_BUILD_OPT})
add_option(TIMEMORY_BUILD_CALIPER
    "Enable building Caliper submodule (set to OFF for external)" ${_BUILD_CALIPER})
add_option(TIMEMORY_BUILD_OMPT
    "Enable building OpenMP-Tools from submodule" OFF)
add_option(TIMEMORY_BUILD_DEVELOPER
    "Enable building with developer flags" OFF)
add_option(TIMEMORY_FORCE_GPERFTOOLS_PYTHON
    "Enable gperftools + Python (may cause termination errors)" OFF)
add_option(TIMEMORY_BUILD_QUIET
    "Disable verbose messages" OFF NO_FEATURE)
add_option(TIMEMORY_REQUIRE_PACKAGES
    "All find_package(...) use REQUIRED" OFF)
if(_NON_APPLE_UNIX)
    add_option(TIMEMORY_BUILD_GOTCHA
        "Enable building GOTCHA (set to OFF for external)" ON)
endif()

if(TIMEMORY_BUILD_QUIET)
    set(TIMEMORY_FIND_QUIETLY QUIET)
endif()

if(TIMEMORY_REQUIRE_PACKAGES)
    set(TIMEMORY_FIND_REQUIREMENT REQUIRED)
endif()

# Features
if(${PROJECT_NAME}_MASTER_PROJECT)
    add_feature(${PROJECT_NAME}_C_FLAGS "C compiler flags")
    add_feature(${PROJECT_NAME}_CXX_FLAGS "C++ compiler flags")
    add_feature(${PROJECT_NAME}_CUDA_FLAGS "CUDA compiler flags")
endif()

define_default_option(_MPI ON)
define_default_option(_UPCXX ON)
define_default_option(_TAU ON)
define_default_option(_PAPI ${_USE_PAPI})
define_default_option(_GPERFTOOLS ON)
define_default_option(_VTUNE ON)
define_default_option(_CUDA ${_USE_CUDA})
define_default_option(_CALIPER ${_BUILD_CALIPER})
define_default_option(_PYTHON ${TIMEMORY_BUILD_PYTHON})
define_default_option(_DYNINST ON)
define_default_option(_ALLINEA_MAP ON)
define_default_option(_CRAYPAT ON)
define_default_option(_OMPT ON)
define_default_option(_LIKWID ${_NON_APPLE_UNIX})
define_default_option(_GOTCHA ${_NON_APPLE_UNIX})

# timemory options
add_option(TIMEMORY_USE_STATISTICS
    "Enable statistics by default" ON)
add_option(TIMEMORY_USE_MPI
    "Enable MPI usage" ${_MPI})
add_option(TIMEMORY_USE_UPCXX
    "Enable UPCXX usage (MPI support takes precedence)" ${_UPCXX})
add_option(TIMEMORY_USE_SANITIZER
    "Enable -fsanitize flag (=${SANITIZER_TYPE})" OFF)
add_option(TIMEMORY_USE_TAU
    "Enable TAU marking API" ${_TAU})
add_option(TIMEMORY_USE_PAPI
    "Enable PAPI" ${_PAPI})
add_option(TIMEMORY_USE_CLANG_TIDY
    "Enable running clang-tidy" OFF)
add_option(TIMEMORY_USE_COVERAGE
    "Enable code-coverage" ${_USE_COVERAGE})
add_option(TIMEMORY_USE_GPERFTOOLS
    "Enable gperftools" ${_GPERFTOOLS})
add_option(TIMEMORY_USE_GPERFTOOLS_STATIC
    "Enable gperftools static targets (enable if gperftools library are built with -fPIC)" OFF)
add_option(TIMEMORY_USE_ARCH
    "Enable architecture flags" OFF)
add_option(TIMEMORY_USE_VTUNE
    "Enable VTune marking API" ${_VTUNE})
add_option(TIMEMORY_USE_CUDA
    "Enable CUDA option for GPU measurements" ${_CUDA})
add_option(TIMEMORY_USE_NVTX
    "Enable NVTX marking API" ${_CUDA})
add_option(TIMEMORY_USE_CUPTI
    "Enable CUPTI profiling for NVIDIA GPUs" ${_CUDA})
add_option(TIMEMORY_USE_CALIPER
    "Enable Caliper" ${_CALIPER})
add_option(TIMEMORY_USE_PYTHON
    "Enable Python" ${_PYTHON})
add_option(TIMEMORY_USE_COMPILE_TIMING
    "Enable -ftime-report for compilation times" OFF)
add_option(TIMEMORY_USE_DYNINST
    "Enable dynamic instrumentation" ${_DYNINST})
add_option(TIMEMORY_USE_ALLINEA_MAP
    "Enable control for AllineaMAP sampler" ${_ALLINEA_MAP})
add_option(TIMEMORY_USE_CRAYPAT
    "Enable CrayPAT support" ${_CRAYPAT})
add_option(TIMEMORY_USE_OMPT
    "Enable OpenMP tooling" ${_OMPT})
add_option(TIMEMORY_USE_LIKWID
    "Enable LIKWID marker forwarding" ${_LIKWID})
add_option(TIMEMORY_USE_GOTCHA
    "Enable GOTCHA" ${_GOTCHA})
if(CMAKE_CXX_COMPILER_IS_CLANG)
    add_option(TIMEMORY_USE_XRAY
        "Enable XRay instrumentation" OFF)
endif()


# disable these for Debug builds
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(TIMEMORY_BUILD_LTO OFF)
    set(TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS OFF)
endif()

if(${PROJECT_NAME}_MASTER_PROJECT)
    add_feature(TIMEMORY_gperftools_COMPONENTS "gperftool components")
endif()

if(TIMEMORY_USE_CUDA)
    add_option(TIMEMORY_DISABLE_CUDA_HALF "Disable half/half2 if CUDA_ARCH < 60" OFF)
endif()

# cereal options
add_option(WITH_WERROR "Compile with '-Werror' C++ compiler flag" OFF NO_FEATURE)
add_option(THREAD_SAFE "Compile Cereal with THREAD_SAFE option" ON NO_FEATURE)
add_option(JUST_INSTALL_CEREAL "Skip testing of Cereal" ON NO_FEATURE)
add_option(SKIP_PORTABILITY_TEST "Skip Cereal portability test" ON NO_FEATURE)

if(TIMEMORY_BUILD_DOCS)
    add_option(TIMEMORY_BUILD_DOXYGEN "Include `doc` make target in all" OFF NO_FEATURE)
    mark_as_advanced(TIMEMORY_BUILD_DOXYGEN)
endif()

if(TIMEMORY_BUILD_PYTHON)
    set(PYBIND11_INSTALL ON CACHE BOOL "Don't install Pybind11")
else()
    set(PYBIND11_INSTALL OFF)
endif()

# clang-tidy
macro(_TIMEMORY_ACTIVATE_CLANG_TIDY)
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
            # configure_file(${PROJECT_SOURCE_DIR}/.clang-tidy ${PROJECT_SOURCE_DIR}/.clang-tidy COPYONLY)
        endif()
    endif()
endmacro()

option(TIMEMORY_SOURCE_GROUP "Enable source_group" OFF)
mark_as_advanced(TIMEMORY_SOURCE_GROUP)

# these variables conflict with variables in examples, leading to things like: -lON flags
get_property(DEFAULT_OPTION_VARIABLES GLOBAL PROPERTY DEFAULT_OPTION_VARIABLES)
foreach(_VAR ${DEFAULT_OPTION_VARIABLES})
    # message(STATUS "Reseting: ${_VAR} :: ${${_VAR}}")
    unset(${_VAR})
    # message(STATUS "Result: ${_VAR} :: ${${_VAR}}")
endforeach()
