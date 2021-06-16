# include guard
include_guard(DIRECTORY)

# placeholder folder PyCTest testing which will write it's own CTestTestfile.cmake
# this MUST come before "enable_testing"
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
enable_testing()

##########################################################################################
#
#                       External Packages are found here
#
##########################################################################################

add_interface_library(timemory-headers
    "Provides minimal set of include flags to compile with timemory")
add_interface_library(timemory-precompiled-headers
    "Provides timemory-headers + precompiles headers if CMAKE_VERSION >= 3.16")
add_interface_library(timemory-xml
    "Enables XML serialization support")
add_interface_library(timemory-extern
    "Enables pre-processor directive to ensure all extern templates are used")
add_interface_library(timemory-statistics
    "Enables statistics for all components which define TIMEMORY_STATISTICS_TYPE(...)")
add_interface_library(timemory-disable
    "Enables pre-processor directive for disabling timemory completely")
add_interface_library(timemory-default-disabled
    "Enables pre-processor directive for disabling timemory by default at runtime")

set(TIMEMORY_REQUIRED_INTERFACES
    timemory-headers)

add_interface_library(timemory-dmp
    "Enables the default distributed memory parallelism library (e.g. MPI, UPC++)")
add_interface_library(timemory-mpi
    "Enables MPI support")
add_interface_library(timemory-no-mpi-init
    "Disables the generation of MPI_Init and MPI_Init_thread symbols")
add_interface_library(timemory-upcxx
    "Enables UPC++ support")
add_interface_library(timemory-threading
    "Enables multithreading support")

add_interface_library(timemory-papi
    "Enables PAPI support")
add_interface_library(timemory-papi-static
    "Enables PAPI support + links to static library")
add_interface_library(timemory-cuda
    "Enables CUDA support")
add_interface_library(timemory-cuda-compiler
    "Enables some CUDA compiler flags")
add_interface_library(timemory-cupti
    "Enables CUPTI support (requires linking to libcuda)")
add_interface_library(timemory-cudart
    "Link to CUDA runtime (shared library)")
add_interface_library(timemory-cudart-device
    "Link to CUDA device runtime")
add_interface_library(timemory-cudart-static
    "Link to CUDA runtime (static library)")
add_interface_library(timemory-nccl
    "Enables CUDA NCCL support")
add_interface_library(timemory-nvml
    "Enables NVML support (NVIDIA)")
add_interface_library(timemory-caliper
    "Enables Caliper support")
add_interface_library(timemory-gotcha
    "Enables Gotcha support")
add_interface_library(timemory-likwid
    "Enables LIKWID support")
add_interface_library(timemory-vtune
    "Enables VTune support (ittnotify)")
add_interface_library(timemory-tau
    "Enables TAU support")
add_interface_library(timemory-ompt
    "Enables OpenMP-tools support")
add_interface_library(timemory-python
    "Enables python support (embedded interpreter)")
add_interface_library(timemory-plotting
    "Enables python plotting support (system call)")
add_interface_library(timemory-allinea-map
    "Enables Allinea-MAP support")
add_interface_library(timemory-craypat
    "Enables CrayPAT support")
add_interface_library(timemory-libunwind
    "Enables libunwind support")

add_interface_library(timemory-coverage
    "Enables code-coverage flags")
add_interface_library(timemory-gperftools
    "Enables user-selected gperftools component (${_GPERF_COMPONENTS})")

add_interface_library(timemory-roofline
    "Enables flags and libraries for proper roofline generation")
add_interface_library(timemory-cpu-roofline
    "Enables flags and libraries for proper CPU roofline generation")
add_interface_library(timemory-gpu-roofline
    "Enables flags and libraries for proper GPU roofline generation")
add_interface_library(timemory-roofline-options
    "Compiler flags for roofline generation")

add_interface_library(timemory-dyninst
    "Provides flags and libraries for Dyninst (dynamic instrumentation")

add_interface_library(timemory-mpip-library
    "Provides MPIP library for MPI performance analysis")
add_interface_library(timemory-ompt-library
    "Provides OMPT library for OpenMP performance analysis")
add_interface_library(timemory-ncclp-library
    "Provides NCCLP library for NCCL performance analysis")
add_interface_library(timemory-mallocp-library
    "Provides MALLOCP library for tracking memory allocations")
add_interface_library(timemory-compiler-instrument
    "Provides library for compiler instrumentation")

if(TIMEMORY_USE_MPI)
    target_link_libraries(timemory-mpip-library INTERFACE timemory-mpi timemory-gotcha)
endif()

if(TIMEMORY_USE_NCCL)
    target_link_libraries(timemory-ncclp-library INTERFACE timemory-nccl timemory-gotcha)
endif()

set(_DMP_LIBRARIES)

if(TIMEMORY_USE_MPI)
    list(APPEND _DMP_LIBRARIES timemory-mpi)
    target_link_libraries(timemory-dmp INTERFACE timemory-mpi)
endif()

if(TIMEMORY_USE_UPCXX)
    list(APPEND _DMP_LIBRARIES timemory-upcxx)
    target_link_libraries(timemory-dmp INTERFACE timemory-upcxx)
endif()

set(TIMEMORY_RUNTIME_INTERFACES
    #
    timemory-dmp
    timemory-threading
    #
)

set(TIMEMORY_EXTENSION_INTERFACES
    #
    timemory-statistics
    #
    timemory-cuda
    timemory-nccl
    timemory-nvml
    timemory-cupti
    timemory-cudart
    timemory-cudart-device
    #
    timemory-papi
    timemory-gperftools
    #
    timemory-python
    timemory-plotting
    #
    timemory-caliper
    timemory-gotcha
    timemory-likwid
    timemory-vtune
    timemory-tau
    timemory-ompt
    timemory-craypat
    timemory-allinea-map
    timemory-libunwind)

set(TIMEMORY_EXTERNAL_SHARED_INTERFACES
    timemory-threading
    timemory-statistics
    timemory-papi
    timemory-cuda
    timemory-cudart
    timemory-nccl
    timemory-nvml
    timemory-cupti
    timemory-cudart-device
    timemory-caliper
    timemory-gotcha
    timemory-likwid
    timemory-vtune
    timemory-tau
    timemory-ompt
    timemory-craypat
    timemory-allinea-map
    timemory-plotting
    timemory-libunwind
    ${_DMP_LIBRARIES})

set(TIMEMORY_EXTERNAL_STATIC_INTERFACES
    timemory-threading
    timemory-statistics
    timemory-papi
    timemory-cuda
    timemory-cudart-static
    timemory-nccl
    timemory-nvml
    timemory-cupti
    timemory-cudart-device
    timemory-caliper
    timemory-likwid
    timemory-vtune
    timemory-tau
    timemory-ompt
    timemory-craypat
    timemory-allinea-map
    timemory-plotting
    timemory-libunwind
    ${_DMP_LIBRARIES})

set(_GPERF_IN_LIBRARY OFF)
# if not python or force requested
if(NOT TIMEMORY_USE_PYTHON)
    list(APPEND TIMEMORY_EXTERNAL_SHARED_INTERFACES timemory-gperftools)
    list(APPEND TIMEMORY_EXTERNAL_STATIC_INTERFACES timemory-gperftools)
    set(_GPERF_IN_LIBRARY ON)
endif()

add_interface_library(timemory-extensions
    "Provides a single target for all the timemory extensions which were found")
target_link_libraries(timemory-extensions INTERFACE ${TIMEMORY_EXTENSION_INTERFACES})

add_interface_library(timemory-external-shared
    "Provides a single target for all the timemory extensions (shared libraries)")
target_link_libraries(timemory-external-shared INTERFACE ${TIMEMORY_EXTERNAL_SHARED_INTERFACES})

add_interface_library(timemory-external-static
    "Provides a single target for all the timemory extensions (static libraries)")
target_link_libraries(timemory-external-static INTERFACE ${TIMEMORY_EXTERNAL_STATIC_INTERFACES})

add_interface_library(timemory-analysis-tools
    "Internal. Provides sanitizer, gperftools-cpu, coverage, xray")

if(TIMEMORY_USE_SANITIZER)
    target_link_libraries(timemory-analysis-tools INTERFACE timemory-sanitizer)
endif()

if(TIMEMORY_USE_GPERFTOOLS AND NOT TIMEMORY_USE_COVERAGE)
    target_link_libraries(timemory-analysis-tools INTERFACE timemory-gperftools)
endif()

if(TIMEMORY_USE_COVERAGE)
    target_link_libraries(timemory-analysis-tools INTERFACE timemory-coverage)
endif()

if(TIMEMORY_USE_XRAY)
    target_link_libraries(timemory-analysis-tools INTERFACE timemory-xray)
endif()

# not exported
add_library(timemory-google-test INTERFACE)


#----------------------------------------------------------------------------------------#
#
#                           generate composite interface
#
#----------------------------------------------------------------------------------------#

function(GENERATE_COMPOSITE_INTERFACE _TARGET)
    # parse args
    if(NOT TARGET ${_TARGET})
        message(AUTHOR_WARNING "A non-existant target was passed to INFORM_EMPTY_INTERFACE: ${_TARGET}")
    endif()

    set(_FOUND ON)
    set(_LINK)

    foreach(_DEPENDS ${ARGN})
        if(${_DEPENDS} IN_LIST TIMEMORY_EMPTY_INTERFACE_LIBRARIES)
            timemory_message(STATUS  "[interface] '${_TARGET}' depends on '${_DEPENDS}' which is empty...")
            set(_FOUND OFF)
        else()
            list(APPEND _LINK ${_DEPENDS})
        endif()
    endforeach()

    if(_FOUND)
        target_link_libraries(${_TARGET} INTERFACE ${_LINK})
    else()
        add_disabled_interface(${_TARGET})
    endif()
endfunction()

#----------------------------------------------------------------------------------------#
#
#                               function for configuring
#                                 an interface library
#
#----------------------------------------------------------------------------------------#

function(find_package_interface)
    set(_option_args)
    set(_single_args NAME INTERFACE DESCRIPTION)
    set(_multiv_args FIND_ARGS INCLUDE_DIRS COMPILE_DEFINITIONS COMPILE_OPTIONS LINK_LIBRARIES)

    cmake_parse_arguments(PACKAGE
        "${_option_args}" "${_single_args}" "${_multiv_args}" ${ARGN})

    if("${PACKAGE_NAME}" STREQUAL "")
        message(FATAL_ERROR "find_package_interface :: missing variable: NAME")
    endif()

    if("${PACKAGE_INTERFACE}" STREQUAL "")
        message(FATAL_ERROR "find_package_interface (${PACKAGE_NAME}) :: missing variable: INTERFACE")
    endif()

    if(NOT TARGET ${PACKAGE_INTERFACE})
        add_library(${PACKAGE_INTERFACE} INTERFACE)
        add_library(${PROJECT_NAME}::${PACKAGE_INTERFACE} ALIAS ${PACKAGE_INTERFACE})
    endif()

    if("${PACKAGE_DESCRIPTION}" STREQUAL "")
        set(PACKAGE_DESCRIPTION "${PACKAGE_INTERFACE}")
    endif()

    # find the package
    find_package(${PACKAGE_NAME} ${PACKAGE_FIND_ARGS})

    if(${PACKAGE_NAME}_FOUND)
        # include the directories
        target_include_directories(${PACKAGE_INTERFACE} SYSTEM INTERFACE
            ${PACKAGE_INCLUDE_DIRS} ${${PACKAGE_NAME}_INCLUDE_DIRS})

        # link libraries
        target_link_libraries(${PACKAGE_INTERFACE} INTERFACE
            ${PACKAGE_LINK_LIBRARIES} ${${PACKAGE_NAME}_LIBRARIES})

        # add any compile definitions
        foreach(_DEF ${PACKAGE_COMPILE_DEFINITIONS})
            timemory_target_compile_definitions(${PACKAGE_INTERFACE} INTERFACE ${_DEF})
        endforeach()

        # add any compile-flags
        foreach(_FLAG ${PACKAGE_COMPILE_OPTIONS})
            add_target_flag_if_avail(${PACKAGE_INTERFACE} "${_FLAG}")
        endforeach()
    else()
        inform_empty_interface(${PACKAGE_INTERFACE} "${PACKAGE_DESCRIPTION}")
    endif()

endfunction()

#----------------------------------------------------------------------------------------#
#
#                               timemory headers
#
#----------------------------------------------------------------------------------------#

timemory_target_compile_definitions(timemory-disable INTERFACE
    TIMEMORY_ENABLED=0)

timemory_target_compile_definitions(timemory-default-disabled INTERFACE
    TIMEMORY_DEFAULT_ENABLED=false)

# this target is always linked whenever timemory is used via cmake
timemory_target_compile_definitions(timemory-headers INTERFACE TIMEMORY_CMAKE)

if(TIMEMORY_USE_WINSOCK)
    timemory_target_compile_definitions(timemory-headers INTERFACE TIMEMORY_USE_WINSOCK)
endif()

if(TIMEMORY_BUILD_TESTING)
    target_compile_definitions(timemory-headers INTERFACE $<BUILD_INTERFACE:TIMEMORY_INTERNAL_TESTING>)
endif()

target_include_directories(timemory-headers INTERFACE
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>)

target_include_directories(timemory-headers SYSTEM INTERFACE
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# dynamic linking library (searched for in BuildSettings)
if(dl_LIBRARY)
    target_link_libraries(timemory-headers INTERFACE ${dl_LIBRARY})
elseif(TIMEMORY_LINK_DL)
    target_link_libraries(timemory-headers INTERFACE dl)
endif()

# Realtime Extensions library (searched for in BuildSettings)
if(rt_LIBRARY)
    target_link_libraries(timemory-headers INTERFACE ${rt_LIBRARY})
elseif(TIMEMORY_LINK_RT)
    target_link_libraries(timemory-headers INTERFACE rt)
endif()

# include threading because of rooflines
target_link_libraries(timemory-headers INTERFACE timemory-threading)

if(TIMEMORY_USE_XML)
    target_link_libraries(timemory-headers INTERFACE timemory-xml)
endif()

# minimum: C++14
target_compile_features(timemory-headers INTERFACE
    cxx_std_${CMAKE_CXX_STANDARD}
    cxx_auto_type
    cxx_alias_templates
    cxx_constexpr
    cxx_decltype
    cxx_decltype_auto
    cxx_defaulted_functions
    cxx_delegating_constructors
    cxx_deleted_functions
    cxx_extern_templates
    cxx_generic_lambdas
    cxx_noexcept
    cxx_range_for
    cxx_return_type_deduction
    cxx_rvalue_references
    cxx_thread_local
    cxx_trailing_return_types
    cxx_variadic_macros
    cxx_variadic_templates
    cxx_template_template_parameters)

# Set CUDA at end in case we end up disabling it
if(NOT CMAKE_VERSION VERSION_LESS 3.17 AND TIMEMORY_USE_CUDA)
    if(DEFINED CMAKE_CUDA_KNOWN_FEATURES AND
       "cuda_std_${CMAKE_CUDA_STANDARD}" IN_LIST CMAKE_CUDA_KNOWN_FEATURES)
        target_compile_features(timemory-headers INTERFACE
            cuda_std_${CMAKE_CUDA_STANDARD})
    endif()
endif()

if(NOT TIMEMORY_PRECOMPILE_HEADERS)
    inform_empty_interface(timemory-precompiled-headers "Precompiled-headers for timemory")
else()
    if(BUILD_SHARED_LIBS)
        set(_EXTERNAL_INTERFACE timemory::timemory-external-shared)
    else()
        set(_EXTERNAL_INTERFACE timemory::timemory-external-static)
    endif()
    target_link_libraries(timemory-precompiled-headers INTERFACE
        timemory::timemory-headers
        timemory::timemory-vector
        timemory::timemory-plotting
        timemory::timemory-compile-options
        timemory::timemory-default-visibility
        ${TIMEMORY_RUNTIME_INTERFACES}
        ${_EXTERNAL_INTERFACE})
    file(GLOB_RECURSE timemory_precompiled_headers
        ${PROJECT_SOURCE_DIR}/source/timemory/backends*.hpp
        ${PROJECT_SOURCE_DIR}/source/timemory/environment*.hpp
        ${PROJECT_SOURCE_DIR}/source/timemory/ert*.hpp
        ${PROJECT_SOURCE_DIR}/source/timemory/hash*.hpp
        ${PROJECT_SOURCE_DIR}/source/timemory/manager*.hpp
        ${PROJECT_SOURCE_DIR}/source/timemory/mpl*.hpp
        ${PROJECT_SOURCE_DIR}/source/timemory/plotting*.hpp
        ${PROJECT_SOURCE_DIR}/source/timemory/settings*.hpp
        ${PROJECT_SOURCE_DIR}/source/timemory/storage*.hpp
        ${PROJECT_SOURCE_DIR}/source/timemory/tpls*.hpp
        ${PROJECT_SOURCE_DIR}/source/timemory/utility*.hpp
        ${PROJECT_SOURCE_DIR}/source/timemory/variadic*.hpp)
    timemory_target_precompile_headers(timemory-precompiled-headers
        FILES ${timemory_precompiled_headers})
endif()

# find modules
file(GLOB TIMEMORY_FIND_MODULES ${PROJECT_SOURCE_DIR}/cmake/Modules/Find*.cmake)
list(REMOVE_ITEM TIMEMORY_FIND_MODULES
    ${PROJECT_SOURCE_DIR}/cmake/Modules/FindPython3.cmake
    ${PROJECT_SOURCE_DIR}/cmake/Modules/FindPythonLibs.cmake)
if(TIMEMORY_INSTALL_CONFIG)
    install(FILES ${TIMEMORY_FIND_MODULES}
        DESTINATION ${CMAKE_INSTALL_CONFIGDIR}/Modules
        OPTIONAL)
endif()

#----------------------------------------------------------------------------------------#
#
#                        timemory extern initializaiton
#
#----------------------------------------------------------------------------------------#


if(NOT WIN32)
    timemory_target_compile_definitions(timemory-extern INTERFACE TIMEMORY_USE_EXTERN)
endif()


#----------------------------------------------------------------------------------------#
#
#                        timemory statistics
#
#----------------------------------------------------------------------------------------#


timemory_target_compile_definitions(timemory-statistics INTERFACE TIMEMORY_USE_STATISTICS)
if(TIMEMORY_USE_STATISTICS)
    target_link_libraries(timemory-headers INTERFACE timemory-statistics)
endif()


#----------------------------------------------------------------------------------------#
#
#                           Deprecated code
#
#----------------------------------------------------------------------------------------#


if(TIMEMORY_USE_DEPRECATED)
    timemory_target_compile_definitions(timemory-headers INTERFACE TIMEMORY_USE_DEPRECATED)
endif()

#----------------------------------------------------------------------------------------#
#
#                           Cereal (serialization library)
#
#----------------------------------------------------------------------------------------#


timemory_target_compile_definitions(timemory-xml INTERFACE TIMEMORY_USE_XML)


#----------------------------------------------------------------------------------------#
#
#                               Threading
#
#----------------------------------------------------------------------------------------#

if(NOT WIN32)
    set(CMAKE_THREAD_PREFER_PTHREAD ON)
    set(THREADS_PREFER_PTHREAD_FLAG OFF)
endif()

find_library(PTHREADS_LIBRARY pthread)
find_package(Threads ${TIMEMORY_FIND_QUIETLY} ${TIMEMORY_FIND_REQUIREMENT})

if(Threads_FOUND)
    target_link_libraries(timemory-threading INTERFACE ${CMAKE_THREAD_LIBS_INIT})
endif()

if(PTHREADS_LIBRARY AND NOT WIN32)
    target_link_libraries(timemory-threading INTERFACE ${PTHREADS_LIBRARY})
endif()


#----------------------------------------------------------------------------------------#
#
#                               MPI
#
#----------------------------------------------------------------------------------------#
# always try to find MPI even if it is not used
#

# MS-MPI standard install
if(WIN32)
    list(APPEND CMAKE_PREFIX_PATH "C:/Program\ Files\ (x86)/Microsoft\ SDKs/MPI"
        "C:/Program\ Files/Microsoft\ SDKs/MPI")
endif()

# MPI C compiler from environment
if(NOT "$ENV{MPICC}" STREQUAL "")
    set(MPI_C_COMPILER $ENV{MPICC} CACHE FILEPATH "MPI C compiler")
endif()

# MPI C++ compiler from environment
if(NOT "$ENV{MPICXX}" STREQUAL "")
    set(MPI_CXX_COMPILER $ENV{MPICXX} CACHE FILEPATH "MPI C++ compiler")
endif()

if(TIMEMORY_USE_MPI)
    find_package(MPI ${TIMEMORY_FIND_QUIETLY} ${TIMEMORY_FIND_REQUIREMENT})
else()
    find_package(MPI QUIET)
endif()

# interface to kill MPI init in headers
timemory_target_compile_definitions(timemory-no-mpi-init INTERFACE TIMEMORY_MPI_INIT=0)

if(MPI_FOUND)
    target_compile_definitions(timemory-mpi INTERFACE TIMEMORY_USE_MPI)

    foreach(_LANG CXX)
        if(TARGET MPI::MPI_${_LANG})
            target_link_libraries(timemory-mpi INTERFACE MPI::MPI_${_LANG})
        endif()
    endforeach()

    # used by python
    if(NOT MPIEXEC_EXECUTABLE AND MPIEXEC)
        set(MPIEXEC_EXECUTABLE ${MPIEXEC} CACHE FILEPATH "MPI executable")
    endif()

    # used by python
    if(NOT MPIEXEC_EXECUTABLE AND MPI_EXECUTABLE)
        set(MPIEXEC_EXECUTABLE ${MPI_EXECUTABLE} CACHE FILEPATH "MPI executable")
    endif()

    if(NOT TIMEMORY_USE_MPI_INIT)
        target_link_libraries(timemory-mpi INTERFACE timemory-no-mpi-init)
    endif()

    if(NOT "$ENV{CRAYPE_VERSION}" STREQUAL "")
        set(_PMI_INCLUDE "$ENV{CRAY_PMI_INCLUDE_OPTS}")
        set(_PMI_LINKOPT "$ENV{CRAY_PMI_POST_LINK_OPTS}")
        string(REGEX REPLACE "^-I" "" _PMI_INCLUDE "${_PMI_INCLUDE}")
        string(REGEX REPLACE "^-L" "" _PMI_LINKOPT "${_PMI_LINKOPT}")
        string(REGEX REPLACE "^-l" "" _PMI_LINKOPT "${_PMI_LINKOPT}")
        string(REPLACE " " ";" _PMI_INCLUDE "${_PMI_INCLUDE}")
        string(REPLACE " " ";" _PMI_LINKOPT "${_PMI_LINKOPT}")
        string(REPLACE ":" ";" _PMI_LIBPATH "$ENV{CRAY_LD_LIBRARY_PATH}")
        foreach(_DIR ${_PMI_INCLUDE} ${_PMI_LIBPATH})
            get_filename_component(_DIR "${_DIR}" DIRECTORY)
            list(APPEND _PMI_HINTS ${_DIR})
        endforeach()
        find_library(PMI_LIBRARY
            NAMES pmi
            PATHS ${_PMI_HINTS}
            HINTS ${_PMI_HINTS}
            PATH_SUFFIXES lib64 lib)
        if(PMI_LIBRARY)
            message(STATUS "Found PMI library: ${PMI_LIBRARY}")
            target_link_libraries(timemory-mpi INTERFACE ${PMI_LIBRARY})
        endif()
        unset(_PMI_INCLUDE)
        unset(_PMI_LINKOPT)
        unset(_PMI_HINTS)
        unset(_DIR)
    endif()
else()

    set(TIMEMORY_USE_MPI OFF)
    inform_empty_interface(timemory-mpi "MPI")

endif()


#----------------------------------------------------------------------------------------#
#
#                               UPC++
#
#----------------------------------------------------------------------------------------#
# always try to find UPC++ even if it is not used
#

if(TIMEMORY_USE_UPCXX)
    find_package(UPCXX ${TIMEMORY_FIND_QUIETLY} ${TIMEMORY_FIND_REQUIREMENT})
else()
    find_package(UPCXX QUIET)
endif()

if(UPCXX_FOUND)

    add_rpath(${UPCXX_LIBRARIES})
    target_link_libraries(timemory-upcxx INTERFACE ${UPCXX_LIBRARIES})
    target_compile_options(timemory-upcxx INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:${UPCXX_OPTIONS}>)
    target_compile_features(timemory-upcxx INTERFACE cxx_std_${UPCXX_CXX_STANDARD})
    target_include_directories(timemory-upcxx SYSTEM INTERFACE ${UPCXX_INCLUDE_DIRS})
    target_compile_definitions(timemory-upcxx INTERFACE ${UPCXX_DEFINITIONS})
    timemory_target_compile_definitions(timemory-upcxx INTERFACE TIMEMORY_USE_UPCXX)

    target_link_options(timemory-upcxx INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:${UPCXX_LINK_OPTIONS}>)

else()

    set(TIMEMORY_USE_UPCXX OFF)
    inform_empty_interface(timemory-upcxx "UPC++")

endif()


#----------------------------------------------------------------------------------------#
#
#                               PyBind11
#
#----------------------------------------------------------------------------------------#
# if using is enable but not internal pybind11 distribution
if(TIMEMORY_USE_PYTHON AND NOT TIMEMORY_BUILD_PYTHON)

    find_package(pybind11 ${TIMEMORY_FIND_REQUIREMENT})

    if(NOT pybind11_FOUND)
        set(TIMEMORY_USE_PYTHON OFF)
        set(TIMEMORY_BUILD_PYTHON OFF)
    else()
        set(TIMEMORY_PYTHON_VERSION "${PYBIND11_PYTHON_VERSION}" CACHE STRING
            "Python version for timemory")
    endif()

    if(NOT "${TIMEMORY_PYTHON_VERSION}" MATCHES "${PYBIND11_PYTHON_VERSION}*")
        message(STATUS "TIMEMORY_PYTHON_VERSION is set to ${TIMEMORY_PYTHON_VERSION}")
        message(STATUS "PYBIND11_PYTHON_VERSION is set to ${PYBIND11_PYTHON_VERSION}")
        message(FATAL_ERROR
            "Mismatched 'TIMEMORY_PYTHON_VERSION' and 'PYBIND11_PYTHON_VERSION'")
    endif()

endif()

if(TIMEMORY_USE_PYTHON)
    include(PythonConfig)
else()
    set(TIMEMORY_BUILD_PYTHON OFF)
    inform_empty_interface(timemory-python "Python embedded interpreter")
    inform_empty_interface(timemory-plotting "Python plotting from C++")
endif()


#----------------------------------------------------------------------------------------#
#
#                           Google Test
#
#----------------------------------------------------------------------------------------#

# MUST BE AFTER PythonConfig is included!
if(TIMEMORY_BUILD_GOOGLE_TEST)
    checkout_git_submodule(RECURSIVE
        RELATIVE_PATH external/google-test
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/googletest.git
        REPO_BRANCH timemory)

    # add google-test
    set(INSTALL_GTEST OFF CACHE BOOL "Install gtest")
    set(BUILD_GMOCK ON CACHE BOOL "Build gmock")
    if(APPLE)
        set(CMAKE_MACOSX_RPATH ON CACHE BOOL "Enable MACOS_RPATH on targets to suppress warnings")
        mark_as_advanced(CMAKE_MACOSX_RPATH)
    endif()
    add_subdirectory(${PROJECT_SOURCE_DIR}/external/google-test)
    target_link_libraries(timemory-google-test INTERFACE gtest gmock)
    target_include_directories(timemory-google-test SYSTEM INTERFACE
        ${PROJECT_SOURCE_DIR}/google-test/googletest/include
        ${PROJECT_SOURCE_DIR}/google-test/googlemock/include)
endif()


#----------------------------------------------------------------------------------------#
#
#                               PAPI
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_PAPI)
    find_package(PAPI ${TIMEMORY_FIND_QUIETLY} ${TIMEMORY_FIND_REQUIREMENT})
endif()

if(PAPI_FOUND)
    add_rpath(${PAPI_LIBRARIES})
    target_link_libraries(timemory-papi INTERFACE ${PAPI_LIBRARIES})
    target_link_libraries(timemory-papi-static INTERFACE ${PAPI_STATIC_LIBRARIES})
    target_include_directories(timemory-papi SYSTEM INTERFACE ${PAPI_INCLUDE_DIRS})
    target_include_directories(timemory-papi-static SYSTEM INTERFACE ${PAPI_INCLUDE_DIRS})
    timemory_target_compile_definitions(timemory-papi INTERFACE TIMEMORY_USE_PAPI)
    timemory_target_compile_definitions(timemory-papi-static INTERFACE TIMEMORY_USE_PAPI)
else()
    set(TIMEMORY_USE_PAPI OFF)
    inform_empty_interface(timemory-papi "PAPI (shared libraries)")
    inform_empty_interface(timemory-papi-static "PAPI (static libraries)")
    inform_empty_interface(timemory-cpu-roofline "CPU roofline")
endif()


#----------------------------------------------------------------------------------------#
#
#                               Coverage
#
#----------------------------------------------------------------------------------------#

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    find_library(GCOV_LIBRARY gcov ${TIMEMORY_FIND_QUIETLY})

    add_target_flag_if_avail(timemory-coverage "-fprofile-abs-path" "--coverage")
    add_target_flag(timemory-coverage "-fprofile-arcs" "-ftest-coverage" "-O0" "-g")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_target_flag(timemory-coverage "--coverage")
        target_link_options(timemory-coverage INTERFACE --coverage)
    else()
        target_link_options(timemory-coverage INTERFACE -fprofile-arcs)
    endif()

else()
    inform_empty_interface(timemory-coverage "coverage")
    set(TIMEMORY_USE_COVERAGE OFF)
endif()


#----------------------------------------------------------------------------------------#
#
#                                   CUDA
#
#----------------------------------------------------------------------------------------#

set(TIMEMORY_USE_NVTX ${TIMEMORY_USE_CUDA})
if(TIMEMORY_USE_CUDA)

    set(PROJECT_USE_CUDA_OPTION            TIMEMORY_USE_CUDA)
    set(PROJECT_CUDA_DEFINITION            TIMEMORY_USE_CUDA)
    set(PROJECT_CUDA_INTERFACE_PREFIX      timemory)
    set(PROJECT_CUDA_USE_HALF_OPTION       TIMEMORY_USE_CUDA_HALF)
    set(PROJECT_CUDA_USE_HALF_DEFINITION   TIMEMORY_USE_CUDA_HALF)

    include(CUDAConfig)

    find_package(NVTX ${TIMEMORY_FIND_QUIETLY})
    if(NVTX_FOUND)
        add_rpath(${NVTX_LIBRARIES})
        target_link_libraries(timemory-cuda INTERFACE ${NVTX_LIBRARIES})
        target_include_directories(timemory-cuda SYSTEM INTERFACE ${NVTX_INCLUDE_DIRS})
    endif()
    target_compile_definitions(timemory-cuda INTERFACE TIMEMORY_USE_NVTX)

    if(TIMEMORY_BUILD_LTO AND CMAKE_CUDA_COMPILER_IS_NVIDIA AND NOT CUDA_VERSION VERSION_LESS 11.2)
        add_target_cuda_flag(timemory-lto "-dlto")
        target_link_options(timemory-lto INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:$<$<CUDA_COMPILER_ID:NVIDIA>:-dlto>>)
    endif()

else()
    set(TIMEMORY_USE_CUDA OFF)
    set(TIMEMORY_USE_NVTX OFF)
    set(TIMEMORY_USE_CUPTI OFF)
    inform_empty_interface(timemory-cuda "CUDA")
    inform_empty_interface(timemory-cudart "CUDA Runtime (shared)")
    inform_empty_interface(timemory-cudart-device "CUDA Runtime (device)")
    inform_empty_interface(timemory-cudart-static "CUDA Runtime (static)")
endif()


#----------------------------------------------------------------------------------------#
#
#                               CUPTI
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_CUPTI)
    find_package(CUPTI ${TIMEMORY_FIND_REQUIREMENT})
endif()

if(CUPTI_FOUND)

    timemory_target_compile_definitions(timemory-cupti INTERFACE TIMEMORY_USE_CUPTI)

    target_include_directories(timemory-cupti SYSTEM INTERFACE
        ${CUPTI_INCLUDE_DIRS})

    target_link_libraries(timemory-cupti INTERFACE
        ${CUPTI_LIBRARIES}
        timemory-cuda
        # timemory-cudart
        timemory-cudart-device)

    target_link_directories(timemory-cupti INTERFACE
        $<INSTALL_INTERFACE:${CUPTI_LIBRARY_DIRS}>)

    set_target_properties(timemory-cupti PROPERTIES
        INTERFACE_INSTALL_RPATH                 ""
        INTERFACE_INSTALL_RPATH_USE_LINK_PATH   ${HAS_CUDA_DRIVER_LIBRARY})

    if(CUPTI_nvperf_host_FOUND AND CUPTI_nvperf_target_FOUND)
        timemory_target_compile_definitions(timemory-cupti INTERFACE
            TIMEMORY_USE_CUPTI_NVPERF)
        add_rpath(${CUPTI_nvperf_host_LIBRARY} ${CUPTI_nvperf_target_LIBRARY})
    endif()

    if(CUPTI_pcsampling_FOUND)
        timemory_target_compile_definitions(timemory-cupti INTERFACE
            TIMEMORY_USE_CUPTI_PCSAMPLING)
    endif()

    if(CUPTI_pcsampling_util_FOUND)
        timemory_target_compile_definitions(timemory-cupti INTERFACE
            TIMEMORY_USE_CUPTI_PCSAMPLING_UTIL)
        add_rpath(${CUPTI_pcsampling_util_LIBRARY})
    endif()

    add_rpath(${CUPTI_cupti_LIBRARY})

else()
    set(TIMEMORY_USE_CUPTI OFF)
    inform_empty_interface(timemory-cupti "CUPTI")
    inform_empty_interface(timemory-gpu-roofline "GPU roofline (CUPTI)")
endif()


#----------------------------------------------------------------------------------------#
#
#                               NCCL
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_NCCL)
    find_package(NCCL ${TIMEMORY_FIND_QUIETLY} ${TIMEMORY_FIND_REQUIREMENT})
endif()

if(NCCL_FOUND AND TIMEMORY_USE_CUDA)
    add_rpath(${NCCL_LIBRARIES})
    target_link_libraries(timemory-nccl INTERFACE ${NCCL_LIBRARIES})
    target_include_directories(timemory-nccl SYSTEM INTERFACE ${NCCL_INCLUDE_DIRS})
    timemory_target_compile_definitions(timemory-nccl INTERFACE TIMEMORY_USE_NCCL)
else()
    set(TIMEMORY_USE_NCCL OFF)
    inform_empty_interface(timemory-nccl "NCCL")
endif()


#----------------------------------------------------------------------------------------#
#
#                               NVML
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_NVML)
    find_package(NVML ${TIMEMORY_FIND_QUIETLY} ${TIMEMORY_FIND_REQUIREMENT})
endif()

if(NVML_FOUND)
    add_rpath(${NVML_LIBRARIES})
    target_link_libraries(timemory-nvml INTERFACE ${NVML_LIBRARIES})
    target_include_directories(timemory-nvml SYSTEM INTERFACE ${NVML_INCLUDE_DIRS})
    timemory_target_compile_definitions(timemory-nvml INTERFACE TIMEMORY_USE_NVML)
else()
    set(TIMEMORY_USE_NVML OFF)
    inform_empty_interface(timemory-nvml "NVML")
endif()

#----------------------------------------------------------------------------------------#
#
#                               LIBUNWIND
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_LIBUNWIND)
    find_package(libunwind ${TIMEMORY_FIND_QUIETLY} ${TIMEMORY_FIND_REQUIREMENT})
endif()

if(libunwind_FOUND)
    target_link_libraries(timemory-libunwind INTERFACE ${libunwind_LIBRARIES})
    target_include_directories(timemory-libunwind SYSTEM INTERFACE ${libunwind_INCLUDE_DIRS})
    timemory_target_compile_definitions(timemory-libunwind INTERFACE
        TIMEMORY_USE_LIBUNWIND UNW_LOCAL_ONLY)
else()
    set(TIMEMORY_USE_LIBUNWIND OFF)
    inform_empty_interface(timemory-libunwind "libunwind")
endif()


#----------------------------------------------------------------------------------------#
#
#                               Google PerfTools
#
#----------------------------------------------------------------------------------------#

set(gperftools_PREFER_SHARED ON CACHE BOOL "Prefer goerftools shared libraries")
set(_GPERF_COMPONENTS ${TIMEMORY_gperftools_COMPONENTS})
if(_GPERF_COMPONENTS)
    list(REMOVE_DUPLICATES _GPERF_COMPONENTS)
endif()

if(NOT TIMEMORY_FORCE_GPERFTOOLS_PYTHON)
    if(TIMEMORY_USE_PYTHON)
        set(_GPERF_COMPONENTS )
        set(TIMEMORY_gperftools_COMPONENTS )
    endif()

    if(TIMEMORY_BUILD_PYTHON)
        set(_GPERF_COMPONENTS )
        set(TIMEMORY_gperftools_COMPONENTS )
    endif()
endif()

if(TIMEMORY_USE_GPERFTOOLS)
    #
    # general set of compiler flags when using gperftools
    #
    target_link_libraries(timemory-gperftools INTERFACE timemory-compile-debuginfo)

    # NOTE:
    #   When compiling with programs with gcc, that you plan to link
    #   with libtcmalloc, it's safest to pass in the flags
    #
    #    -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free
    #
    #   when compiling.  gcc makes some optimizations assuming it is using its
    #   own, built-in malloc; that assumption obviously isn't true with
    #   tcmalloc.  In practice, we haven't seen any problems with this, but
    #   the expected risk is highest for users who register their own malloc
    #   hooks with tcmalloc (using gperftools/malloc_hook.h).  The risk is
    #   lowest for folks who use tcmalloc_minimal (or, of course, who pass in
    #   the above flags :-) ).
    #
    # Reference: https://github.com/gperftools/gperftools and "TCMALLOC" section
    #
    if("tcmalloc" IN_LIST _GPERF_COMPONENTS)
        add_target_flag_if_avail(timemory-gperftools
            "-fno-builtin-malloc" "-fno-builtin-calloc"
            "-fno-builtin-realloc" "-fno-builtin-free")
    endif()

    #
    # NOTE:
    #   if tcmalloc is dynamically linked to Python, the lazy loading of tcmalloc
    #   changes malloc/free after Python has used libc malloc, which commonly
    #   corrupts the deletion of the Python interpreter at the end of the application
    #
    if(TIMEMORY_BUILD_PYTHON)
        set(gperftools_PREFER_STATIC OFF)
    endif()

    set(_DEFINITIONS)
    foreach(_COMP ${_GPERF_COMPONENTS})
        if("tcmalloc" MATCHES "${_COMP}")
            list(APPEND _DEFINITIONS TIMEMORY_USE_GPERFTOOLS_TCMALLOC)
        endif()
        if("profiler")
            list(APPEND _DEFINITIONS TIMEMORY_USE_GPERFTOOLS_PROFILER)
        endif()
    endforeach()

    if(_DEFINITIONS)
        list(REMOVE_DUPLICATES _DEFINITIONS)
    endif()
    find_package_interface(
        NAME                    gperftools
        INTERFACE               timemory-gperftools
        INCLUDE_DIRS            ${gperftools_INCLUDE_DIRS}
        COMPILE_DEFINITIONS     ${_DEFINITIONS}
        DESCRIPTION             "gperftools with user defined components"
        FIND_ARGS               COMPONENTS ${_GPERF_COMPONENTS})


    target_include_directories(timemory-gperftools SYSTEM INTERFACE ${gperftools_INCLUDE_DIRS})

    add_rpath(${gperftools_LIBRARIES} ${gperftools_ROOT_DIR}/lib ${gperftools_ROOT_DIR}/lib64)

endif()


#----------------------------------------------------------------------------------------#
#
#                               Caliper
#
#----------------------------------------------------------------------------------------#
if(NOT TIMEMORY_USE_CALIPER)
    # override locally to suppress building
    set(TIMEMORY_BUILD_CALIPER OFF)
endif()

if(TIMEMORY_BUILD_CALIPER)
    set(caliper_FOUND ON)
    checkout_git_submodule(RECURSIVE
        RELATIVE_PATH external/caliper
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/Caliper.git
        REPO_BRANCH master)
    include(CaliperDepends)
    set(_ORIG_CEXT ${CMAKE_C_EXTENSIONS})
    set(_ORIG_TESTING ${BUILD_TESTING})
    set(CMAKE_C_EXTENSIONS ON)
    set(BUILD_TESTING OFF)
    set(BUILD_TESTING OFF CACHE BOOL "")
    add_subdirectory(${PROJECT_SOURCE_DIR}/external/caliper)
    set(BUILD_TESTING ${_ORIG_TESTING})
    set(CMAKE_C_EXTENSIONS ${_ORIG_CEXT})
    set(caliper_DIR ${CMAKE_INSTALL_PREFIX}/share/cmake/caliper)
    foreach(_TARGET caliper caliper-serial caliper-tools-util caliper-mpi)
        if(TARGET ${_TARGET})
            list(APPEND TIMEMORY_PACKAGE_LIBRARIES ${_TARGET})
            if(TIMEMORY_INSTALL_CONFIG)
                install(
                    TARGETS     ${_TARGET}
                    DESTINATION ${CMAKE_INSTALL_LIBDIR}
                    EXPORT      ${PROJECT_NAME}-library-depends
                    OPTIONAL)
            endif()
        endif()
    endforeach()
else()
    if(TIMEMORY_USE_CALIPER)
        find_package(caliper ${TIMEMORY_FIND_QUIETLY} ${TIMEMORY_FIND_REQUIREMENT})
    endif()
endif()

if(caliper_FOUND)
    timemory_target_compile_definitions(timemory-caliper INTERFACE TIMEMORY_USE_CALIPER)
    if(TIMEMORY_BUILD_CALIPER)
        target_include_directories(timemory-caliper SYSTEM INTERFACE
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/caliper/include>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/caliper/include>)
        target_link_libraries(timemory-caliper INTERFACE caliper)
        if(WITH_CUPTI)
            target_link_libraries(timemory-caliper INTERFACE timemory-cupti)
        endif()
        if(WITH_PAPI)
            target_link_libraries(timemory-caliper INTERFACE timemory-papi)
        endif()
        set_target_properties(timemory-caliper PROPERTIES
            INTERFACE_LINK_DIRECTORIES $<INSTALL_INTERFACE:${CMAKE_INSTALL_LIBDIR}>)
    else()
        target_include_directories(timemory-caliper SYSTEM INTERFACE ${caliper_INCLUDE_DIR})
        target_link_libraries(timemory-caliper INTERFACE caliper)
    endif()
else()
    set(TIMEMORY_USE_CALIPER OFF)
    inform_empty_interface(timemory-caliper "caliper")
endif()


#----------------------------------------------------------------------------------------#
#
#                               GOTCHA
#
#----------------------------------------------------------------------------------------#
if(UNIX AND NOT APPLE)
    set(GOTCHA_BUILD_EXAMPLES OFF CACHE BOOL "Build GOTCHA examples")
    if(TIMEMORY_BUILD_GOTCHA AND TIMEMORY_USE_GOTCHA)
        set(gotcha_FOUND ON)
        checkout_git_submodule(RECURSIVE
            RELATIVE_PATH external/gotcha
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            REPO_URL https://github.com/jrmadsen/GOTCHA.git
            REPO_BRANCH timemory)
        add_subdirectory(${PROJECT_SOURCE_DIR}/external/gotcha)
        foreach(_TARGET gotcha gotcha-include Gotcha)
            if(TARGET ${_TARGET})
                list(APPEND TIMEMORY_PACKAGE_LIBRARIES ${_TARGET})
                if(TIMEMORY_INSTALL_CONFIG)
                    install(
                        TARGETS     ${_TARGET}
                        DESTINATION ${CMAKE_INSTALL_LIBDIR}
                        EXPORT      ${PROJECT_NAME}-library-depends
                        OPTIONAL)
                endif()
            endif()
        endforeach()
        set(gotcha_DIR ${CMAKE_INSTALL_PREFIX}/share/cmake/gotcha)
    elseif(TIMEMORY_USE_GOTCHA)
        find_package(gotcha ${TIMEMORY_FIND_QUIETLY} ${TIMEMORY_FIND_REQUIREMENT})
        set(TIMEMORY_BUILD_GOTCHA OFF)
    else()
        set(gotcha_FOUND OFF)
        set(TIMEMORY_BUILD_GOTCHA OFF)
    endif()
else()
    set(gotcha_FOUND OFF)
endif()

if(gotcha_FOUND)
    timemory_target_compile_definitions(timemory-gotcha INTERFACE TIMEMORY_USE_GOTCHA)
    foreach(_LIB gotcha gotcha-include Gotcha Gotcha::gotcha Gotcha::Gotcha)
        if(TARGET ${_LIB})
            target_link_libraries(timemory-gotcha INTERFACE ${_LIB})
        endif()
    endforeach()
    if(NOT (CMAKE_CXX_COMPILER_IS_CLANG AND APPLE))
        add_target_flag_if_avail(timemory-gotcha "-rdynamic")
    endif()
    if(TIMEMORY_BUILD_GOTCHA)
        set_target_properties(timemory-gotcha PROPERTIES
            INTERFACE_LINK_DIRECTORIES $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)
        set_target_properties(timemory-gotcha PROPERTIES
            INTERFACE_LINK_DIRECTORIES $<INSTALL_INTERFACE:${CMAKE_INSTALL_LIBDIR}>)
    endif()
else()
    set(TIMEMORY_USE_GOTCHA OFF)
    inform_empty_interface(timemory-gotcha "GOTCHA")
endif()


#----------------------------------------------------------------------------------------#
#
#                               LIKWID
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_LIKWID)
    find_package(LIKWID ${TIMEMORY_FIND_REQUIREMENT})
endif()

if(LIKWID_FOUND)
    target_link_libraries(timemory-likwid INTERFACE ${LIKWID_LIBRARIES})
    target_include_directories(timemory-likwid SYSTEM INTERFACE ${LIKWID_INCLUDE_DIRS})
    timemory_target_compile_definitions(timemory-likwid INTERFACE TIMEMORY_USE_LIKWID)
    if(TIMEMORY_USE_LIKWID AND NOT TIMEMORY_USE_LIKWID_PERFMON AND NOT TIMEMORY_USE_LIKWID_NVMON)
        set(TIMEMORY_USE_LIKWID_PERFMON ${TIMEMORY_USE_LIKWID})
    endif()
    if(TIMEMORY_USE_LIKWID_PERFMON)
        timemory_target_compile_definitions(timemory-likwid INTERFACE TIMEMORY_USE_LIKWID_PERFMON)
        target_compile_definitions(timemory-likwid INTERFACE LIKWID_PERFMON)
    endif()
    if(TIMEMORY_USE_LIKWID_NVMON)
        timemory_target_compile_definitions(timemory-likwid INTERFACE TIMEMORY_USE_LIKWID_NVMON)
        target_compile_definitions(timemory-likwid INTERFACE LIKWID_NVMON)
    endif()
    add_rpath(${LIKWID_LIBRARIES})
else()
    set(TIMEMORY_USE_LIKWID OFF)
    set(TIMEMORY_USE_LIKWID_PERFMON OFF)
    set(TIMEMORY_USE_LIKWID_NVMON OFF)
    inform_empty_interface(timemory-likwid "LIKWID")
endif()


#----------------------------------------------------------------------------------------#
#
#                               OpenMP
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_OMPT)
    if(TIMEMORY_BUILD_OMPT)
        set(OPENMP_STANDALONE_BUILD ON CACHE BOOL "Needed by ompt")
        set(OPENMP_ENABLE_TESTING OFF CACHE BOOL "Do not test")
        if(TIMEMORY_USE_CUDA)
            set(OPENMP_ENABLE_LIBOMPTARGET ON CACHE BOOL "OpenMP target tooling")
        else()
            set(OPENMP_ENABLE_LIBOMPTARGET OFF CACHE BOOL "OpenMP target tooling")
        endif()
        checkout_git_submodule(RECURSIVE
            RELATIVE_PATH external/llvm-ompt
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            REPO_URL https://github.com/NERSC/LLVM-openmp.git
            REPO_BRANCH timemory)
        add_subdirectory(${PROJECT_SOURCE_DIR}/external/llvm-ompt)
        target_include_directories(timemory-ompt SYSTEM INTERFACE
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/llvm-ompt/runtime/src>)
        foreach(_TARGET omp omptarget)
            if(TARGET ${_TARGET})
                list(APPEND TIMEMORY_PACKAGE_LIBRARIES ${_TARGET})
                if(TIMEMORY_INSTALL_CONFIG)
                    install(TARGETS ${_TARGET}
                        DESTINATION ${CMAKE_INSTALL_LIBDIR}
                        EXPORT ${PROJECT_NAME}-library-depends
                        OPTIONAL)
                endif()
            endif()
        endforeach()
    endif()
else()
    set(TIMEMORY_BUILD_OMPT OFF)
endif()

if(TIMEMORY_USE_OMPT AND TIMEMORY_BUILD_OMPT)
    foreach(_TARG omp ompimp omptarget)
        if(TARGET ${_TARG})
            target_link_libraries(timemory-ompt INTERFACE ${_TARG})
        endif()
    endforeach()
    timemory_target_compile_definitions(timemory-ompt INTERFACE TIMEMORY_USE_OMPT)
elseif(TIMEMORY_USE_OMPT)
    timemory_target_compile_definitions(timemory-ompt INTERFACE TIMEMORY_USE_OMPT)
else()
    set(TIMEMORY_BUILD_OMPT OFF)
    set(TIMEMORY_USE_OMPT OFF)
    inform_empty_interface(timemory-ompt "OpenMP")
endif()


#----------------------------------------------------------------------------------------#
#
#                               VTune
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_VTUNE)
    find_package(ittnotify ${TIMEMORY_FIND_REQUIREMENT})
endif()

if(ittnotify_FOUND)
    target_link_libraries(timemory-vtune INTERFACE ${ITTNOTIFY_LIBRARIES})
    target_include_directories(timemory-vtune SYSTEM INTERFACE ${ITTNOTIFY_INCLUDE_DIRS})
    timemory_target_compile_definitions(timemory-vtune INTERFACE TIMEMORY_USE_VTUNE)
    add_rpath(${ITTNOTIFY_LIBRARIES})
else()
    set(TIMEMORY_USE_VTUNE OFF)
    inform_empty_interface(timemory-vtune "VTune (ittnotify)")
endif()


#----------------------------------------------------------------------------------------#
#
#                               TAU
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_TAU)
    find_package(TAU ${TIMEMORY_FIND_QUIETLY} ${TIMEMORY_FIND_REQUIREMENT})
endif()

if(TAU_FOUND)
    target_link_libraries(timemory-tau INTERFACE ${TAU_LIBRARIES})
    target_include_directories(timemory-tau SYSTEM INTERFACE ${TAU_INCLUDE_DIRS})
    timemory_target_compile_definitions(timemory-tau INTERFACE TIMEMORY_USE_TAU)
    add_rpath(${TAU_LIBRARIES})
else()
    set(TIMEMORY_USE_TAU OFF)
    inform_empty_interface(timemory-tau "TAU")
endif()

#----------------------------------------------------------------------------------------#
#
#                               Roofline
#
#----------------------------------------------------------------------------------------#

target_link_libraries(timemory-roofline-options INTERFACE
    timemory-compile-extra
    timemory-arch)

target_link_libraries(timemory-cpu-roofline INTERFACE
    timemory-roofline-options
    timemory-papi)

target_link_libraries(timemory-gpu-roofline INTERFACE
    timemory-roofline-options
    timemory-cupti
    timemory-cuda
    timemory-cudart-device)

generate_composite_interface(timemory-roofline
    timemory-cpu-roofline
    timemory-gpu-roofline)


#----------------------------------------------------------------------------------------#
#
#                               Dyninst
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_DYNINST)
    find_package(Dyninst ${TIMEMORY_FIND_QUIETLY} ${TIMEMORY_FIND_REQUIREMENT})
    set(_BOOST_COMPONENTS atomic system thread date_time)
    set(TIMEMORY_BOOST_COMPONENTS "${_BOOST_COMPONENTS}" CACHE STRING
        "Boost components used by Dyninst in timemory")
    if(Dyninst_FOUND)
        set(Boost_NO_BOOST_CMAKE ON)
        find_package(Boost QUIET ${TIMEMORY_FIND_REQUIREMENT}
            COMPONENTS ${TIMEMORY_BOOST_COMPONENTS})
    endif()
endif()

if(Dyninst_FOUND AND Boost_FOUND)

    set(_Dyninst)
    # some installs of dyninst don't set this properly
    if(EXISTS "${DYNINST_INCLUDE_DIR}" AND NOT DYNINST_HEADER_DIR)
        get_filename_component(DYNINST_HEADER_DIR "${DYNINST_INCLUDE_DIR}" REALPATH CACHE)
    else()
        find_path(DYNINST_HEADER_DIR
            NAMES BPatch.h dyninstAPI_RT.h
            HINTS ${Dyninst_ROOT_DIR} ${Dyninst_DIR} ${Dyninst_DIR}/../../..
            PATHS ${Dyninst_ROOT_DIR} ${Dyninst_DIR} ${Dyninst_DIR}/../../..
            PATH_SUFFIXES include)
    endif()

    # useful for defining the location of the runtime API
    find_library(DYNINST_API_RT dyninstAPI_RT
        HINTS ${Dyninst_ROOT_DIR} ${Dyninst_DIR}
        PATHS ${Dyninst_ROOT_DIR} ${Dyninst_DIR}
        PATH_SUFFIXES lib)

    find_path(TBB_INCLUDE_DIR
        NAMES tbb/tbb.h
    PATH_SUFFIXES include)

    if(TBB_INCLUDE_DIR)
        set(TBB_INCLUDE_DIRS ${TBB_INCLUDE_DIR})
    endif()

    if(DYNINST_API_RT)
        target_compile_definitions(timemory-dyninst INTERFACE
            DYNINST_API_RT="${DYNINST_API_RT}")
    endif()

    if(Boost_DIR)
        get_filename_component(Boost_RPATH_DIR "${Boost_DIR}" DIRECTORY)
        get_filename_component(Boost_RPATH_DIR "${Boost_RPATH_DIR}" DIRECTORY)
        if(EXISTS "${Boost_RPATH_DIR}" AND IS_DIRECTORY "${Boost_RPATH_DIR}")
            set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${Boost_RPATH_DIR}")
        endif()
    endif()

    add_rpath(${DYNINST_LIBRARIES} ${Boost_LIBRARIES})
    target_link_libraries(timemory-dyninst INTERFACE
        ${DYNINST_LIBRARIES} ${Boost_LIBRARIES})
    foreach(_TARG Dyninst::dyninst Boost::headers Boost::atomic
            Boost::system Boost::thread Boost::date_time)
        if(TARGET ${_TARG})
            target_link_libraries(timemory-dyninst INTERFACE ${_TARG})
        endif()
    endforeach()
    target_include_directories(timemory-dyninst SYSTEM INTERFACE
        ${TBB_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${DYNINST_HEADER_DIR})
    timemory_target_compile_definitions(timemory-dyninst INTERFACE TIMEMORY_USE_DYNINST)
else()
    set(TIMEMORY_USE_DYNINST OFF)
    inform_empty_interface(timemory-dyninst "dyninst")
endif()

if(TIMEMORY_USE_DYNINST)
    set(TIMEMORY_BUILD_DYNINST_TOOLS ${TIMEMORY_USE_DYNINST})
endif()

if(DYNINST_API_RT)
    add_cmake_defines(DYNINST_API_RT VALUE QUOTE DEFAULT)
else()
    add_cmake_defines(DYNINST_API_RT VALUE QUOTE)
endif()


#----------------------------------------------------------------------------------------#
#
#                               AllineaMAP
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_ALLINEA_MAP)
    find_package(AllineaMAP ${TIMEMORY_FIND_REQUIREMENT})
endif()

if(AllineaMAP_FOUND)
    add_rpath(${AllineaMAP_LIBRARIES})
    target_link_libraries(timemory-allinea-map INTERFACE ${AllineaMAP_LIBRARIES})
    target_include_directories(timemory-allinea-map SYSTEM INTERFACE ${AllineaMAP_INCLUDE_DIRS})
    timemory_target_compile_definitions(timemory-allinea-map INTERFACE TIMEMORY_USE_ALLINEA_MAP)
else()
    set(TIMEMORY_USE_ALLINEA_MAP OFF)
    inform_empty_interface(timemory-allinea-map "Allinea MAP")
endif()


#----------------------------------------------------------------------------------------#
#
#                               CrayPAT
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_CRAYPAT)
    find_package(CrayPAT ${TIMEMORY_FIND_REQUIREMENT} COMPONENTS ${CrayPAT_COMPONENTS})
endif()

if(CrayPAT_FOUND)
    add_rpath(${CrayPAT_LIBRARIES})
    target_link_libraries(timemory-craypat INTERFACE ${CrayPAT_LIBRARIES})
    target_link_directories(timemory-craypat INTERFACE ${CrayPAT_LIBRARY_DIRS})
    target_include_directories(timemory-craypat SYSTEM INTERFACE ${CrayPAT_INCLUDE_DIRS})
    timemory_target_compile_definitions(timemory-craypat INTERFACE TIMEMORY_USE_CRAYPAT)
    target_compile_definitions(timemory-craypat INTERFACE CRAYPAT)
    add_target_flag_if_avail(timemory-craypat "-g" "-debug pubnames"
        "-Qlocation,ld,${CrayPAT_LIBRARY_DIR}" "-fno-omit-frame-pointer"
        "-fno-optimize-sibling-calls")
else()
    set(TIMEMORY_USE_CRAYPAT OFF)
    inform_empty_interface(timemory-craypat "CrayPAT")
endif()

#----------------------------------------------------------------------------------------#
#
#                       PTL (Parallel Tasking Library)
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_PTL OR TIMEMORY_BUILD_TESTING)
    checkout_git_submodule(RECURSIVE
        RELATIVE_PATH external/ptl
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/PTL.git
        REPO_BRANCH master)

    timemory_message(STATUS "Adding external/ptl")
    option(PTL_USE_TBB "Enable TBB backend support in PTL" OFF)
    set(PTL_DIR ${PROJECT_BINARY_DIR}/external/ptl CACHE PATH "Path to PTL build" FORCE)
    if(PTL_BUILD_EXAMPLES)
        set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PROJECT_BINARY_DIR}/external/ptl)
    endif()
    add_subdirectory(${PROJECT_SOURCE_DIR}/external/ptl)
endif()

#----------------------------------------------------------------------------------------#
#
#                       Include customizable UserPackages file
#
#----------------------------------------------------------------------------------------#

include(UserPackages)

add_feature(CMAKE_INSTALL_RPATH "Installation RPATH")

if(TIMEMORY_INSTALL_CONFIG)
    install(FILES ${PROJECT_SOURCE_DIR}/cmake/Modules/LocalFindUtilities.cmake
        DESTINATION ${CMAKE_INSTALL_CONFIGDIR}/Modules
        OPTIONAL)
endif()
