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
add_interface_library(timemory-cereal
    "Provides include flags for serialization library")
add_interface_library(timemory-cereal-xml
    "Enables XML serialization output")
add_interface_library(timemory-extern
    "Enables pre-processor directive to ensure all extern templates are used")
add_interface_library(timemory-statistics
    "Enables statistics for all components which define TIMEMORY_STATISTICS_TYPE(...)")

set(TIMEMORY_REQUIRED_INTERFACES
    timemory-headers
    timemory-cereal)

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
add_interface_library(timemory-nvtx
    "Enables CUDA NVTX support")
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

add_interface_library(timemory-coverage
    "Enables code-coverage flags")
add_interface_library(timemory-gperftools-compile-options
    "Enables compiler flags for resolving function calls in gperftools output")
add_interface_library(timemory-all-gperftools
    "Enables all gperftools components (cpu and heap profilers)")
add_interface_library(timemory-gperftools
    "Enables user-selected gperftools component (${_GPERF_COMPONENTS})")
add_interface_library(timemory-gperftools-cpu
    "Enables gperftools cpu profiler support")
add_interface_library(timemory-gperftools-heap
    "Enables gperftools heap profiler support")
add_interface_library(timemory-gperftools-static
    "Enables gperftools support via static linking")
add_interface_library(timemory-tcmalloc-minimal
    "Enables gperftools tcmalloc_minimal library")

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

target_link_libraries(timemory-mpip-library INTERFACE timemory-mpi timemory-gotcha)

set(_DMP_LIBRARIES)

if(TIMEMORY_USE_MPI)
    list(APPEND _DMP_LIBRARIES timemory-mpi)
    target_link_libraries(timemory-dmp INTERFACE timemory-mpi)
endif()

if(TIMEMORY_USE_UPCXX)
    list(APPEND _DMP_LIBRARIES timemory-upcxx)
    target_link_libraries(timemory-dmp INTERFACE timemory-upcxx)
endif()

set(TIMEMORY_EXTENSION_INTERFACES
    timemory-mpi
    timemory-upcxx
    timemory-threading
    #
    timemory-statistics
    #
    timemory-papi
    #
    timemory-cuda
    # timemory-cudart
    timemory-nvtx
    timemory-cupti
    timemory-cudart-device
    #
    timemory-gperftools-cpu
    timemory-gperftools-heap
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
    timemory-allinea-map)

set(TIMEMORY_EXTERNAL_SHARED_INTERFACES
    timemory-threading
    timemory-statistics
    timemory-papi
    timemory-cuda
    timemory-cudart
    timemory-nvtx
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
    ${_DMP_LIBRARIES})

set(TIMEMORY_EXTERNAL_STATIC_INTERFACES
    timemory-threading
    timemory-statistics
    timemory-papi-static
    timemory-cuda
    timemory-cudart-static
    timemory-nvtx
    timemory-cupti
    timemory-cudart-device
    timemory-caliper
    timemory-vtune
    timemory-tau
    timemory-ompt
    timemory-craypat
    timemory-allinea-map
    timemory-plotting
    ${_DMP_LIBRARIES})

set(_GPERF_IN_LIBRARY OFF)
# if not python or force requested
if(NOT TIMEMORY_USE_PYTHON OR TIMEMORY_FORCE_GPERFTOOLS_PYTHON)
    list(APPEND TIMEMORY_EXTERNAL_SHARED_INTERFACES timemory-gperftools)
    list(APPEND TIMEMORY_EXTERNAL_STATIC_INTERFACES timemory-gperftools-static)
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
    target_link_libraries(timemory-analysis-tools INTERFACE timemory-gperftools-cpu)
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
            message(STATUS  "[interface] '${_TARGET}' depends on '${_DEPENDS}' which is empty...")
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
            target_compile_definitions(${PACKAGE_INTERFACE} INTERFACE ${_DEF})
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

# this target is always linked whenever timemory is used via cmake
target_compile_definitions(timemory-headers INTERFACE TIMEMORY_CMAKE)

target_include_directories(timemory-headers INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>)

target_include_directories(timemory-headers SYSTEM INTERFACE
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(TIMEMORY_LINK_RT)
    target_link_libraries(timemory-headers INTERFACE rt)
endif()
# include threading because of rooflines
target_link_libraries(timemory-headers INTERFACE timemory-threading)

#----------------------------------------------------------------------------------------#
#
#                        timemory extern initializaiton
#
#----------------------------------------------------------------------------------------#

if(NOT WIN32)
    target_compile_definitions(timemory-extern INTERFACE TIMEMORY_USE_EXTERN)
endif()

#----------------------------------------------------------------------------------------#
#
#                        timemory statistics
#
#----------------------------------------------------------------------------------------#

target_compile_definitions(timemory-statistics INTERFACE TIMEMORY_USE_STATISTICS)
if(TIMEMORY_USE_STATISTICS)
    target_link_libraries(timemory-headers INTERFACE timemory-statistics)
endif()

#----------------------------------------------------------------------------------------#
#
#                           Cereal (serialization library)
#
#----------------------------------------------------------------------------------------#

#set(DEV_WARNINGS ${CMAKE_SUPPRESS_DEVELOPER_WARNINGS})

checkout_git_submodule(RECURSIVE
    RELATIVE_PATH external/cereal
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    REPO_URL https://github.com/jrmadsen/cereal.git
    REPO_BRANCH timemory)

# add cereal
add_subdirectory(${PROJECT_SOURCE_DIR}/external/cereal)

target_include_directories(timemory-cereal SYSTEM INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/cereal/include>)

# timemory-headers always provides timemory-cereal
target_link_libraries(timemory-headers INTERFACE timemory-cereal)
target_link_libraries(timemory-cereal-xml INTERFACE timemory-cereal)
target_compile_definitions(timemory-cereal-xml INTERFACE TIMEMORY_USE_XML_ARCHIVE)

#----------------------------------------------------------------------------------------#
#
#                           Google Test
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_BUILD_GOOGLE_TEST)
    checkout_git_submodule(RECURSIVE
        RELATIVE_PATH external/google-test
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/google/googletest.git
        REPO_BRANCH master)

    # add google-test
    set(INSTALL_GTEST OFF CACHE BOOL "Install gtest")
    set(BUILD_GMOCK ON CACHE BOOL "Build gmock")
    if(APPLE)
        set(CMAKE_MACOSX_RPATH ON CACHE BOOL "Enable MACOS_RPATH on targets to suppress warnings")
        mark_as_advanced(CMAKE_MACOSX_RPATH)
    endif()
    add_subdirectory(${PROJECT_SOURCE_DIR}/external/google-test)
    target_link_libraries(timemory-google-test INTERFACE gtest gmock gtest_main)
    target_include_directories(timemory-google-test SYSTEM INTERFACE
        ${PROJECT_SOURCE_DIR}/google-test/googletest/include
        ${PROJECT_SOURCE_DIR}/google-test/googlemock/include)
endif()


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

if(TIMEMORY_USE_MPI)
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

    find_package(MPI ${TIMEMORY_FIND_REQUIREMENT})
else()
    set(MPI_FOUND OFF)
endif()

# interface to kill MPI init in headers
target_compile_definitions(timemory-no-mpi-init INTERFACE TIMEMORY_MPI_INIT=0)

if(MPI_FOUND)

    foreach(_LANG C CXX)
        # include directories
        target_include_directories(timemory-mpi SYSTEM INTERFACE ${MPI_${_LANG}_INCLUDE_PATH})

        # link targets
        set(_TYPE )
        if(MPI_${_LANG}_LIBRARIES)
            target_link_libraries(timemory-mpi INTERFACE ${MPI_${_LANG}_LIBRARIES})
	    # add_rpath(${MPI_${_LANG}_LIBRARIES})
        endif()

        # compile flags
        to_list(_FLAGS "${MPI_${_LANG}_COMPILE_FLAGS}")
        foreach(_FLAG ${_FLAGS})
            if("${_LANG}" STREQUAL "CXX")
                add_cxx_flag_if_avail("${_FLAG}" timemory-mpi)
            else()
                add_c_flag_if_avail("${_FLAG}" timemory-mpi)
            endif()
        endforeach()
        unset(_FLAGS)

        option(TIMEMORY_USE_MPI_LINK_FLAGS "Use MPI link flags" OFF)
        mark_as_advanced(TIMEMORY_USE_MPI_LINK_FLAGS)
        # compile flags
        if(TIMEMORY_USE_MPI_LINK_FLAGS)
            to_list(_FLAGS "${MPI_${_LANG}_LINK_FLAGS}")
            foreach(_FLAG ${_FLAGS})
                if(EXISTS "${_FLAG}" AND IS_DIRECTORY "${_FLAG}")
                    continue()
                endif()
                if(NOT CMAKE_VERSION VERSION_LESS 3.13)
                    target_link_options(timemory-mpi INTERFACE
                        $<$<COMPILE_LANGUAGE:${_LANG}>:${_FLAG}>)
                else()
                    set_target_properties(timemory-mpi PROPERTIES
                        INTERFACE_LINK_OPTIONS $<$<COMPILE_LANGUAGE:${_LANG}>:${_FLAG}>)
                endif()
            endforeach()
        endif()
        unset(_FLAGS)

    endforeach()

    if(MPI_EXTRA_LIBRARY)
        target_link_libraries(timemory-mpi INTERFACE ${MPI_EXTRA_LIBRARY})
    endif()

    if(MPI_INCLUDE_PATH)
        target_include_directories(timemory-mpi SYSTEM INTERFACE ${MPI_INCLUDE_PATH})
    endif()

    target_compile_definitions(timemory-mpi INTERFACE TIMEMORY_USE_MPI)

    # used by python
    if(NOT MPIEXEC_EXECUTABLE AND MPIEXEC)
        set(MPIEXEC_EXECUTABLE ${MPIEXEC} CACHE FILEPATH "MPI executable")
    endif()

    # used by python
    if(NOT MPIEXEC_EXECUTABLE AND MPI_EXECUTABLE)
        set(MPIEXEC_EXECUTABLE ${MPI_EXECUTABLE} CACHE FILEPATH "MPI executable")
    endif()

    add_option(TIMEMORY_USE_MPI_INIT "Enable MPI_Init and MPI_Init_thread wrappers" OFF
        CMAKE_DEFINE)
    if(NOT TIMEMORY_USE_MPI_INIT)
        target_link_libraries(timemory-mpi INTERFACE timemory-no-mpi-init)
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

if(TIMEMORY_USE_UPCXX)
    find_package(UPCXX ${TIMEMORY_FIND_QUIETLY} ${TIMEMORY_FIND_REQUIREMENT})
endif()

if(UPCXX_FOUND)

    add_rpath(${UPCXX_LIBRARIES})
    target_link_libraries(timemory-upcxx INTERFACE ${UPCXX_LIBRARIES})
    target_compile_options(timemory-upcxx INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${UPCXX_OPTIONS}>)
    target_compile_features(timemory-upcxx INTERFACE cxx_std_${UPCXX_CXX_STANDARD})
    target_include_directories(timemory-upcxx SYSTEM INTERFACE ${UPCXX_INCLUDE_DIRS})
    target_compile_definitions(timemory-upcxx INTERFACE ${UPCXX_DEFINITIONS} TIMEMORY_USE_UPCXX)

    if(NOT CMAKE_VERSION VERSION_LESS 3.13)
        target_link_options(timemory-upcxx INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${UPCXX_LINK_OPTIONS}>)
    else()
        set_target_properties(timemory-upcxx PROPERTIES INTERFACE_LINK_OPTIONS
            $<$<COMPILE_LANGUAGE:CXX>:${UPCXX_LINK_OPTIONS}>)
    endif()

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
#                               PAPI
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_PAPI)
    find_package(PAPI ${TIMEMORY_FIND_QUIETLY} ${TIMEMORY_FIND_REQUIREMENT})
endif()

if(PAPI_FOUND)
    add_rpath(${PAPI_LIBRARY})
    target_link_libraries(timemory-papi INTERFACE papi-shared)
    target_link_libraries(timemory-papi-static INTERFACE papi-static)
    cache_list(APPEND ${PROJECT_NAME_UC}_INTERFACE_LIBRARIES papi-shared papi-static)
    target_compile_definitions(timemory-papi INTERFACE TIMEMORY_USE_PAPI)
    target_compile_definitions(timemory-papi-static INTERFACE TIMEMORY_USE_PAPI)
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

    add_target_flag_if_avail(timemory-coverage "-fprofile-arcs" "-ftest-coverage")
    add_target_flag(timemory-coverage "-O0" "-g" "--coverage")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
        target_link_options(timemory-coverage INTERFACE --coverage)
    else()
        target_link_libraries(timemory-coverage INTERFACE --coverage)
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

if(TIMEMORY_USE_CUDA)

    set(PROJECT_USE_CUDA_OPTION                 TIMEMORY_USE_CUDA)
    set(PROJECT_CUDA_DEFINITION                 TIMEMORY_USE_CUDA)
    set(PROJECT_CUDA_INTERFACE_PREFIX           timemory)
    set(PROJECT_CUDA_DISABLE_HALF2_OPTION       TIMEMORY_DISABLE_CUDA_HALF)
    set(PROJECT_CUDA_DISABLE_HALF2_DEFINITION   TIMEMORY_DISABLE_CUDA_HALF)

    include(CUDAConfig)

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

    target_compile_definitions(timemory-cupti INTERFACE
        TIMEMORY_USE_CUPTI)

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

    add_rpath(${CUPTI_cupti_LIBRARY} ${CUPTI_nvperf_host_LIBRARY}
        ${CUPTI_nvperf_target_LIBRARY})
else()
    set(TIMEMORY_USE_CUPTI OFF)
    inform_empty_interface(timemory-cupti "CUPTI")
    inform_empty_interface(timemory-gpu-roofline "GPU roofline (CUPTI)")
endif()


#----------------------------------------------------------------------------------------#
#
#                               NVTX
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_NVTX)
    find_package(NVTX ${TIMEMORY_FIND_QUIETLY} ${TIMEMORY_FIND_REQUIREMENT})
endif()

if(NVTX_FOUND AND TIMEMORY_USE_CUDA)
    add_rpath(${NVTX_LIBRARIES})
    target_link_libraries(timemory-nvtx INTERFACE ${NVTX_LIBRARIES})
    target_include_directories(timemory-nvtx SYSTEM INTERFACE ${NVTX_INCLUDE_DIRS})
    target_compile_definitions(timemory-nvtx INTERFACE TIMEMORY_USE_NVTX)
else()
    set(TIMEMORY_USE_NVTX OFF)
    inform_empty_interface(timemory-nvtx "NVTX")
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
    if(NOT CMAKE_CXX_COMPILER_IS_CLANG AND APPLE)
        add_target_flag_if_avail(timemory-gperftools-compile-options "-g" "-rdynamic")
    else()
        add_target_flag_if_avail(timemory-gperftools-compile-options "-g")
    endif()

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
    add_target_flag_if_avail(timemory-gperftools-compile-options
        "-fno-builtin-malloc" "-fno-builtin-calloc"
        "-fno-builtin-realloc" "-fno-builtin-free")

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
        LINK_LIBRARIES          timemory-gperftools-compile-options
        DESCRIPTION             "gperftools with user defined components"
        FIND_ARGS               COMPONENTS ${_GPERF_COMPONENTS})

    find_package_interface(
        NAME                    gperftools
        INTERFACE               timemory-all-gperftools
        INCLUDE_DIRS            ${gperftools_INCLUDE_DIRS}
        COMPILE_DEFINITIONS     TIMEMORY_USE_GPERFTOOLS
        LINK_LIBRARIES          timemory-gperftools-compile-options
        DESCRIPTION             "tcmalloc_and_profiler (preference for shared)"
        FIND_ARGS               ${TIMEMORY_FIND_QUIETLY} COMPONENTS tcmalloc_and_profiler)

    find_package_interface(
        NAME                    gperftools
        INTERFACE               timemory-gperftools-cpu
        INCLUDE_DIRS            ${gperftools_INCLUDE_DIRS}
        COMPILE_DEFINITIONS     TIMEMORY_USE_GPERFTOOLS_PROFILER
        LINK_LIBRARIES          timemory-gperftools-compile-options
        DESCRIPTION             "CPU profiler"
        FIND_ARGS               ${TIMEMORY_FIND_QUIETLY} COMPONENTS profiler)

    find_package_interface(
        NAME                    gperftools
        INTERFACE               timemory-gperftools-heap
        INCLUDE_DIRS            ${gperftools_INCLUDE_DIRS}
        COMPILE_DEFINITIONS     TIMEMORY_USE_GPERFTOOLS_TCMALLOC
        LINK_LIBRARIES          timemory-gperftools-compile-options
        DESCRIPTION             "heap profiler and heap checker"
        FIND_ARGS               ${TIMEMORY_FIND_QUIETLY} COMPONENTS tcmalloc)

    find_package_interface(
        NAME                    gperftools
        INTERFACE               timemory-tcmalloc-minimal
        INCLUDE_DIRS            ${gperftools_INCLUDE_DIRS}
        LINK_LIBRARIES          timemory-gperftools-compile-options
        DESCRIPTION             "threading-optimized malloc replacement"
        FIND_ARGS               ${TIMEMORY_FIND_QUIETLY} COMPONENTS tcmalloc_minimal)

    target_include_directories(timemory-gperftools SYSTEM INTERFACE ${gperftools_INCLUDE_DIRS})
    target_include_directories(timemory-gperftools-static SYSTEM INTERFACE ${gperftools_INCLUDE_DIRS})
    
    add_rpath(${gperftools_LIBRARIES} ${gperftools_ROOT_DIR}/lib ${gperftools_ROOT_DIR}/lib64)
    
    if(TIMEMORY_USE_GPERFTOOLS_STATIC)
        # set local overloads
        set(gperftools_PREFER_SHARED OFF)
        set(gperftools_PREFER_STATIC ON)

        find_package_interface(
            NAME                    gperftools
            INTERFACE               timemory-gperftools-static
            LINK_LIBRARIES          timemory-gperftools-compile-options
            DESCRIPTION             "tcmalloc_and_profiler (preference for static)"
            FIND_ARGS               ${TIMEMORY_FIND_QUIETLY} COMPONENTS tcmalloc_and_profiler)

        # remove local overloads
        unset(gperftools_PREFER_SHARED)
        unset(gperftools_PREFER_STATIC)
    else()
        inform_empty_interface(timemory-gperftools-static "gperftools static linking")
    endif()
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
else()
    if(TIMEMORY_USE_CALIPER)
        find_package(caliper ${TIMEMORY_FIND_QUIETLY} ${TIMEMORY_FIND_REQUIREMENT})
    endif()
endif()

if(caliper_FOUND)
    target_compile_definitions(timemory-caliper INTERFACE TIMEMORY_USE_CALIPER)
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
            REPO_BRANCH cmake-updates)
        add_subdirectory(${PROJECT_SOURCE_DIR}/external/gotcha)
        foreach(_LIB gotcha gotcha-include Gotcha)
            if(TARGET ${_LIB})
                list(APPEND TIMEMORY_PACKAGE_LIBRARIES ${_LIB})
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
    target_compile_definitions(timemory-gotcha INTERFACE TIMEMORY_USE_GOTCHA)
    foreach(_LIB gotcha gotcha-include Gotcha Gotcha::gotcha Gotcha::Gotcha)
        if(TARGET ${_LIB})
            target_link_libraries(timemory-gotcha INTERFACE ${_LIB})
        endif()
    endforeach()
    if(NOT CMAKE_CXX_COMPILER_IS_CLANG AND APPLE)
        add_target_flag_if_avail(timemory-gotcha "-rdynamic")
    endif()
    if(TIMEMORY_BUILD_GOTCHA)
        set_target_properties(timemory-gotcha PROPERTIES
            INTERFACE_LINK_DIRECTORIES $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>)
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
    target_compile_definitions(timemory-likwid INTERFACE TIMEMORY_USE_LIKWID)
    add_rpath(${LIKWID_LIBRARIES})
else()
    set(TIMEMORY_USE_LIKWID OFF)
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
        add_subdirectory(external/llvm-ompt)
        target_include_directories(timemory-ompt SYSTEM INTERFACE
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/llvm-ompt/runtime/src>)
    endif()
else()
    set(TIMEMORY_BUILD_OMPT OFF)
endif()

if(TIMEMORY_USE_OMPT AND TIMEMORY_BUILD_OMPT)
    set(OMPT_EXPORT_TARGETS)
    foreach(_TARG omp ompimp)
        if(TARGET ${_TARG})
            target_link_libraries(timemory-ompt INTERFACE ${_TARG})
            list(APPEND OMPT_EXPORT_TARGETS ${_TARG})
        endif()
    endforeach()
    target_compile_definitions(timemory-ompt INTERFACE TIMEMORY_USE_OMPT)
elseif(TIMEMORY_USE_OMPT)
    target_compile_definitions(timemory-ompt INTERFACE TIMEMORY_USE_OMPT)
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
    target_compile_definitions(timemory-vtune INTERFACE TIMEMORY_USE_VTUNE)
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
    target_compile_definitions(timemory-tau INTERFACE TIMEMORY_USE_TAU)
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

if(NOT TIMEMORY_USE_COVERAGE)
    add_target_flag_if_avail(timemory-roofline-options
        "-finline-functions" "-funroll-loops" "-ftree-vectorize"
        "-ftree-loop-optimize" "-ftree-loop-vectorize")
endif()

set(VECTOR_DEFINITION               TIMEMORY_VEC)
set(VECTOR_INTERFACE_TARGET         timemory-roofline-options)
set(ARCH_INTERFACE_TARGET           timemory-roofline-options)

include(ArchConfig)

target_link_libraries(timemory-cpu-roofline INTERFACE
    timemory-roofline-options
    timemory-papi)

target_link_libraries(timemory-gpu-roofline INTERFACE
    timemory-roofline-options
    timemory-cupti
    timemory-cuda
    # timemory-cudart
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
        find_package(Boost ${TIMEMORY_FIND_QUIETLY} ${TIMEMORY_FIND_REQUIREMENT}
            COMPONENTS ${TIMEMORY_BOOST_COMPONENTS})
        # install the revision of FindBoost.cmake which is quiet
        install(FILES ${PROJECT_SOURCE_DIR}/cmake/Modules/FindBoost.cmake
            DESTINATION ${CMAKE_INSTALL_CONFIGDIR}/Modules)
    endif()
endif()

if(Dyninst_FOUND AND Boost_FOUND)
    # some installs of dyninst don't set this properly
    find_path(DYNINST_HEADER_DIR
        NAMES BPatch.h dyninstAPI_RT.h
        HINTS ${Dyninst_DIR}
        PATHS ${Dyninst_DIR}
        PATH_SUFFIXES include)

    # useful for defining the location of the runtime API
    find_library(DYNINST_API_RT dyninstAPI_RT
        HINTS ${Dyninst_DIR}
        PATHS ${Dyninst_DIR}
        PATH_SUFFIXES lib)

    find_path(TBB_INCLUDE_DIR
        NAMES tbb/tbb.h
	PATH_SUFFIXES include)

    if(TBB_INCLUDE_DIR)
        set(TBB_INCLUDE_DIRS ${TBB_INCLUDE_DIR})
    endif()
    
    if(DYNINST_HEADER_DIR)
        target_include_directories(timemory-dyninst SYSTEM INTERFACE ${DYNINST_HEADER_DIR})
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
        ${DYNINST_INCLUDE_DIRS} ${DYNINST_INCLUDE_DIR}
        ${Dyninst_INCLUDE_DIRS} ${Dyninst_INCLUDE_DIR}
        ${TBB_INCLUDE_DIRS}     ${Boost_INCLUDE_DIRS})
    target_compile_definitions(timemory-dyninst INTERFACE TIMEMORY_USE_DYNINST)
else()
    set(TIMEMORY_USE_DYNINST OFF)
    inform_empty_interface(timemory-dyninst "dyninst")
endif()

add_cmake_defines(DYNINST_API_RT VALUE QUOTE)


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
    target_compile_definitions(timemory-allinea-map INTERFACE TIMEMORY_USE_ALLINEA_MAP)
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
    target_compile_definitions(timemory-craypat INTERFACE TIMEMORY_USE_CRAYPAT CRAYPAT)
    add_target_flag_if_avail(timemory-craypat "-g" "-debug pubnames"
        "-Qlocation,ld,${CrayPAT_LIBRARY_DIR}" "-fno-omit-frame-pointer"
        "-fno-optimize-sibling-calls")
else()
    set(TIMEMORY_USE_CRAYPAT OFF)
    inform_empty_interface(timemory-craypat "CrayPAT")
endif()


#----------------------------------------------------------------------------------------#
#
#                       Include customizable UserPackages file
#
#----------------------------------------------------------------------------------------#

include(UserPackages)

add_feature(CMAKE_INSTALL_RPATH "Installation RPATH")
