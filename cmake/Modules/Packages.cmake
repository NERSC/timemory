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

add_interface_library(timemory-headers)
add_interface_library(timemory-cereal)
add_interface_library(timemory-extern-init)

set(TIMEMORY_REQUIRED_INTERFACES
    timemory-headers
    timemory-cereal)

add_interface_library(timemory-mpi)
add_interface_library(timemory-upcxx)
add_interface_library(timemory-threading)

add_interface_library(timemory-papi)
add_interface_library(timemory-papi-static)
add_interface_library(timemory-cuda)
add_interface_library(timemory-cuda-compiler)
add_interface_library(timemory-cupti)
add_interface_library(timemory-cudart)
add_interface_library(timemory-cudart-device)
add_interface_library(timemory-cudart-static)
add_interface_library(timemory-nvtx)
add_interface_library(timemory-caliper)
add_interface_library(timemory-gotcha)
add_interface_library(timemory-likwid)
add_interface_library(timemory-vtune)
add_interface_library(timemory-tau)
add_interface_library(timemory-python)

add_interface_library(timemory-coverage)
add_interface_library(timemory-gperftools)
add_interface_library(timemory-gperftools-cpu)
add_interface_library(timemory-gperftools-heap)

add_interface_library(timemory-roofline)
add_interface_library(timemory-cpu-roofline)
add_interface_library(timemory-gpu-roofline)
add_interface_library(timemory-roofline-options)

set(_DMP_LIBRARIES)

if(TIMEMORY_USE_MPI)
    list(APPEND _DMP_LIBRARIES timemory-mpi)
endif()

if(TIMEMORY_USE_UPCXX)
    list(APPEND _DMP_LIBRARIES timemory-upcxx)
endif()

set(TIMEMORY_EXTENSION_INTERFACES
    timemory-mpi
    timemory-upcxx
    timemory-threading
    timemory-papi
    timemory-cuda
    timemory-cudart
    timemory-nvtx
    timemory-cupti
    timemory-cudart-device
    timemory-coverage
    timemory-gperftools
    timemory-gperftools-cpu
    timemory-gperftools-heap
    timemory-santizier
    timemory-caliper
    timemory-gotcha
    timemory-likwid
    timemory-vtune
    timemory-tau)

set(TIMEMORY_EXTERNAL_SHARED_INTERFACES
    timemory-threading
    timemory-papi
    timemory-cuda
    timemory-cudart
    timemory-nvtx
    timemory-cupti
    timemory-cudart-device
    timemory-gperftools-cpu
    timemory-caliper
    timemory-gotcha
    timemory-likwid
    timemory-vtune
    timemory-tau
    ${_DMP_LIBRARIES})

set(TIMEMORY_EXTERNAL_STATIC_INTERFACES
    timemory-threading
    timemory-papi-static
    timemory-cuda
    timemory-cudart-static
    timemory-nvtx
    timemory-cupti
    timemory-cudart-device
    timemory-gperftools-cpu
    timemory-caliper
    timemory-vtune
    timemory-tau
    ${_DMP_LIBRARIES})

add_interface_library(timemory-extensions)
target_link_libraries(timemory-extensions INTERFACE ${TIMEMORY_EXTENSION_INTERFACES})

add_interface_library(timemory-external-shared)
target_link_libraries(timemory-external-shared INTERFACE ${TIMEMORY_EXTERNAL_SHARED_INTERFACES})

add_interface_library(timemory-external-static)
target_link_libraries(timemory-external-static INTERFACE ${TIMEMORY_EXTERNAL_STATIC_INTERFACES})

add_interface_library(timemory-analysis-tools)

if(TIMEMORY_USE_SANITIZER)
    target_link_libraries(timemory-analysis-tools INTERFACE timemory-sanitizer)
endif()

if(TIMEMORY_USE_GPERF)
    target_link_libraries(timemory-analysis-tools INTERFACE timemory-gperftools)
endif()

if(TIMEMORY_USE_COVERAGE)
    target_link_libraries(timemory-analysis-tools INTERFACE timemory-coverage)
endif()

# not exported
add_library(timemory-google-test INTERFACE)

function(INFORM_EMPTY_INTERFACE _TARGET _PACKAGE)
    if(NOT TARGET ${_TARGET})
        message(AUTHOR_WARNING "A non-existant target was passed to INFORM_EMPTY_INTERFACE: ${_TARGET}")
    endif()
    if(NOT ${_TARGET} IN_LIST TIMEMORY_EMPTY_INTERFACE_LIBRARIES)
        message(STATUS  "[interface] ${_PACKAGE} not found. '${_TARGET}' interface will not provide ${_PACKAGE}...")
        set(TIMEMORY_EMPTY_INTERFACE_LIBRARIES ${TIMEMORY_EMPTY_INTERFACE_LIBRARIES} ${_TARGET} PARENT_SCOPE)
    endif()
    add_disabled_interface(${_TARGET})
endfunction()

function(GENERATE_NON_EMPTY_INTERFACE _TARGET)
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
#                               timemory headers
#
#----------------------------------------------------------------------------------------#

target_include_directories(timemory-headers INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>
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
    target_compile_definitions(timemory-extern-init INTERFACE TIMEMORY_EXTERN_INIT)
    if(TIMEMORY_USE_EXTERN_INIT)
        # target_link_libraries(timemory-headers INTERFACE timemory-extern-init)
    endif()
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
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/cereal/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

# timemory-headers always provides timemory-cereal
target_link_libraries(timemory-headers INTERFACE timemory-cereal)


#----------------------------------------------------------------------------------------#
#
#                           Google Test
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_BUILD_GTEST)
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
find_package(Threads QUIET)

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

    find_package(MPI)
else()
    set(MPI_FOUND OFF)
endif()

if(MPI_FOUND)

    foreach(_LANG C CXX)
        # include directories
        target_include_directories(timemory-mpi SYSTEM INTERFACE ${MPI_${_LANG}_INCLUDE_PATH})

        # link targets
        set(_TYPE )
        if(MPI_${_LANG}_LIBRARIES)
            target_link_libraries(timemory-mpi INTERFACE ${MPI_${_LANG}_LIBRARIES})
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

        # compile flags
        to_list(_FLAGS "${MPI_${_LANG}_LINK_FLAGS}")
        foreach(_FLAG ${_FLAGS})
            if(NOT CMAKE_VERSION VERSION_LESS 3.13)
                target_link_options(timemory-mpi INTERFACE
                    $<$<COMPILE_LANGUAGE:${_LANG}>:${_FLAG}>)
            else()
                set_target_properties(timemory-mpi PROPERTIES
                    INTERFACE_LINK_OPTIONS $<$<COMPILE_LANGUAGE:${_LANG}>:${_FLAG}>)
            endif()
        endforeach()
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
    find_package(UPCXX QUIET)
endif()

if(UPCXX_FOUND)

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
if(TIMEMORY_USE_PYTHON)
    if(NOT TIMEMORY_BUILD_PYTHON)
        find_package(pybind11 REQUIRED)
        if(NOT PYTHON_EXECUTABLE)
            find_package(PythonInterp REQUIRED)
            find_package(PythonLibs REQUIRED)
        endif()
    endif()
elseif(NOT TIMEMORY_USE_PYTHON)
    set(TIMEMORY_BUILD_PYTHON OFF)
endif()

if(TIMEMORY_USE_PYTHON OR TIMEMORY_BUILD_PYTHON)

    # C++ standard
    if(NOT WIN32 AND NOT "${PYBIND11_CPP_STANDARD}" STREQUAL "-std=c++${CMAKE_CXX_STANDARD}")
        set(PYBIND11_CPP_STANDARD -std=c++${CMAKE_CXX_STANDARD}
            CACHE STRING "PyBind11 CXX standard" FORCE)
    endif()

    set(PYBIND11_INSTALL ON CACHE BOOL "Enable Pybind11 installation")

    if(NOT TIMEMORY_USE_PYTHON OR NOT pybind11_FOUND)
        # checkout PyBind11 if not checked out
        checkout_git_submodule(RECURSIVE
            RELATIVE_PATH external/pybind11
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            REPO_URL https://github.com/jrmadsen/pybind11.git
            REPO_BRANCH master)

        # add PyBind11 to project
        if(NOT TARGET pybind11)
            add_subdirectory(${PROJECT_SOURCE_DIR}/external/pybind11)
        endif()
    endif()


    if(NOT PYBIND11_PYTHON_VERSION)
        unset(PYBIND11_PYTHON_VERSION CACHE)
        execute_process(COMMAND ${PYTHON_EXECUTABLE}
            -c "import sys; print('{}.{}'.format(sys.version_info[0], sys.version_info[1]))"
            OUTPUT_VARIABLE PYTHON_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION}" CACHE STRING "Python version")
    endif()

    add_feature(PYBIND11_CPP_STANDARD "PyBind11 C++ standard")
    add_feature(PYBIND11_PYTHON_VERSION "PyBind11 Python version")

    execute_process(COMMAND ${PYTHON_EXECUTABLE}
        -c "import time ; print('{} {}'.format(time.ctime(), time.tzname[0]))"
        OUTPUT_VARIABLE TIMEMORY_INSTALL_DATE
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

    string(REPLACE "  " " " TIMEMORY_INSTALL_DATE "${TIMEMORY_INSTALL_DATE}")

    if(SKBUILD)
        set(CMAKE_INSTALL_PYTHONDIR ${CMAKE_INSTALL_PREFIX}/timemory)
    else()
        set(CMAKE_INSTALL_PYTHONDIR
            ${CMAKE_INSTALL_LIBDIR}/python${PYBIND11_PYTHON_VERSION}/site-packages/timemory)
    endif()

    if(NOT TIMEMORY_USE_PYTHON OR NOT pybind11_FOUND)
        target_compile_definitions(timemory-python INTERFACE TIMEMORY_USE_PYTHON)
        target_include_directories(timemory-python SYSTEM INTERFACE
            ${PYTHON_INCLUDE_DIRS}
            $<BUILD_INTERFACE:${PYBIND11_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
        target_link_libraries(timemory-python INTERFACE ${PYTHON_LIBRARIES})
    elseif(pybind11_FOUND)
        target_compile_definitions(timemory-python INTERFACE TIMEMORY_USE_PYTHON)
        target_include_directories(timemory-python SYSTEM INTERFACE
            ${PYTHON_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIR} ${PYBIND11_INCLUDE_DIRS})
        target_link_libraries(timemory-python INTERFACE ${PYTHON_LIBRARIES})
    endif()
else()
    inform_empty_interface(timemory-python "Python embedded interpreter")
endif()


#----------------------------------------------------------------------------------------#
#
#                               PAPI
#
#----------------------------------------------------------------------------------------#

find_package(PAPI QUIET)

if(TIMEMORY_USE_PAPI AND PAPI_FOUND)
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
    find_library(GCOV_LIBRARY gcov QUIET)

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
    set(PROJECT_CUDA_DISABLE_HALF2_OPTION       TIMEMORY_DISABLE_CUDA_HALF2)
    set(PROJECT_CUDA_DISABLE_HALF2_DEFINITION   TIMEMORY_DISABLE_CUDA_HALF2)

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
    find_package(CUPTI)
endif()

if(CUPTI_FOUND)

    target_compile_definitions(timemory-cupti INTERFACE
        TIMEMORY_USE_CUPTI)

    target_include_directories(timemory-cupti SYSTEM INTERFACE
        ${CUPTI_INCLUDE_DIRS})

    target_link_libraries(timemory-cupti INTERFACE
        ${CUPTI_LIBRARIES}
        timemory-cuda
        timemory-cudart
        timemory-cudart-device)

    target_link_directories(timemory-cupti INTERFACE
        $<INSTALL_INTERFACE:${CUPTI_LIBRARY_DIRS}>)

    set_target_properties(timemory-cupti PROPERTIES
        INTERFACE_INSTALL_RPATH                 ""
        INTERFACE_INSTALL_RPATH_USE_LINK_PATH   ${HAS_CUDA_DRIVER_LIBRARY})

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
    find_package(NVTX QUIET)
endif()

if(NVTX_FOUND AND TIMEMORY_USE_CUDA)
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

set(_GPERF_COMPONENTS ${TIMEMORY_GPERF_COMPONENTS} profiler tcmalloc)
list(REMOVE_DUPLICATES _GPERF_COMPONENTS)

if(NOT "${TIMEMORY_GPERF_COMPONENTS}" STREQUAL "")
    find_package(gperftools QUIET COMPONENTS ${_GPERF_COMPONENTS})
endif()

if(gperftools_FOUND)
    set(_HAS_PROFILER OFF)
    set(_HAS_TCMALLOC OFF)
    if("profiler" IN_LIST TIMEMORY_GPERF_COMPONENTS OR
        "tcmalloc_and_profiler" IN_LIST TIMEMORY_GPERF_COMPONENTS)
        target_compile_definitions(timemory-gperftools INTERFACE
            TIMEMORY_USE_GPERF_CPU_PROFILER)
        if(gperftools_PROFILER_LIBRARY)
            target_compile_definitions(timemory-gperftools-cpu INTERFACE
                TIMEMORY_USE_GPERF_CPU_PROFILER)
            target_include_directories(timemory-gperftools-cpu SYSTEM INTERFACE
                ${gperftools_INCLUDE_DIRS})
            target_link_libraries(timemory-gperftools-cpu INTERFACE
                ${gperftools_PROFILER_LIBRARY})
            add_target_flag_if_avail(timemory-gperftools-cpu "-g")
        else()
            inform_empty_interface(timemory-gperftools-cpu "gperftools-cpu")
        endif()
        set(_HAS_PROFILER ON)
    endif()
    if("tcmalloc" IN_LIST TIMEMORY_GPERF_COMPONENTS OR
        "tcmalloc_and_profiler" IN_LIST TIMEMORY_GPERF_COMPONENTS OR
        "tcmalloc_debug" IN_LIST TIMEMORY_GPERF_COMPONENTS OR
        "tcmalloc_minimal" IN_LIST TIMEMORY_GPERF_COMPONENTS OR
        "tcmalloc_minimal_debug" IN_LIST TIMEMORY_GPERF_COMPONENTS)
        target_compile_definitions(timemory-gperftools INTERFACE
            TIMEMORY_USE_GPERF_HEAP_PROFILER)
        if(gperftools_TCMALLOC_LIBRARY)
            target_compile_definitions(timemory-gperftools-heap INTERFACE
                TIMEMORY_USE_GPERF_HEAP_PROFILER)
            target_include_directories(timemory-gperftools-heap SYSTEM INTERFACE
                ${gperftools_INCLUDE_DIRS})
            target_link_libraries(timemory-gperftools-heap INTERFACE
                ${gperftools_TCMALLOC_LIBRARY})
            add_target_flag_if_avail(timemory-gperftools-heap "-g")
        else()
            inform_empty_interface(timemory-gperftools-heap "gperftools-heap")
        endif()
        set(_HAS_TCMALLOC ON)
    endif()
    if(_HAS_PROFILER AND _HAS_TCMALLOC)
        target_compile_definitions(timemory-gperftools INTERFACE
            TIMEMORY_USE_GPERF)
    endif()
    target_include_directories(timemory-gperftools SYSTEM INTERFACE
        ${gperftools_INCLUDE_DIRS})
    target_link_libraries(timemory-gperftools INTERFACE
        ${gperftools_LIBRARIES})
    add_target_flag_if_avail(timemory-gperftools "-g")
else()
    set(TIMEMORY_USE_GPERF OFF)
    inform_empty_interface(timemory-gperftools "gperftools")
    inform_empty_interface(timemory-gperftools-cpu "gperftools-cpu")
    inform_empty_interface(timemory-gperftools-heap "gperftools-heap")
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
    set(caliper_DIR ${CMAKE_INSTALL_PREFIX})
else()
    if(TIMEMORY_USE_CALIPER)
        find_package(caliper QUIET)
    endif()
endif()

if(caliper_FOUND)
    target_compile_definitions(timemory-caliper INTERFACE TIMEMORY_USE_CALIPER)
    if(TIMEMORY_BUILD_CALIPER)
        target_include_directories(timemory-caliper SYSTEM INTERFACE
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/caliper/include>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/caliper/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
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
        list(APPEND TIMEMORY_ADDITIONAL_EXPORT_TARGETS gotcha gotcha-include)
    elseif(TIMEMORY_USE_GOTCHA)
        find_package(gotcha QUIET)
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
    target_link_libraries(timemory-gotcha INTERFACE gotcha)
    add_target_flag_if_avail(timemory-gotcha "-rdynamic")
    set_target_properties(timemory-gotcha PROPERTIES
        INTERFACE_LINK_DIRECTORIES $<INSTALL_INTERFACE:${CMAKE_INSTALL_LIBDIR}>)
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
    find_package(LIKWID)
endif()

if(LIKWID_FOUND)
    target_link_libraries(timemory-likwid INTERFACE ${LIKWID_LIBRARIES})
    target_include_directories(timemory-likwid SYSTEM INTERFACE ${LIKWID_INCLUDE_DIRS})
    target_compile_definitions(timemory-likwid INTERFACE TIMEMORY_USE_LIKWID)
else()
    set(TIMEMORY_USE_LIKWID OFF)
    inform_empty_interface(timemory-likwid "LIKWID")
endif()


#----------------------------------------------------------------------------------------#
#
#                               VTune
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_VTUNE)
    find_package(ittnotify)
endif()

if(ittnotify_FOUND)
    target_link_libraries(timemory-vtune INTERFACE ${ITTNOTIFY_LIBRARIES})
    target_include_directories(timemory-vtune SYSTEM INTERFACE ${ITTNOTIFY_INCLUDE_DIRS})
    target_compile_definitions(timemory-vtune INTERFACE TIMEMORY_USE_VTUNE)
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
    find_package(TAU QUIET)
endif()

if(TAU_FOUND)
    target_link_libraries(timemory-tau INTERFACE ${TAU_LIBRARIES})
    target_include_directories(timemory-tau SYSTEM INTERFACE ${TAU_INCLUDE_DIRS})
    target_compile_definitions(timemory-tau INTERFACE TIMEMORY_USE_TAU)
else()
    set(TIMEMORY_USE_TAU OFF)
    inform_empty_interface(timemory-tau "TAU")
endif()

#----------------------------------------------------------------------------------------#
#
#                               Roofline
#
#----------------------------------------------------------------------------------------#

add_target_flag_if_avail(timemory-roofline-options INTERFACE
    "-finline-functions" "-funroll-loops" "-ftree-vectorize"
    "-ftree-loop-optimize" "-ftree-loop-vectorize" "-O3")

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
    timemory-cudart
    timemory-cudart-device)

generate_non_empty_interface(timemory-roofline
    timemory-cpu-roofline
    timemory-gpu-roofline)

#----------------------------------------------------------------------------------------#
# activate clang-tidy if enabled
#
_timemory_activate_clang_tidy()
