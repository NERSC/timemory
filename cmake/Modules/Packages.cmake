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
add_interface_library(timemory-threading)

add_interface_library(timemory-papi)
add_interface_library(timemory-papi-static)
add_interface_library(timemory-cuda)
add_interface_library(timemory-cupti)
add_interface_library(timemory-cudart)
add_interface_library(timemory-cudart-device)
add_interface_library(timemory-cudart-static)
add_interface_library(timemory-nvtx)
add_interface_library(timemory-caliper)
add_interface_library(timemory-gotcha)

add_interface_library(timemory-coverage)
add_interface_library(timemory-exceptions)
add_interface_library(timemory-gperftools)
add_interface_library(timemory-gperftools-cpu)
add_interface_library(timemory-gperftools-heap)

set(_MPI_INTERFACE_LIBRARY)
if(TIMEMORY_USE_MPI)
    set(_MPI_INTERFACE_LIBRARY timemory-mpi)
endif()

set(TIMEMORY_EXTENSION_INTERFACES
    timemory-mpi
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
    timemory-gotcha)

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
    ${_MPI_INTERFACE_LIBRARY})

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
    ${_MPI_INTERFACE_LIBRARY})

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
    message(STATUS  "[interface] ${_PACKAGE} not found. '${_TARGET}' interface will not provide ${_PACKAGE}...")
    # message(AUTHOR_WARNING "[interface] ${_PACKAGE} not found. '${_TARGET}' interface will not provide ${_PACKAGE}...")
    set(TIMEMORY_EMPTY_INTERFACE_LIBRARIES ${TIMEMORY_EMPTY_INTERFACE_LIBRARIES} ${_TARGET} PARENT_SCOPE)
    add_disabled_interface(${_TARGET})
endfunction()

#----------------------------------------------------------------------------------------#
#
#                               timemory headers
#
#----------------------------------------------------------------------------------------#

target_include_directories(timemory-headers INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

if(TIMEMORY_LINK_RT)
    target_link_libraries(timemory-headers INTERFACE rt)
endif()
# include threading because of rooflines
target_link_libraries(timemory-headers INTERFACE timemory-threading)

#----------------------------------------------------------------------------------------#
#
#                               timemory exceptions
#
#----------------------------------------------------------------------------------------#

target_compile_definitions(timemory-exceptions INTERFACE TIMEMORY_EXCEPTIONS)


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
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# add cereal
add_subdirectory(${PROJECT_SOURCE_DIR}/external/cereal)

target_include_directories(timemory-cereal INTERFACE
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
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

    # add google-test
    set(INSTALL_GTEST OFF CACHE BOOL "Install gtest")
    set(BUILD_GMOCK ON CACHE BOOL "Build gmock")
    if(APPLE)
        set(CMAKE_MACOSX_RPATH ON CACHE BOOL "Enable MACOS_RPATH on targets to suppress warnings")
        mark_as_advanced(CMAKE_MACOSX_RPATH)
    endif()
    add_subdirectory(${PROJECT_SOURCE_DIR}/external/google-test)
    target_link_libraries(timemory-google-test INTERFACE gtest gmock gtest_main)
    target_include_directories(timemory-google-test INTERFACE
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
        target_include_directories(timemory-mpi INTERFACE ${MPI_${_LANG}_INCLUDE_PATH})

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
        target_include_directories(timemory-mpi INTERFACE ${MPI_INCLUDE_PATH})
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
#                               PyBind11
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_BUILD_PYTHON)

    # checkout PyBind11 if not checked out
    checkout_git_submodule(RECURSIVE
        RELATIVE_PATH external/pybind11
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

    # C++ standard
    if(NOT "${PYBIND11_CPP_STANDARD}" STREQUAL "${CMAKE_CXX_STANDARD}")
        set(PYBIND11_CPP_STANDARD -std=c++${CMAKE_CXX_STANDARD}
            CACHE STRING "PyBind11 CXX standard" FORCE)
    endif()

    set(PYBIND11_INSTALL OFF)
    # add PyBind11 to project
    if(NOT TARGET pybind11)
        add_subdirectory(${PROJECT_SOURCE_DIR}/external/pybind11)
    endif()

    if(NOT PYBIND11_PYTHON_VERSION)
        unset(PYBIND11_PYTHON_VERSION CACHE)
        execute_process(COMMAND ${PYTHON_EXECUTABLE}
            -c "import sys; print('{}.{}'.format(sys.version_info[0], sys.version_info[1]))"
            OUTPUT_VARIABLE PYTHON_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION}" CACHE STRING "Python version")
    endif()

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
        target_link_options(timemory-coverage INTERFACE
            $<$<COMPILE_LANGUAGE:C>:--coverage>
            $<$<COMPILE_LANGUAGE:CXX>:--coverage>)
    else()
        target_link_libraries(timemory-coverage INTERFACE
            $<$<COMPILE_LANGUAGE:C>:--coverage>
            $<$<COMPILE_LANGUAGE:CXX>:--coverage>)
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
    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    find_package(CUDA QUIET)

    if("CUDA" IN_LIST LANGUAGES AND CUDA_FOUND)

        target_compile_definitions(timemory-cuda INTERFACE TIMEMORY_USE_CUDA)
        target_include_directories(timemory-cuda INTERFACE ${CUDA_INCLUDE_DIRS}
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

        set_target_properties(timemory-cuda PROPERTIES
            INTERFACE_CUDA_STANDARD                 ${CMAKE_CUDA_STANDARD}
            INTERFACE_CUDA_STANDARD_REQUIRED        ${CMAKE_CUDA_STANDARD_REQUIRED}
            INTERFACE_CUDA_RESOLVE_DEVICE_SYMBOLS   ON
            INTERFACE_CUDA_SEPARABLE_COMPILATION    ON)

        set(CUDA_AUTO_ARCH "auto")
        set(CUDA_ARCHITECTURES auto kepler tesla maxwell pascal volta turing)
        set(CUDA_ARCH "${CUDA_AUTO_ARCH}" CACHE STRING
            "CUDA architecture (options: ${CUDA_ARCHITECTURES})")
        add_feature(CUDA_ARCH "CUDA architecture (options: ${CUDA_ARCHITECTURES})")
        set_property(CACHE CUDA_ARCH PROPERTY STRINGS ${CUDA_ARCHITECTURES})

        set(cuda_kepler_arch    30)
        set(cuda_tesla_arch     35)
        set(cuda_maxwell_arch   50)
        set(cuda_pascal_arch    60)
        set(cuda_volta_arch     70)
        set(cuda_turing_arch    75)

        if(NOT "${CUDA_ARCH}" STREQUAL "${CUDA_AUTO_ARCH}")
            if(NOT "${CUDA_ARCH}" IN_LIST CUDA_ARCHITECTURES)
                message(WARNING
                    "CUDA architecture \"${CUDA_ARCH}\" not known. Options: ${CUDA_ARCH}")
                unset(CUDA_ARCH CACHE)
                set(CUDA_ARCH "${CUDA_AUTO_ARCH}")
            else()
                set(_ARCH_NUM ${cuda_${CUDA_ARCH}_arch})
                if(_ARCH_NUM LESS 60)
                    set(TIMEMORY_DISABLE_CUDA_HALF2 ON)
                endif()
            endif()
        endif()

        option(TIMEMORY_DEPRECATED_CUDA_SUPPORT "Enable support for old CUDA flags" OFF)
        mark_as_advanced(TIMEMORY_DEPRECATED_CUDA_SUPPORT)

        if(TIMEMORY_DEPRECATED_CUDA_SUPPORT)
            add_interface_library(timemory-cuda-8)
            target_compile_options(timemory-cuda-8 INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
                $<IF:$<STREQUAL:${CUDA_ARCH},${CUDA_AUTO_ARCH}>,-arch=sm_30,-arch=sm_${_ARCH_NUM}>
                -gencode=arch=compute_30,code=sm_30
                -gencode=arch=compute_35,code=sm_35
                -gencode=arch=compute_50,code=sm_50
                -gencode=arch=compute_52,code=sm_52
                -gencode=arch=compute_60,code=sm_60
                -gencode=arch=compute_61,code=sm_61
                -gencode=arch=compute_61,code=compute_61
                >)
        endif()

        add_interface_library(timemory-cuda-9)
        target_compile_options(timemory-cuda-9 INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
            $<IF:$<STREQUAL:${CUDA_ARCH},${CUDA_AUTO_ARCH}>,-arch=sm_60,-arch=sm_${_ARCH_NUM}>
            -gencode=arch=compute_60,code=sm_60
            -gencode=arch=compute_61,code=sm_61
            -gencode=arch=compute_70,code=sm_70
            -gencode=arch=compute_70,code=compute_70
            >)

        add_interface_library(timemory-cuda-10)
        target_compile_options(timemory-cuda-10 INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
            $<IF:$<STREQUAL:"${CUDA_ARCH}","${CUDA_AUTO_ARCH}">,-arch=sm_60,-arch=sm_${_ARCH_NUM}>
            -gencode=arch=compute_60,code=sm_60
            -gencode=arch=compute_61,code=sm_61
            -gencode=arch=compute_70,code=sm_70
            -gencode=arch=compute_75,code=sm_75
            -gencode=arch=compute_75,code=compute_75
            >)

        string(REPLACE "." ";" CUDA_MAJOR_VERSION "${CUDA_VERSION}")
        list(GET CUDA_MAJOR_VERSION 0 CUDA_MAJOR_VERSION)

        if(CUDA_MAJOR_VERSION VERSION_GREATER 10 OR CUDA_MAJOR_VERSION MATCHES 10)
            target_link_libraries(timemory-cuda INTERFACE timemory-cuda-10)
        elseif(CUDA_MAJOR_VERSION MATCHES 9)
            target_link_libraries(timemory-cuda INTERFACE timemory-cuda-9)
        else()
            if(TIMEMORY_DEPRECATED_CUDA_SUPPORT)
                if(CUDA_MAJOR_VERSION MATCHES 8)
                    target_link_libraries(timemory-cuda INTERFACE timemory-cuda-8)
                elseif(CUDA_MAJOR_VERSION MATCHES 7)
                    target_link_libraries(timemory-cuda INTERFACE timemory-cuda-7)
                endif()
            else()
                message(WARNING "CUDA version < 9 detected. Enable TIMEMORY_DEPRECATED_CUDA_SUPPORT")
            endif()
        endif()

        #   30, 32      + Kepler support
        #               + Unified memory programming
        #   35          + Dynamic parallelism support
        #   50, 52, 53  + Maxwell support
        #   60, 61, 62  + Pascal support
        #   70, 72      + Volta support
        #   75          + Turing support

        #target_compile_options(timemory-cuda INTERFACE
        #    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)

        add_user_flags(timemory-cuda "CUDA")

        target_compile_options(timemory-cuda INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

        if(TIMEMORY_DISABLE_CUDA_HALF2)
            target_compile_definitions(timemory-cuda INTERFACE
                TIMEMORY_DISABLE_CUDA_HALF2)
        endif()

        if(NOT WIN32)
            target_compile_options(timemory-cuda INTERFACE
                $<$<COMPILE_LANGUAGE:CUDA>:--compiler-bindir=${CMAKE_CXX_COMPILER}>)
        endif()

        target_include_directories(timemory-cuda INTERFACE ${CUDA_INCLUDE_DIRS}
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

        find_library(CUDA_dl_LIBRARY
            NAMES dl)

        target_compile_options(timemory-cudart INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:--cudart=shared>)

        target_compile_options(timemory-cudart-static INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:--cudart=static>)

        target_link_libraries(timemory-cudart INTERFACE
            ${CUDA_CUDART_LIBRARY} ${CUDA_rt_LIBRARY})

        target_link_libraries(timemory-cudart-device INTERFACE
            ${CUDA_cudadevrt_LIBRARY} ${CUDA_rt_LIBRARY})

        target_link_libraries(timemory-cudart-static INTERFACE
            ${CUDA_cudart_static_LIBRARY} ${CUDA_rt_LIBRARY})

        if(CUDA_dl_LIBRARY)
            target_link_libraries(timemory-cudart INTERFACE
                ${CUDA_dl_LIBRARY})

            target_link_libraries(timemory-cudart-device INTERFACE
                ${CUDA_dl_LIBRARY})

            target_link_libraries(timemory-cudart-static INTERFACE
                ${CUDA_dl_LIBRARY})
        endif()

    else()
        inform_empty_interface(timemory-cuda "CUDA")
        set(TIMEMORY_USE_CUDA OFF)
    endif()
else()
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

    set(_CUDA_PATHS $ENV{CUDA_HOME} ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR})
    set(_CUDA_INC ${CUDA_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    find_path(CUPTI_ROOT_DIR
        NAMES           include/cupti.h
        HINTS           ${_CUDA_PATHS}
        PATHS           ${_CUDA_PATHS}
        PATH_SUFFIXES   extras/CUPTI)

    # try to find cupti header
    find_path(CUDA_cupti_INCLUDE_DIR
        NAMES           cupti.h
        HINTS           ${CUPTI_ROOT_DIR} ${_CUDA_INC} ${_CUDA_PATHS}
        PATHS           ${CUPTI_ROOT_DIR} ${_CUDA_INC} ${_CUDA_PATHS}
        PATH_SUFFIXES   extras/CUPTI/include extras/CUPTI extras/include CUTPI/include include)

    # try to find cuda driver library
    find_library(CUDA_cupti_LIBRARY
        NAMES           cupti
        HINTS           ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
        PATHS           ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
        PATH_SUFFIXES   lib lib64 lib/nvidia lib64/nvidia nvidia)

    # try to find cuda driver library
    find_library(CUDA_cuda_LIBRARY
        NAMES           cuda
        HINTS           ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
        PATHS           ${CUPTI_ROOT_DIR} ${_CUDA_PATHS}
        PATH_SUFFIXES   lib lib64 lib/nvidia lib64/nvidia nvidia)

    # try to find cuda driver stubs library if no real driver
    if(NOT CUDA_cuda_LIBRARY)
        find_library(CUDA_cuda_LIBRARY
            NAMES           cuda
            HINTS           ${_CUDA_PATHS}
            PATHS           ${_CUDA_PATHS}
            PATH_SUFFIXES   lib/stubs lib64/stubs stubs)
        set(HAS_CUDA_cuda_LIBRARY OFF CACHE BOOL "Using stubs library")
    else()
        set(HAS_CUDA_cuda_LIBRARY ON CACHE BOOL "Using stubs library")
    endif()

    find_package_handle_standard_args(CUDA_CUPTI DEFAULT_MSG
        CUDA_cupti_INCLUDE_DIR
        CUDA_cupti_LIBRARY
        CUDA_cuda_LIBRARY)

    # if header and library found
    if(NOT CUDA_CUPTI_FOUND)
        set(TIMEMORY_USE_CUPTI OFF)
    else()
        target_compile_definitions(timemory-cupti INTERFACE TIMEMORY_USE_CUPTI)
        target_include_directories(timemory-cupti INTERFACE ${CUDA_INCLUDE_DIRS}
            ${CUDA_cupti_INCLUDE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        target_link_libraries(timemory-cupti INTERFACE ${CUDA_cupti_LIBRARY} ${CUDA_cuda_LIBRARY})
        set_target_properties(timemory-cupti PROPERTIES
            INTERFACE_INSTALL_RPATH               ""
            INTERFACE_INSTALL_RPATH_USE_LINK_PATH ${HAS_CUDA_cuda_LIBRARY})
    endif()

    # clean-up
    unset(_CUDA_PATHS)
    unset(_CUDA_INC)

else()
    inform_empty_interface(timemory-cupti "CUPTI")
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
    target_include_directories(timemory-nvtx INTERFACE ${NVTX_INCLUDE_DIRS})
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
find_package(gperftools QUIET COMPONENTS ${_GPERF_COMPONENTS})

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
            target_include_directories(timemory-gperftools-cpu INTERFACE
                ${gperftools_INCLUDE_DIRS})
            target_link_libraries(timemory-gperftools-cpu INTERFACE
                ${gperftools_PROFILER_LIBRARY})
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
            target_include_directories(timemory-gperftools-heap INTERFACE
                ${gperftools_INCLUDE_DIRS})
            target_link_libraries(timemory-gperftools-heap INTERFACE
                ${gperftools_TCMALLOC_LIBRARY})
        else()
            inform_empty_interface(timemory-gperftools-heap "gperftools-heap")
        endif()
        set(_HAS_TCMALLOC ON)
    endif()
    if(_HAS_PROFILER AND _HAS_TCMALLOC)
        target_compile_definitions(timemory-gperftools INTERFACE
            TIMEMORY_USE_GPERF)
    endif()
    target_include_directories(timemory-gperftools INTERFACE
        ${gperftools_INCLUDE_DIRS})
    target_link_libraries(timemory-gperftools INTERFACE
        ${gperftools_LIBRARIES})
else()
    set(TIMEMORY_USE_GPERF OFF)
    inform_empty_interface(timemory-gperftools "gperftools")
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
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
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
        target_include_directories(timemory-caliper INTERFACE
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
    else()
        target_include_directories(timemory-caliper INTERFACE ${caliper_INCLUDE_DIR})
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
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
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
else()
    set(TIMEMORY_USE_GOTCHA OFF)
    inform_empty_interface(timemory-gotcha "GOTCHA")
endif()


#----------------------------------------------------------------------------------------#
# activate clang-tidy if enabled
#
_timemory_activate_clang_tidy()
