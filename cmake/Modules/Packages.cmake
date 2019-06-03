##########################################################################################
#
#                       External Packages are found here
#
##########################################################################################

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME external)


#----------------------------------------------------------------------------------------#
#
#                               TiMemory headers
#
#----------------------------------------------------------------------------------------#

add_interface_library(timemory-headers)
target_include_directories(timemory-headers INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>)
target_include_directories(timemory-headers SYSTEM INTERFACE
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)


#----------------------------------------------------------------------------------------#
#
#                               TiMemory extern-templates
#
#----------------------------------------------------------------------------------------#

add_interface_library(timemory-extern-templates)
if(TIMEMORY_BUILD_EXTERN_TEMPLATES)
    target_compile_definitions(timemory-extern-templates INTERFACE TIMEMORY_EXTERN_TEMPLATES)
endif()


#----------------------------------------------------------------------------------------#
#
#                               TiMemory external libraries
#
#----------------------------------------------------------------------------------------#

add_interface_library(timemory-extensions)


#----------------------------------------------------------------------------------------#
#
#                           Cereal (serialization library)
#
#----------------------------------------------------------------------------------------#

checkout_git_submodule(RECURSIVE
    RELATIVE_PATH source/cereal
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

set(DEV_WARNINGS ${CMAKE_SUPPRESS_DEVELOPER_WARNINGS})
# this gets annoying
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)

# add cereal
if(NOT TIMEMORY_SETUP_PY OR TIMEMORY_DEVELOPER_INSTALL)
    add_subdirectory(${PROJECT_SOURCE_DIR}/source/cereal)
endif()

add_external_library(timemory-cereal)
target_include_directories(timemory-cereal SYSTEM INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/cereal/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ${DEV_WARNINGS} CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)

# timemory-headers always provides timemory-cereal
target_link_libraries(timemory-headers INTERFACE timemory-cereal)


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

# empty target if nothing found
add_external_library(timemory-threading)

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

    if(MPI_FOUND)
        add_external_library(timemory-mpi)

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
        message(WARNING "MPI not found! Proceeding without MPI...")

    endif()

endif()


#----------------------------------------------------------------------------------------#
#
#                               PyBind11
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_BUILD_PYTHON)

    # checkout PyBind11 if not checked out
    checkout_git_submodule(RECURSIVE
        RELATIVE_PATH source/python/pybind11
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

    # make sure pybind11 gets installed in same place as TiMemory
    if(PYBIND11_INSTALL)
        set(PYBIND11_CMAKECONFIG_INSTALL_DIR
            "${TIMEMORY_INSTALL_DATAROOTDIR}/cmake/pybind11"
            CACHE STRING "install path for pybind11Config.cmake" FORCE)
        set(CMAKE_INSTALL_INCLUDEDIR ${TIMEMORY_INSTALL_INCLUDEDIR}
            CACHE PATH "Include file installation path" FORCE)
    endif()

    # C++ standard
    set(PYBIND11_CPP_STANDARD -std=c++${CMAKE_CXX_STANDARD}
        CACHE STRING "PyBind11 CXX standard" FORCE)

    # add PyBind11 to project
    add_subdirectory(${PROJECT_SOURCE_DIR}/source/python/pybind11)

    if(NOT PYBIND11_PYTHON_VERSION)
        execute_process(COMMAND ${PYTHON_EXECUTABLE}
            -c "import sys; print('{}.{}'.format(sys.version_info[0], sys.version_info[1]))"
            OUTPUT_VARIABLE PYTHON_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        message(STATUS "Python version: ${PYTHON_VERSION}")
        set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION}"
            CACHE STRING "Python version" FORCE)
    endif(NOT PYBIND11_PYTHON_VERSION)

    add_feature(PYBIND11_PYTHON_VERSION "PyBind11 Python version")

    execute_process(COMMAND ${PYTHON_EXECUTABLE}
        -c "import time ; print('{} {}'.format(time.ctime(), time.tzname[0]))"
        OUTPUT_VARIABLE TIMEMORY_INSTALL_DATE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)

    string(REPLACE "  " " " TIMEMORY_INSTALL_DATE "${TIMEMORY_INSTALL_DATE}")

    ########################################
    #   Python installation directories
    ########################################
    set(TIMEMORY_STAGING_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE PATH
        "Installation prefix (relevant in pip staged builds)")

    if(TIMEMORY_SETUP_PY)

        set(TIMEMORY_INSTALL_PYTHONDIR ${TIMEMORY_STAGING_PREFIX}/timemory CACHE PATH
            "Installation prefix of python" FORCE)

        set(TIMEMORY_INSTALL_FULL_PYTHONDIR
            ${CMAKE_INSTALL_PREFIX}/lib/python${PYBIND11_PYTHON_VERSION}/site-packages/timemory)

        add_feature(TIMEMORY_INSTALL_PYTHONDIR "TiMemory Python installation directory")
        add_feature(TIMEMORY_STAGING_PREFIX "Installation prefix (relevant in pip staged builds)")

    else(TIMEMORY_SETUP_PY)

        set(TIMEMORY_INSTALL_PYTHONDIR
            ${CMAKE_INSTALL_LIBDIR}/python${PYBIND11_PYTHON_VERSION}/site-packages/timemory
            CACHE PATH "Installation directory for python")

        set(TIMEMORY_INSTALL_FULL_PYTHONDIR
            ${CMAKE_INSTALL_PREFIX}/${TIMEMORY_INSTALL_PYTHONDIR})

    endif(TIMEMORY_SETUP_PY)

    set(TIMEMORY_CONFIG_PYTHONDIR
        ${CMAKE_INSTALL_LIBDIR}/python${PYBIND11_PYTHON_VERSION}/site-packages/timemory)

else()

    set(TIMEMORY_CONFIG_PYTHONDIR ${CMAKE_INSTALL_PREFIX})

endif()


#----------------------------------------------------------------------------------------#
#
#                               PAPI
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_PAPI)
    find_package(PAPI)

    if(PAPI_FOUND)
        add_external_library(timemory-papi)
        target_include_directories(timemory-papi SYSTEM INTERFACE ${PAPI_INCLUDE_DIRS})
        target_link_libraries(timemory-papi INTERFACE ${PAPI_LIBRARIES})
        target_compile_definitions(timemory-papi INTERFACE TIMEMORY_USE_PAPI)
    else()
        set(TIMEMORY_USE_PAPI OFF)
        message(WARNING "PAPI package not found! Proceeding without PAPI...")
    endif()

endif()


#----------------------------------------------------------------------------------------#
#
#                               Coverage
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_COVERAGE)
    if(CMAKE_CXX_COMPILER_IS_GNU)
        find_library(GCOV_LIBRARY gcov QUIET)

        if(GCOV_LIBRARY OR CMAKE_CXX_COMPILER_IS_GNU)
            add_external_library(timemory-coverage)
            add_c_flag_if_avail("-ftest-coverage" timemory-coverage)
            add_cxx_flag_if_avail("-ftest-coverage" timemory-coverage)
            if(cxx_ftest_coverage)
                if(NOT CMAKE_VERSION VERSION_LESS 3.13)
                    target_link_options(timemory-mpi INTERFACE "-fprofile-arcs")
                else()
                    set_target_properties(timemory-coverage PROPERTIES
                        INTERFACE_LINK_OPTIONS "-fprofile-arcs")
                endif()
            endif()
        endif()

        if(GCOV_LIBRARY)
            target_link_libraries(timemory-coverage INTERFACE ${COVERAGE_LIBRARY})
        elseif(CMAKE_CXX_COMPILER_IS_GNU)
            target_link_libraries(timemory-coverage INTERFACE gcov)
        else()
            message(WARNING "GCov library not found. Disabling coverage...")
            set(TIMEMORY_USE_COVERAGE OFF)
        endif()
    else()
        message(WARNING "Coverage only available for GNU compilers...")
        set(TIMEMORY_USE_COVERAGE OFF)
    endif()
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

        add_external_library(timemory-cuda)

        target_compile_definitions(timemory-cuda INTERFACE TIMEMORY_USE_CUDA)
        target_include_directories(timemory-cuda INTERFACE ${CUDA_INCLUDE_DIRS}
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

        set_target_properties(timemory-cuda PROPERTIES
            INTERFACE_CUDA_STANDARD                 ${CMAKE_CUDA_STANDARD}
            INTERFACE_CUDA_STANDARD_REQUIRED        ${CMAKE_CUDA_STANDARD_REQUIRED}
            INTERFACE_CUDA_RESOLVE_DEVICE_SYMBOLS   ON
            INTERFACE_CUDA_SEPARABLE_COMPILATION    ON)

        set(CUDA_GENERIC_ARCH "version")
        set(CUDA_ARCHITECTURES version kepler tesla maxwell pascal volta turing)
        set(CUDA_ARCH "${CUDA_GENERIC_ARCH}" CACHE STRING "CUDA architecture (options: ${CUDA_ARCHITECTURES})")
        add_feature(CUDA_ARCH "CUDA architecture (options: ${CUDA_ARCHITECTURES})")
        set_property(CACHE CUDA_ARCH PROPERTY STRINGS ${CUDA_ARCHITECTURES})

        set(cuda_kepler_arch    30)
        set(cuda_tesla_arch     35)
        set(cuda_maxwell_arch   50)
        set(cuda_pascal_arch    60)
        set(cuda_volta_arch     70)
        set(cuda_turing_arch    75)

        if(NOT "${CUDA_ARCH}" STREQUAL "${CUDA_GENERIC_ARCH}")
            if(NOT "${CUDA_ARCH}" IN_LIST CUDA_ARCHITECTURES)
                message(WARNING "CUDA architecture \"${CUDA_ARCH}\" not known. Options: ${CUDA_ARCH}")
                unset(CUDA_ARCH CACHE)
                set(CUDA_ARCH "${CUDA_GENERIC_ARCH}")
            else()
                set(_ARCH_NUM ${cuda_${CUDA_ARCH}_arch})
            endif()
        endif()

        message(STATUS "${CUDA_GENERIC_ARCH} ${CUDA_ARCH}")
        add_interface_library(timemory-cuda-7)
        target_compile_options(timemory-cuda-7 INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
            $<IF:$<STREQUAL:${CUDA_ARCH},${CUDA_GENERIC_ARCH}>,-arch=sm_30,-arch=sm_${_ARCH_NUM}>
            -gencode=arch=compute_20,code=sm_20
            -gencode=arch=compute_30,code=sm_30
            -gencode=arch=compute_50,code=sm_50
            -gencode=arch=compute_52,code=sm_52
            -gencode=arch=compute_52,code=compute_52
            >)

        add_interface_library(timemory-cuda-8)
        target_compile_options(timemory-cuda-8 INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
            $<IF:$<STREQUAL:${CUDA_ARCH},${CUDA_GENERIC_ARCH}>,-arch=sm_30,-arch=sm_${_ARCH_NUM}>
            -gencode=arch=compute_20,code=sm_20
            -gencode=arch=compute_30,code=sm_30
            -gencode=arch=compute_50,code=sm_50
            -gencode=arch=compute_52,code=sm_52
            -gencode=arch=compute_60,code=sm_60
            -gencode=arch=compute_61,code=sm_61
            -gencode=arch=compute_61,code=compute_61
            >)

        add_interface_library(timemory-cuda-9)
        target_compile_options(timemory-cuda-9 INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
            $<IF:$<STREQUAL:${CUDA_ARCH},${CUDA_GENERIC_ARCH}>,-arch=sm_50,-arch=sm_${_ARCH_NUM}>
            -gencode=arch=compute_50,code=sm_50
            -gencode=arch=compute_52,code=sm_52
            -gencode=arch=compute_60,code=sm_60
            -gencode=arch=compute_61,code=sm_61
            -gencode=arch=compute_70,code=sm_70
            -gencode=arch=compute_70,code=compute_70
            >)

        add_interface_library(timemory-cuda-10)
        target_compile_options(timemory-cuda-10 INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
            $<IF:$<STREQUAL:"${CUDA_ARCH}","${CUDA_GENERIC_ARCH}">,-arch=sm_50,-arch=sm_${_ARCH_NUM}>
            -gencode=arch=compute_50,code=sm_50
            -gencode=arch=compute_52,code=sm_52
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
        elseif(CUDA_MAJOR_VERSION MATCHES 8)
            target_link_libraries(timemory-cuda INTERFACE timemory-cuda-8)
        elseif(CUDA_MAJOR_VERSION MATCHES 7)
            target_link_libraries(timemory-cuda INTERFACE timemory-cuda-7)
        else()
            target_link_libraries(timemory-cuda INTERFACE timemory-cuda-7)
        endif()

        #   30, 32      + Kepler support
        #               + Unified memory programming
        #   35          + Dynamic parallelism support
        #   50, 52, 53  + Maxwell support
        #   60, 61, 62  + Pascal support
        #   70, 72      + Volta support
        #   75          + Turing support

        target_compile_options(timemory-cuda INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)

        if(NOT WIN32)
            target_compile_options(timemory-cuda INTERFACE
                $<$<COMPILE_LANGUAGE:CUDA>:--compiler-bindir=${CMAKE_CXX_COMPILER}>)
        endif()

        target_include_directories(timemory-cuda INTERFACE ${CUDA_INCLUDE_DIRS}
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

        if(TIMEMORY_USE_CUPTI)
            set(_CUDA_PATHS $ENV{CUDA_HOME} ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR})

            # try to find cupti header
            find_path(CUDA_cupti_INCLUDE_DIR
                NAMES           cupti.h
                HINTS           ${CUDA_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${_CUDA_PATHS}
                PATHS           ${CUDA_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${_CUDA_PATHS}
                PATH_SUFFIXES   extras/CUPTI/include extras/CUPTI extras/include CUTPI/include)

            # try to find cuda driver library
            find_library(CUDA_cuda_LIBRARY
                NAMES           cuda
                HINTS           ${_CUDA_PATHS}
                PATHS           ${_CUDA_PATHS}
                PATH_SUFFIXES   lib lib64 lib/nvidia lib64/nvidia nvidia)

            # try to find cuda driver stubs library if no real driver
            if(NOT CUDA_cuda_LIBRARY)
                set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF CACHE BOOL "Use link path for RPATH" FORCE)
                find_library(CUDA_cuda_LIBRARY
                    NAMES           cuda
                    HINTS           ${_CUDA_PATHS}
                    PATHS           ${_CUDA_PATHS}
                    PATH_SUFFIXES   lib/stubs lib64/stubs stubs)
            endif()

            # if header and library found
            if(CUDA_cupti_INCLUDE_DIR AND CUDA_cupti_LIBRARY AND CUDA_cuda_LIBRARY)
            else()
                set(_MSG "Warning! Unable to find CUPTI. Missing variables:")
                foreach(_VAR CUDA_cupti_INCLUDE_DIR CUDA_cupti_LIBRARY CUDA_cuda_LIBRARY)
                    if(NOT ${_VAR})
                        add(_MSG ${_VAR})
                    endif()
                endforeach()
                set(_MSG "${_MSG}. Disabling TIMEMORY_USE_CUPTI...")
                message(WARNING "${_MSG}")
                set(TIMEMORY_USE_CUPTI OFF)
                unset(_MSG)
            endif()

            # clean-up
            unset(_CUDA_PATHS)
        endif()
    else()
        message(WARNING "CUDA not available!")
        set(TIMEMORY_USE_CUPTI OFF)
        set(TIMEMORY_USE_CUDA OFF)
    endif()
else()
    set(TIMEMORY_USE_CUPTI OFF)
endif()


#----------------------------------------------------------------------------------------#
#
#                               Google PerfTools
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_GPERF)
    find_package(GPerfTools COMPONENTS profiler tcmalloc)

    if(GPerfTools_FOUND)
        add_external_library(timemory-gperf)
        target_compile_definitions(timemory-gperf INTERFACE TIMEMORY_USE_GPERF)
        target_include_directories(timemory-gperf INTERFACE ${GPerfTools_INCLUDE_DIRS})
        target_link_libraries(timemory-gperf INTERFACE ${GPerfTools_LIBRARIES})
    else()
        set(TIMEMORY_USE_GPERF OFF)
        message(WARNING "GPerfTools package not found!")
    endif()

endif()


#----------------------------------------------------------------------------------------#
#
#                               External variables
#
#----------------------------------------------------------------------------------------#

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME development)
