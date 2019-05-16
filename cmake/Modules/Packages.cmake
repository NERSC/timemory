################################################################################
#
#                               Component
#
################################################################################

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME external)


################################################################################
#
#                               Threading
#
################################################################################

find_library(PTHREADS_LIBRARY pthread)
if(PTHREADS_LIBRARY)
    list(APPEND EXTERNAL_LIBRARIES ${PTHREADS_LIBRARY})
else()
    if(NOT WIN32)
        set(CMAKE_THREAD_PREFER_PTHREAD ON)
    endif()

    find_package(Threads)

    if(Threads_FOUND)
        list(APPEND PRIVATE_EXTERNAL_LIBRARIES Threads::Threads)
    endif()
endif()


################################################################################
#
#                               MPI
#
################################################################################

if(TIMEMORY_USE_MPI)

    if(WIN32)
        if(EXISTS "C:/Program\ Files\ (x86)/Microsoft\ SDKs/MPI")
            list(APPEND CMAKE_PREFIX_PATH "C:/Program\ Files\ (x86)/Microsoft\ SDKs/MPI")
        endif(EXISTS "C:/Program\ Files\ (x86)/Microsoft\ SDKs/MPI")

        if(EXISTS "C:/Program\ Files/Microsoft\ SDKs/MPI")
            list(APPEND CMAKE_PREFIX_PATH "C:/Program\ Files/Microsoft\ SDKs/MPI")
        endif(EXISTS "C:/Program\ Files/Microsoft\ SDKs/MPI")
    endif()

    # MPI C compiler from environment
    set(_ENV MPICC)
    if(NOT DEFINED MPI_C_COMPILER AND NOT "$ENV{${_ENV}}" STREQUAL "")
        message(STATUS "Setting MPI C compiler to: $ENV{${_ENV}}")
        set(MPI_C_COMPILER $ENV{${_ENV}} CACHE FILEPATH "MPI C compiler")
    endif()

    # MPI C++ compiler from environment
    set(_ENV MPICXX)
    if(NOT DEFINED MPI_CXX_COMPILER AND NOT "$ENV{${_ENV}}" STREQUAL "")
        message(STATUS "Setting MPI C++ compiler to: $ENV{${_ENV}}")
        set(MPI_CXX_COMPILER $ENV{${_ENV}} CACHE FILEPATH "MPI C++ compiler")
    endif()
    unset(_ENV)

    find_package(MPI)

    if(MPI_FOUND)
        set(_MPI_LIBRARIES )
        foreach(_LANG C CXX)
            # include directories
            list(APPEND EXTERNAL_INCLUDE_DIRS ${MPI_${_LANG}_INCLUDE_PATH})

            # link targets
            set(_TYPE MPI_${_LANG}_LIBRARIES)
            if(${_TYPE})
                list(APPEND EXTERNAL_LIBRARIES ${${_TYPE}})
            endif()

            # compile flags
            to_list(_FLAGS "${MPI_${_LANG}_COMPILE_FLAGS}")
            foreach(_FLAG ${_FLAGS})
                if("${_LANG}" STREQUAL "CXX")
                    add_cxx_flag_if_avail("${_FLAG}")
                else()
                    add_c_flag_if_avail("${_FLAG}")
                endif()
            endforeach()
            unset(_FLAGS)

            # compile flags
            to_list(_FLAGS "${MPI_${_LANG}_LINK_FLAGS}")
            foreach(_FLAG ${_FLAGS})
                list(APPEND EXTERNAL_${_LANG}_LINK_OPTIONS ${_FLAG})
            endforeach()
            unset(_FLAGS)

        endforeach()

        if(MPI_EXTRA_LIBRARY)
            list(APPEND EXTERNAL_LIBRARIES ${MPI_EXTRA_LIBRARY})
        endif()

        if(MPI_INCLUDE_PATH)
            list(APPEND EXTERNAL_INCLUDE_DIRS ${MPI_INCLUDE_PATH})
        endif()

        list(APPEND ${PROJECT_NAME}_DEFINITIONS TIMEMORY_USE_MPI)

        if(NOT MPIEXEC_EXECUTABLE AND MPIEXEC)
            set(MPIEXEC_EXECUTABLE ${MPIEXEC} CACHE FILEPATH "MPI executable")
        endif()

        if(NOT MPIEXEC_EXECUTABLE AND MPI_EXECUTABLE)
            set(MPIEXEC_EXECUTABLE ${MPI_EXECUTABLE} CACHE FILEPATH "MPI executable")
        endif()

    else()

        set(TIMEMORY_USE_MPI OFF)
        message(WARNING "MPI not found. Proceeding without MPI")

    endif()

endif()


################################################################################
#
#                               PyBind11
#
################################################################################

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


################################################################################
#
#                               PAPI
#
################################################################################

if(TIMEMORY_USE_PAPI)
    find_package(PAPI)

    if(PAPI_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${PAPI_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${PAPI_LIBRARIES})
        list(APPEND ${PROJECT_NAME}_DEFINITIONS TIMEMORY_USE_PAPI)
    else(PAPI_FOUND)
        set(TIMEMORY_USE_PAPI OFF)
        message(WARNING "PAPI package not found!")
    endif(PAPI_FOUND)

endif()


################################################################################
#
#        Coverage
#
################################################################################

if(TIMEMORY_USE_COVERAGE)

    find_library(GCOV_LIBRARY gcov QUIET)

    if(GCOV_LIBRARY)
        list(APPEND EXTERNAL_LIBRARIES ${COVERAGE_LIBRARY})
    elseif(CMAKE_CXX_COMPILER_IS_GNU)
        list(APPEND EXTERNAL_LIBRARIES gcov)
    else()
        message(STATUS "GCov library not found. Disabling coverage...")
        set(TIMEMORY_USE_COVERAGE OFF)
    endif()

endif()


################################################################################
#
#        CUDA
#
################################################################################

if(TIMEMORY_USE_CUDA)
    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    find_package(CUDA)

    if("CUDA" IN_LIST LANGUAGES AND CUDA_FOUND)
        list(APPEND ${PROJECT_NAME}_DEFINITIONS TIMEMORY_USE_CUDA)
        add_feature(CUDA_ARCH "CUDA architecture (e.g. '35' means '-arch=sm_35')")

        #   30, 32      + Kepler support
        #               + Unified memory programming
        #   35          + Dynamic parallelism support
        #   50, 52, 53  + Maxwell support
        #   60, 61, 62  + Pascal support
        #   70, 72      + Volta support
        #   75          + Turing support
        if(NOT DEFINED CUDA_ARCH)
            set(CUDA_ARCH "35")
        endif()

        list(APPEND ${PROJECT_NAME}_CUDA_FLAGS
            -arch=sm_${CUDA_ARCH}
            --default-stream per-thread
        )

        if(NOT WIN32)
            list(APPEND ${PROJECT_NAME}_CUDA_FLAGS}
                --compiler-bindir=${CMAKE_CXX_COMPILER})
        endif()

        list(APPEND EXTERNAL_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    else()
        set(TIMEMORY_USE_CUDA OFF)
    endif()
endif()


################################################################################
#
#        Google PerfTools
#
################################################################################

if(TIMEMORY_USE_GPERF)
    find_package(GPerfTools COMPONENTS profiler)

    if(GPerfTools_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${GPerfTools_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${GPerfTools_LIBRARIES})
        list(APPEND ${PROJECT_NAME}_DEFINITIONS TIMEMORY_USE_GPERF)
    else()
        set(TIMEMORY_USE_GPERF OFF)
        message(WARNING "GPerfTools package not found!")
    endif()

endif()


################################################################################
#
#        Checkout Cereal if not checked out
#
################################################################################

checkout_git_submodule(RECURSIVE
    RELATIVE_PATH source/cereal
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})


################################################################################
#
#        External variables
#
################################################################################

# including the directories
safe_remove_duplicates(EXTERNAL_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS})
safe_remove_duplicates(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES})
safe_remove_duplicates(PRIVATE_EXTERNAL_INCLUDE_DIRS ${PRIVATE_EXTERNAL_INCLUDE_DIRS})
safe_remove_duplicates(PRIVATE_EXTERNAL_LIBRARIES ${PRIVATE_EXTERNAL_LIBRARIES})

set(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES} PRIVATE ${PRIVATE_EXTERNAL_LIBRARIES})

list(APPEND ${PROJECT_NAME}_TARGET_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS})
list(APPEND ${PROJECT_NAME}_TARGET_LIBRARIES ${EXTERNAL_LIBRARIES})


################################################################################
#
#                               Component
#
################################################################################

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME development)
