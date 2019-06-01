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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source>)
target_include_directories(timemory-headers SYSTEM INTERFACE
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

add_exported_interface_library(timemory-extern-templates)
add_exported_interface_library(timemory-shared-extern-templates)
add_exported_interface_library(timemory-static-extern-templates)

target_link_libraries(timemory-extern-templates INTERFACE timemory-headers)
target_link_libraries(timemory-shared-extern-templates INTERFACE timemory-headers)
target_link_libraries(timemory-static-extern-templates INTERFACE timemory-headers)

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

if(Threads_FOUND OR (PTHREADS_LIBRARY AND NOT WIN32))
    add_interface_library(timemory-threading)
    target_link_libraries(timemory-headers INTERFACE timemory-threading)
endif()

if(Threads_FOUND)
    target_link_libraries(timemory-threading INTERFACE ${CMAKE_THREAD_LIBS_INIT})
elseif(PTHREADS_LIBRARY AND NOT WIN32)
    target_link_libraries(timemory-threading INTERFACE ${PTHREADS_LIBRARY})
endif()


#----------------------------------------------------------------------------------------#
#
#                               MPI
#
#----------------------------------------------------------------------------------------#

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
        add_interface_library(timemory-mpi)
        target_link_libraries(timemory-headers INTERFACE timemory-mpi)

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
                list(APPEND EXTERNAL_${_LANG}_LINK_OPTIONS ${_FLAG})
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
        message(WARNING "MPI not found. Proceeding without MPI")

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
        add_interface_library(timemory-papi)
        target_include_directories(timemory-papi SYSTEM INTERFACE ${PAPI_INCLUDE_DIRS})
        target_link_libraries(timemory-papi INTERFACE ${PAPI_LIBRARIES})
        target_compile_definitions(timemory-papi INTERFACE TIMEMORY_USE_PAPI)
        target_link_libraries(timemory-headers INTERFACE timemory-papi)
    else()
        set(TIMEMORY_USE_PAPI OFF)
        message(WARNING "PAPI package not found!")
    endif()

endif()


#----------------------------------------------------------------------------------------#
#
#                               Coverage
#
#----------------------------------------------------------------------------------------#

if(TIMEMORY_USE_COVERAGE)

    find_library(GCOV_LIBRARY gcov QUIET)

    if(GCOV_LIBRARY OR CMAKE_CXX_COMPILER_IS_GNU)
        add_interface_library(timemory-coverage)
    endif()

    if(GCOV_LIBRARY)
        target_link_libraries(timemory-coverage INTERFACE ${COVERAGE_LIBRARY})
    elseif(CMAKE_CXX_COMPILER_IS_GNU)
        target_link_libraries(timemory-coverage INTERFACE gcov)
    else()
        message(STATUS "GCov library not found. Disabling coverage...")
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
    find_package(CUDA)

    if("CUDA" IN_LIST LANGUAGES AND CUDA_FOUND)

        add_interface_library(timemory-cuda)

        target_compile_definitions(timemory-cuda INTERFACE TIMEMORY_USE_CUDA)
        target_include_directories(timemory-cuda INTERFACE ${CUDA_INCLUDE_DIRS}
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

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

        target_compile_options(timemory-cuda INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${CUDA_ARCH} --default-stream per-thread>)

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
                PATH_SUFFIXES   lib lib64 lib/nvidia lib64/nvidia nvidia lib/stubs lib64/stubs stubs)

            # if header and library found
            if(CUDA_cupti_INCLUDE_DIR AND CUDA_cupti_LIBRARY AND CUDA_cuda_LIBRARY)
                target_include_directories(timemory-cuda INTERFACE ${CUDA_cupti_INCLUDE_DIR})
                target_link_libraries(timemory-cuda INTERFACE ${CUDA_cupti_LIBRARY} ${CUDA_cuda_LIBRARY})
                target_compile_definitions(timemory-cuda INTERFACE TIMEMORY_USE_CUPTI)
            else()
                set(_MSG "Warning! Unable to find CUPTI. Missing variables:")
                foreach(_VAR CUDA_cupti_INCLUDE_DIR CUDA_cupti_LIBRARY CUDA_cuda_LIBRARY)
                    if(NOT ${_VAR})
                        add(_MSG ${_VAR})
                    endif()
                endforeach()
                set(_MSG "${_MSG}. Disabling TIMEMORY_USE_CUPTI...")
                message(STATUS "${_MSG}")
                set(TIMEMORY_USE_CUPTI OFF)
                unset(_MSG)
            endif()

            # clean-up
            unset(_CUDA_PATHS)
        endif()

        # timemory-headers provides timemory-cuda
        target_link_libraries(timemory-headers INTERFACE timemory-cuda)

    else()
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
    find_package(GPerfTools COMPONENTS profiler)

    if(GPerfTools_FOUND)
        add_interface_library(timemory-gperf)
        target_compile_definitions(timemory-gperf INTERFACE TIMEMORY_USE_GPERF)
        target_include_directories(timemory-gperf INTERFACE ${GPerfTools_INCLUDE_DIRS})
        target_link_libraries(timemory-gperf INTERFACE ${GPerfTools_LIBRARIES})
        target_link_libraries(timemory-headers INTERFACE timemory-gperf)
    else()
        set(TIMEMORY_USE_GPERF OFF)
        message(WARNING "GPerfTools package not found!")
    endif()

endif()


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

add_interface_library(timemory-cereal IMPORTED GLOBAL)
target_include_directories(timemory-cereal SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source/cereal/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ${DEV_WARNINGS} CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)

target_link_libraries(timemory-headers INTERFACE timemory-cereal)


#----------------------------------------------------------------------------------------#
#
#                               External variables
#
#----------------------------------------------------------------------------------------#

# including the directories
safe_remove_duplicates(EXTERNAL_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS})
safe_remove_duplicates(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES})
safe_remove_duplicates(PRIVATE_EXTERNAL_INCLUDE_DIRS ${PRIVATE_EXTERNAL_INCLUDE_DIRS})
safe_remove_duplicates(PRIVATE_EXTERNAL_LIBRARIES ${PRIVATE_EXTERNAL_LIBRARIES})

set(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES} PRIVATE ${PRIVATE_EXTERNAL_LIBRARIES})

list(APPEND ${PROJECT_NAME}_TARGET_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS})
list(APPEND ${PROJECT_NAME}_TARGET_LIBRARIES ${EXTERNAL_LIBRARIES})

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME development)
