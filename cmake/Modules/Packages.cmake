################################################################################
#
#        MPI
#
################################################################################
add_option(USE_MPI "Enable MPI usage" ON)

if(USE_MPI)

    if(WIN32)
        if(EXISTS "C:/Program\ Files\ (x86)/Microsoft\ SDKs/MPI")
            list(APPEND CMAKE_PREFIX_PATH "C:/Program\ Files\ (x86)/Microsoft\ SDKs/MPI")
        endif(EXISTS "C:/Program\ Files\ (x86)/Microsoft\ SDKs/MPI")

        if(EXISTS "C:/Program\ Files/Microsoft\ SDKs/MPI")
            list(APPEND CMAKE_PREFIX_PATH "C:/Program\ Files/Microsoft\ SDKs/MPI")
        endif(EXISTS "C:/Program\ Files/Microsoft\ SDKs/MPI")
    endif(WIN32)

    # MPI C compiler from environment
    set(_ENV MPICC)
    if(NOT DEFINED MPI_C_COMPILER AND NOT "$ENV{${_ENV}}" STREQUAL "")
        message(STATUS "Setting MPI C compiler to: $ENV{${_ENV}}")
        set(MPI_C_COMPILER $ENV{${_ENV}} CACHE FILEPATH "MPI C compiler")
    endif(NOT DEFINED MPI_C_COMPILER AND NOT "$ENV{${_ENV}}" STREQUAL "")

    # MPI C++ compiler from environment
    set(_ENV MPICC)
    if(NOT DEFINED MPI_CXX_COMPILER AND NOT "$ENV{${_ENV}}" STREQUAL "")
        message(STATUS "Setting MPI C++ compiler to: $ENV{${_ENV}}")
        set(MPI_CXX_COMPILER $ENV{${_ENV}} CACHE FILEPATH "MPI C++ compiler")
    endif(NOT DEFINED MPI_CXX_COMPILER AND NOT "$ENV{${_ENV}}" STREQUAL "")

    unset(_ENV)

    find_package(MPI)

    set(MPI_LIBRARIES )
    if(MPI_FOUND)

        # Add the MPI-specific compiler and linker flags
        add(CMAKE_CXX_FLAGS  "${MPI_CXX_COMPILE_FLAGS}")
        add(CMAKE_EXE_LINKER_FLAGS "${MPI_CXX_LINK_FLAGS}")
        list(APPEND EXTERNAL_INCLUDE_DIRS
            ${MPI_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

        foreach(_TYPE C_LIBRARIES CXX_LIBRARIES EXTRA_LIBRARY)
            set(_TYPE MPI_${_TYPE})
            if(${_TYPE})
                list(APPEND MPI_LIBRARIES ${${_TYPE}})
            endif(${_TYPE})
        endforeach(_TYPE C_LIBRARIES CXX_LIBRARIES EXTRA_LIBRARY)

        list(APPEND EXTERNAL_LIBRARIES ${MPI_LIBRARIES})

        if(WIN32)
            add_definitions(/DTIMEMORY_USE_MPI)
        else(WIN32)
            add_definitions(-DTIMEMORY_USE_MPI)
        endif(WIN32)

        if(NOT MPIEXEC_EXECUTABLE AND MPIEXEC)
          set(MPIEXEC_EXECUTABLE ${MPIEXEC} CACHE FILEPATH "MPI executable")
        endif(NOT MPIEXEC_EXECUTABLE AND MPIEXEC)

        if(NOT MPIEXEC_EXECUTABLE AND MPI_EXECUTABLE)
          set(MPIEXEC_EXECUTABLE ${MPI_EXECUTABLE} CACHE FILEPATH "MPI executable")
        endif(NOT MPIEXEC_EXECUTABLE AND MPI_EXECUTABLE)

    else(MPI_FOUND)

        message(WARNING "MPI not found. Proceeding without MPI")
        remove_definitions(-DTIMEMORY_USE_MPI)

    endif(MPI_FOUND)

endif(USE_MPI)


################################################################################
#
#        Threading
#
################################################################################

if(NOT WIN32)
    set(CMAKE_THREAD_PREFER_PTHREAD ON)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
endif(NOT WIN32)

find_package(Threads)

if(THREADS_FOUND)
    list(APPEND EXTERNAL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
endif(THREADS_FOUND)


################################################################################
#
#        PyBind11
#
################################################################################

if(NOT SETUP_PY OR PYTHON_DEVELOPER_INSTALL)
    add_dependent_option(PYBIND11_INSTALL "PyBind11 installation" OFF
        "PYTHON_DEVELOPER_INSTALL" ON)
else(NOT SETUP_PY OR PYTHON_DEVELOPER_INSTALL)
    set(PYBIND11_INSTALL OFF CACHE BOOL "Don't install Pybind11" FORCE)
endif(NOT SETUP_PY OR PYTHON_DEVELOPER_INSTALL)

# checkout PyBind11 if not checked out
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/pybind11/CMakeLists.txt")
    find_package(Git REQUIRED)
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE RET)
    if(RET GREATER 0)
        message(FATAL_ERROR "Failure checking out submodules")
    endif(RET GREATER 0)
endif(NOT EXISTS "${CMAKE_SOURCE_DIR}/pybind11/CMakeLists.txt")

# make sure pybind11 gets installed in same place as TiMemory
if(PYBIND11_INSTALL AND PYTHON_DEVELOPER_INSTALL)
    set(PYBIND11_CMAKECONFIG_INSTALL_DIR
        "${TIMEMORY_INSTALL_DATAROOTDIR}/cmake/pybind11"
        CACHE STRING "install path for pybind11Config.cmake" FORCE)
    set(CMAKE_INSTALL_INCLUDEDIR ${TIMEMORY_INSTALL_INCLUDEDIR}
        CACHE PATH "Include file installation path" FORCE)
endif(PYBIND11_INSTALL AND PYTHON_DEVELOPER_INSTALL)

# C++ standard
set(PYBIND11_CPP_STANDARD -std=c++${CMAKE_CXX_STANDARD}
    CACHE STRING "PyBind11 CXX standard" FORCE)

# add PyBind11 to project
add_subdirectory(pybind11)

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

# including the directories
safe_remove_duplicates(EXTERNAL_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS})
safe_remove_duplicates(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES})
foreach(_DIR ${EXTERNAL_INCLUDE_DIRS})
    include_directories(SYSTEM ${_DIR})
endforeach(_DIR ${EXTERNAL_INCLUDE_DIRS})
