cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)

project(TiMemory LANGUAGES C CXX VERSION 1.1.0)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules
    ${CMAKE_MODULE_PATH})

include(Compilers)
include(MacroUtilities)

if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set(SUBPROJECT OFF)
else("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set(SUBPROJECT ON)
endif("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")

add_option(SETUP_PY "Python build from setup.py" OFF NO_FEATURE)

################################################################################
#
# The license for TiMemory allows it to be included directly in the source
# tree of another project
# By setting TIMEMORY_NAMESPACE, the project can directly
# use TiMemory in its own namespace, e.g.:
# TIMEMORY_NAMESPACE=toast
#   tim::timing_manager --> toast::timing_manager
#
################################################################################

set(TIMEMORY_NAMESPACE tim CACHE STRING "Top-level namespace for ${PROJECT_NAME}")
if(NOT SUBPROJECT)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
        set(CMAKE_BUILD_TYPE Release)
    endif()
    add_feature(CMAKE_BUILD_TYPE "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)")
    add_feature(CMAKE_INSTALL_PREFIX "Installation prefix")
    add_feature(CMAKE_CXX_FLAGS "C++ compiler flags")
endif(NOT SUBPROJECT)
add_feature(TIMEMORY_NAMESPACE "Top-level namespace for ${PROJECT_NAME}")


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
find_package(Threads REQUIRED)
list(APPEND EXTERNAL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

################################################################################
#
#        Miscellaneous
#
################################################################################

include(GNUInstallDirs)
add_option(TIMEMORY_EXCEPTIONS "Signal handler throws exceptions (default: exit)" OFF)
if(NOT SUBPROJECT)
    if(WIN32)
        set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ STL standard")
    else(WIN32)
        set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ STL standard")
    endif(WIN32)
    add_feature(CMAKE_CXX_STANDARD "C++11 STL standard")
    add_option(CMAKE_CXX_STANDARD_REQUIRED "Require C++ standard" ON)
    add_option(CMAKE_CXX_EXTENSIONS "Build with CXX extensions (e.g. gnu++11)" OFF)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_INSTALL_MESSAGE LAZY)
    set(SANITIZE_TYPE leak CACHE STRING "-fsantitize=<TYPE>")
    add_option(ENABLE_SANITIZE "Enable -fsanitize flag (=${SANITIZE_TYPE})" OFF)
endif(NOT SUBPROJECT)

# set the output directory (critical on Windows
foreach(_TYPE ARCHIVE LIBRARY RUNTIME)
    # if TIMEMORY_OUTPUT_DIR is not defined, set to CMAKE_BINARY_DIR
    if(NOT DEFINED TIMEMORY_OUTPUT_DIR OR "${TIMEMORY_OUTPUT_DIR}" STREQUAL "")
        set(TIMEMORY_OUTPUT_DIR ${CMAKE_BINARY_DIR})
    endif(NOT DEFINED TIMEMORY_OUTPUT_DIR OR "${TIMEMORY_OUTPUT_DIR}" STREQUAL "")
    # set the CMAKE_{ARCHIVE,LIBRARY,RUNTIME}_OUTPUT_DIRECTORY variables
    if(WIN32)
        # on Windows, separate types into different directories
        string(TOLOWER "${_TYPE}" _LTYPE)
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${TIMEMORY_OUTPUT_DIR}/outputs/${_LTYPE})
    else(WIN32)
        # on UNIX, just set to same directory
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${TIMEMORY_OUTPUT_DIR})
    endif(WIN32)
endforeach(_TYPE ARCHIVE LIBRARY RUNTIME)

# cmake installation folder
set(PROJECT_INSTALL_CMAKEDIR  ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
    CACHE PATH "Installation directory for CMake package config files")

# used by configure_package_*
set(INCLUDE_INSTALL_DIR     ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR         ${CMAKE_INSTALL_LIBDIR})
set(LIBNAME                 timemory)

# set the compiler flags if not on Windows
if(NOT SUBPROJECT AND NOT WIN32)

    add(CMAKE_CXX_FLAGS "-W -Wall -Wextra -faligned-new ${CXXFLAGS} $ENV{CXXFLAGS}")
    add(CMAKE_CXX_FLAGS "-Wno-unused-parameter")
    add(CMAKE_CXX_FLAGS "-Wunused-but-set-parameter -Wno-unused-variable")

    if(NOT CMAKE_CXX_COMPILER_IS_INTEL)
        add(CMAKE_CXX_FLAGS "-Wno-unknown-warning-option")
        add(CMAKE_CXX_FLAGS "-Wno-implicit-fallthrough")
        add(CMAKE_CXX_FLAGS "-Wno-shadow-field-in-constructor-modified")
        add(CMAKE_CXX_FLAGS "-Wno-exceptions")
        add(CMAKE_CXX_FLAGS "-Wno-unknown-warning-option")
        add(CMAKE_CXX_FLAGS "-Wno-unused-private-field")
    endif(NOT CMAKE_CXX_COMPILER_IS_INTEL)

    add(CMAKE_C_FLAGS "-W -Wall -Wextra -faligned-new ${CFLAGS} $ENV{CFLAGS}")
    add(CMAKE_C_FLAGS "-Wno-unused-parameter")
    add(CMAKE_C_FLAGS "-Wunused-but-set-parameter -Wno-unused-variable")

    if(NOT CMAKE_C_COMPILER_IS_INTEL)
        add(CMAKE_C_FLAGS "-Wno-unknown-warning-option")
        add(CMAKE_C_FLAGS "-Wno-implicit-fallthrough")
        add(CMAKE_C_FLAGS "-Wno-shadow-field-in-constructor-modified")
        add(CMAKE_C_FLAGS "-Wno-exceptions")
        add(CMAKE_C_FLAGS "-Wno-unknown-warning-option")
        add(CMAKE_C_FLAGS "-Wno-unused-private-field")
    endif(NOT CMAKE_C_COMPILER_IS_INTEL)

    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

        add(CMAKE_C_FLAGS "-DDEBUG")
        add(CMAKE_CXX_FLAGS "-DDEBUG")

    else("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

        if(CMAKE_CXX_COMPILER_IS_INTEL)
            add(CMAKE_CXX_FLAGS "-xHOST")
        else(CMAKE_CXX_COMPILER_IS_INTEL)
            add(CMAKE_CXX_FLAGS "-march=native")
        endif(CMAKE_CXX_COMPILER_IS_INTEL)

        if(CMAKE_C_COMPILER_IS_INTEL)
            add(CMAKE_C_FLAGS "-xHOST")
        else(CMAKE_C_COMPILER_IS_INTEL)
            add(CMAKE_C_FLAGS "-march=native")
        endif(CMAKE_C_COMPILER_IS_INTEL)

    endif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

    if(UNIX)
        add(CMAKE_CXX_FLAGS "-pthread")
        add(CMAKE_C_FLAGS "-pthread")
    endif(UNIX)

    if(ENABLE_SANITIZE)
        add_subfeature(ENABLE_SANITIZE SANITIZE_TYPE "Sanitizer type")
        add(CMAKE_CXX_FLAGS "-fsanitize=${SANITIZE_TYPE}")
    endif(ENABLE_SANITIZE)

    add_c_flags(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    add_cxx_flags(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

endif(NOT SUBPROJECT AND NOT WIN32)

if(TIMEMORY_EXCEPTIONS)
    if(WIN32)
        add_definitions(/DTIMEMORY_EXCEPTIONS)
    else(WIN32)
        add_definitions(-DTIMEMORY_EXCEPTIONS)
    endif(WIN32)
endif()


################################################################################
#
#        PyBind11
#
################################################################################

add_option(PYBIND11_INSTALL "PyBind11 installation" ON)
set(PYBIND11_CPP_STANDARD -std=c++${CMAKE_CXX_STANDARD}
    CACHE STRING "PyBind11 CXX standard" FORCE)
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/pybind11/CMakeLists.txt")
    find_package(Git REQUIRED)
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE RET)
    if(RET GREATER 0)
        message(FATAL_ERROR "Failure checking out submodules")
    endif(RET GREATER 0)
endif(NOT EXISTS "${CMAKE_SOURCE_DIR}/pybind11/CMakeLists.txt")
add_subdirectory(pybind11)

if(NOT PYBIND11_PYTHON_VERSION)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} --version
        OUTPUT_VARIABLE PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX REPLACE "[ A-Za-z]" "" PYTHON_VERSION "${PYTHON_VERSION}")
    string(REGEX REPLACE "\.([0-9]+)$" "" PYTHON_VERSION "${PYTHON_VERSION}")
    set(PYBIND11_PYTHON_VERSION "${PYTHON_VERSION}"
        CACHE STRING "Python version" FORCE)
endif(NOT PYBIND11_PYTHON_VERSION)
set(CMAKE_INSTALL_PYTHONDIR
    ${CMAKE_INSTALL_LIBDIR}/python${PYBIND11_PYTHON_VERSION}/site-packages
    CACHE PATH "Installation directory for python")
set(CMAKE_INSTALL_FULL_PYTHONDIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_PYTHONDIR})
set(PYTHON_INSTALL_DIR      ${CMAKE_INSTALL_PYTHONDIR})


################################################################################
#
#        TiMemory
#
################################################################################

safe_remove_duplicates(EXTERNAL_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS})
safe_remove_duplicates(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES})

foreach(_DIR ${EXTERNAL_INCLUDE_DIRS})
    include_directories(SYSTEM ${_DIR})
endforeach(_DIR ${EXTERNAL_INCLUDE_DIRS})

if(SETUP_PY)
    set(CMAKE_INSTALL_PYTHONDIR ${CMAKE_INSTALL_PREFIX} CACHE PATH
        "Installation prefix of python" FORCE)
endif(SETUP_PY)

if(WIN32)
    add_definitions(/DNAME_TIM=${TIMEMORY_NAMESPACE})
else(WIN32)
    add_definitions(-DNAME_TIM=${TIMEMORY_NAMESPACE})
endif(WIN32)

add_subdirectory(source)

add_subdirectory(examples)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/Templates/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${PROJECT_INSTALL_CMAKEDIR}
    PATH_VARS
        INCLUDE_INSTALL_DIR
        LIB_INSTALL_DIR
        PYTHON_INSTALL_DIR)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${PROJECT_INSTALL_CMAKEDIR})

set(TEST_FILES timemory_test.py simple_test.py nested_test.py array_test.py)
foreach(_FILE ${TEST_FILES})
    configure_file(${PROJECT_SOURCE_DIR}/tests/${_FILE}
        ${PROJECT_BINARY_DIR}/${_FILE} COPYONLY)
endforeach(_FILE ${TEST_FILES})

if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/Modules/Testing.cmake")
    include(Testing)
endif(EXISTS "${CMAKE_SOURCE_DIR}/cmake/Modules/Testing.cmake")


if(NOT SUBPROJECT)
    # documentation
    set_property(GLOBAL APPEND PROPERTY BUILDTREE_INCLUDE_DIRS
        ${PROJECT_SOURCE_DIR}/source
        ${PROJECT_SOURCE_DIR}/python
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/examples)
    
    set(EXCLUDE_LIST ${PROJECT_SOURCE_DIR}/source/cereal)
    include(Documentation)
    
    if(DOXYGEN_DOCS)
        SET(CMAKE_INSTALL_MESSAGE NEVER)
        Generate_Documentation(Doxyfile.${PROJECT_NAME})
        SET(CMAKE_INSTALL_MESSAGE LAZY)
    endif()

    print_features()
endif(NOT SUBPROJECT)
