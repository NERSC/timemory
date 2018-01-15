cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)

project(TiMemory LANGUAGES C CXX VERSION 1.0.0)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules
    ${CMAKE_MODULE_PATH})
include(Compilers)

################################################################################
#
# The license for TiMemory allows it to be included directly in the source
# tree of another project
# By setting TIMEMORY_NAMESPACE, the project can directly
# use TiMemory in its own namespace, e.g.:
# TIMEMORY_NAMESPACE=toast
#   tim::util::timing_manager --> toast::util::timing_manager
#
################################################################################

set(TIMEMORY_NAMESPACE tim CACHE STRING "Top-level namespace for ${PROJECT_NAME}")


################################################################################
#
#        MPI
#
################################################################################

find_package(MPI REQUIRED)

# Add the MPI-specific compiler and linker flags
add(CMAKE_CXX_FLAGS  "${MPI_CXX_COMPILE_FLAGS}")
add(CMAKE_EXE_LINKER_FLAGS "${MPI_CXX_LINK_FLAGS}")
include_directories(${MPI_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

set(MPI_LIBRARIES )
foreach(_TYPE C_LIBRARIES CXX_LIBRARIES EXTRA_LIBRARY)
    set(_TYPE MPI_${_TYPE})
    if(${_TYPE})
        list(APPEND MPI_LIBRARIES ${${_TYPE}})
    endif(${_TYPE})
endforeach(_TYPE C_LIBRARIES CXX_LIBRARIES EXTRA_LIBRARY)
list(APPEND EXTERNAL_LIBRARIES ${MPI_LIBRARIES})


################################################################################
#
#        Threading
#
################################################################################

set(CMAKE_THREAD_PREFER_PTHREADS ON)
find_package(Threads)
list(APPEND EXTERNAL_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})


################################################################################
#
#        Miscellaneous
#
################################################################################

include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ STL standard")
option(CMAKE_CXX_STANDARD_REQUIRED "Require C++ standard" ON)
option(CMAKE_CXX_EXTENSIONS "Build with CXX extensions (e.g. gnu++11)" OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_MESSAGE LAZY)

foreach(_TYPE ARCHIVE LIBRARY RUNTIME)
    set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
endforeach(_TYPE ARCHIVE LIBRARY RUNTIME)

set(CMAKE_INSTALL_CMAKEDIR  ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
    CACHE PATH "Installation directory for CMake package config files")

set(INCLUDE_INSTALL_DIR     ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR         ${CMAKE_INSTALL_LIBDIR})
set(LIBNAME                 timemory)

# without this, get the following warning in compilation with "-g -DDEBUG":
#   ld: warning: direct access in function 'XXX' from file
#       'XXX' to global weak symbol 'XXX' from file
#       'XXX' means the weak symbol cannot be overridden at runtime.
#       This was likely caused by different translation units being
#       compiled with different visibility settings.
add(CMAKE_CXX_FLAGS "-W -Wall -Wextra")
add(CMAKE_CXX_FLAGS "-Wno-implicit-fallthrough -fvisibility=hidden")
add(CMAKE_CXX_FLAGS "-Wno-shadow-field-in-constructor-modified")
add(CMAKE_CXX_FLAGS "-Wno-unused-private-field")
add_cxx_flags(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


################################################################################
#
#        PyBind11
#
################################################################################

set(PYBIND11_CPP_STANDARD -std=c++${CMAKE_CXX_STANDARD}
    CACHE STRING "PyBind11 CXX standard")
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

add_definitions(-DNAME_TIM=${TIMEMORY_NAMESPACE})

add_subdirectory(source)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/Templates/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
    PATH_VARS
        INCLUDE_INSTALL_DIR
        LIB_INSTALL_DIR
        PYTHON_INSTALL_DIR)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_CMAKEDIR})

foreach(_FILE simple.py timemory_plot.py)
    configure_file(${PROJECT_SOURCE_DIR}/examples/${_FILE}
        ${PROJECT_BINARY_DIR}/${_FILE} COPYONLY)
endforeach(_FILE simple.py timemory_plot.py)
