#   Python configuration
#

# include guard
include_guard(DIRECTORY)

# Stops lookup as soon as a version satisfying version constraints is found.
set(Python3_FIND_STRATEGY "LOCATION" CACHE STRING
    "Stops lookup as soon as a version satisfying version constraints is found")

# virtual environment is used before any other standard paths to look-up for the interpreter
set(Python3_FIND_VIRTUALENV "FIRST" CACHE STRING
    "Virtual environment is used before any other standard paths")
set_property(CACHE Python3_FIND_VIRTUALENV PROPERTY STRINGS "FIRST;LAST;NEVER")

if(APPLE)
    set(Python3_FIND_FRAMEWORK "LAST" CACHE STRING
        "Order of preference between Apple-style and unix-style package components")
    set_property(CACHE Python3_FIND_FRAMEWORK PROPERTY STRINGS "FIRST;LAST;NEVER")
endif()

# PyPy does not support embedding the interpreter
set(Python3_FIND_IMPLEMENTATIONS "CPython" CACHE STRING
    "Different implementations which will be searched.")
set_property(CACHE Python3_FIND_IMPLEMENTATIONS PROPERTY STRINGS
    "CPython;IronPython;PyPy")

# variable is a 3-tuple specifying, in order, pydebug (d), pymalloc (m) and unicode (u)
# set(Python3_FIND_ABI "OFF" "OFF" "OFF" CACHE STRING
#    "variable is a 3-tuple specifying pydebug (d), pymalloc (m) and unicode (u)")

# Create CMake cache entries for the above artifact specification variables so that users
# can edit them interactively. This disables support for multiple version/component
# requirements.
set(Python3_ARTIFACTS_INTERACTIVE ON CACHE BOOL
    "Create CMake cache entries so that users can edit them interactively")

# if("${Python3_USE_STATIC_LIBS}" STREQUAL "ANY")
#    set(Python3_USE_STATIC_LIBS "OFF" CACHE STRING
#        "If ON, only static libs; if OFF, only shared libs; if ANY, shared then static")
#    set_property(CACHE Python3_USE_STATIC_LIBS PROPERTY STRINGS "ON;OFF;ANY")
# else()
#    unset(Python3_USE_STATIC_LIBS)
# endif()

foreach(_VAR FIND_STRATEGY FIND_VIRTUALENV FIND_FRAMEWORK FIND_IMPLEMENTATIONS ARTIFACTS_INTERACTIVE)
    if(DEFINED Python3_${_VAR})
        set(Python_${_VAR} "${Python3_${_VAR}}" CACHE STRING "Set via Python3_${_VAR} setting (timemory)")
        mark_as_advanced(Python_${_VAR})
        mark_as_advanced(Python3_${_VAR})
    endif()
endforeach()

# display version
add_feature(TIMEMORY_PYTHON_VERSION "Python version for timemory" DOC)

# search hint
if(PYTHON_ROOT_DIR AND NOT Python3_ROOT_DIR)
    set(Python3_ROOT_DIR ${PYTHON_ROOT_DIR})
endif()

# legacy specification of interpreter
if(PYTHON_EXECUTABLE AND NOT Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH
        "Path to Python3 interpreter")
endif()

# default python types to search for
set(Python_ADDITIONAL_VERSIONS "3.9;3.8;3.7;3.6" CACHE STRING 
    "Python versions supported by timemory")

# override types to search for
if(TIMEMORY_PYTHON_VERSION)
    set(Python_ADDITIONAL_VERSIONS ${TIMEMORY_PYTHON_VERSION} CACHE STRING
        "Python versions supported by timemory" FORCE)
elseif(PYBIND11_PYTHON_VERSION)
    set(Python_ADDITIONAL_VERSIONS ${PYBIND11_PYTHON_VERSION} CACHE STRING
        "Python versions supported by timemory")
endif()

# unset the version strings
if(_PYVERSION_LAST
        AND (TIMEMORY_PYTHON_VERSION VERSION_LESS _PYVERSION_LAST
            OR TIMEMORY_PYTHON_VERSION VERSION_GREATER _PYVERSION_LAST))
    unset(TIMEMORY_PYTHON_VERSION CACHE)
    unset(PYBIND11_PYTHON_VERSION CACHE)
    unset(CMAKE_INSTALL_PYTHONDIR CACHE)
endif()

# if TIMEMORY_PYTHON_VERSION specified, set to desired python version
set(_PYVERSION ${TIMEMORY_PYTHON_VERSION})

# if TIMEMORY_PYTHON_VERSION is not set but PYBIND11_PYTHON_VERSION is
if("${_PYVERSION}" STREQUAL "" AND PYBIND11_PYTHON_VERSION)
    set(_PYVERSION ${PYBIND11_PYTHON_VERSION})
endif()

# basically just used to get Python3_SITEARCH for installation
find_package(Python3 ${_PYVERSION} MODULE ${TIMEMORY_FIND_REQUIREMENT}
    COMPONENTS Interpreter Development
)

# executable
set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH
    "Set via Python3_EXECUTABLE (timemory)" FORCE)
# includes
set(PYTHON_INCLUDE_DIR "${Python3_INCLUDE_DIRS}" CACHE PATH
    "Set via Python3_INCLUDE_DIR (timemory)" FORCE)
set(PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}" CACHE PATH
    "Set via Python3_INCLUDE_DIRS (timemory)" FORCE)
# libraries
set(PYTHON_LIBRARY_DEBUG "${Python3_LIBRARY_DEBUG}" CACHE FILEPATH
    "Set via Python3_LIBRARY_DEBUG (timemory)" FORCE)
set(PYTHON_LIBRARY_RELEASE "${Python3_LIBRARY_RELEASE}" CACHE FILEPATH
    "Set via Python3_LIBRARY_DEBUG (timemory)" FORCE)
if(Python3_LIBRARY_RELEASE)
    set(PYTHON_LIBRARY "${Python3_LIBRARY_RELEASE}" CACHE FILEPATH
        "Set via Python3_LIBRARY (timemory)" FORCE)
    set(PYTHON_LIBRARIES "${Python3_LIBRARY_RELEASE}" CACHE FILEPATH
        "Set via Python3_LIBRARIES (timemory)" FORCE)
else(Python3_LIBRARY_DEBUG)
    set(PYTHON_LIBRARY "${Python3_LIBRARY_DEBUG}" CACHE FILEPATH
        "Set via Python3_LIBRARY (timemory)" FORCE)
    set(PYTHON_LIBRARIES "${Python3_LIBRARY_DEBUG}" CACHE FILEPATH
        "Set via Python3_LIBRARIES (timemory)" FORCE)
endif()
set(PYTHON_LIBRARY_DIRS "${Python3_LIBRARY_DIRS}" CACHE PATH
    "Set via Python3_LIBRARY_DIRS (timemory)" FORCE)
set(PYTHON_LINK_OPTIONS "${Python3_LINK_OPTIONS}" CACHE STRING
    "Set via Python3_LINK_OPTIONS (timemory)" FORCE)

# module
set(PYTHON_MODULE_EXTENSION "${Python3_MODULE_EXTENSION}" CACHE STRING
    "Set via Python3_MODULE_EXTENSION (timemory)" FORCE)
set(PYTHON_MODULE_PREFIX "${Python3_MODULE_PREFIX}" CACHE STRING
    "Set via Python3_MODULE_PREFIX (timemory)" FORCE)

# version
set(PYTHON_VERSION "${Python3_VERSION}" CACHE STRING
    "Set via Python3_VERSION (timemory)" FORCE)
set(PYTHON_VERSION_MAJOR "${Python3_VERSION_MAJOR}" CACHE STRING
    "Set via Python3_VERSION_MAJOR (timemory)" FORCE)
set(PYTHON_VERSION_MINOR "${Python3_VERSION_MINOR}" CACHE STRING
    "Set via Python3_VERSION_MINOR (timemory)" FORCE)

# find_package
set(PythonInterp_FOUND ${Python3_Interpreter_FOUND})
set(PythonLibs_FOUND ${Python3_Development_FOUND})

# set TIMEMORY_PYTHON_VERSION if we have the python version
if(PYTHON_VERSION_STRING)
    set(TIMEMORY_PYTHON_VERSION "${PYTHON_VERSION_STRING}" CACHE STRING
        "Python version for timemory")
endif()

# if either not found, disable
if(NOT Python3_FOUND)
    set(TIMEMORY_USE_PYTHON OFF)
    set(TIMEMORY_BUILD_PYTHON OFF)
    inform_empty_interface(timemory-plotting "Python plotting from C++")
else()
    set(TIMEMORY_PYTHON_VERSION "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}"
        CACHE STRING "Python version for timemory")
    add_feature(PYTHON_EXECUTABLE "Python executable")
    add_cmake_defines(TIMEMORY_PYTHON_PLOTTER QUOTE VALUE DEFAULT)
    set(TIMEMORY_PYTHON_PLOTTER "${PYTHON_EXECUTABLE}")
    timemory_target_compile_definitions(timemory-plotting INTERFACE TIMEMORY_USE_PLOTTING)
    target_link_libraries(timemory-headers INTERFACE timemory-plotting)
endif()

# C++ standard
if(NOT MSVC)
    if(NOT "${PYBIND11_CPP_STANDARD}" STREQUAL "-std=c++${CMAKE_CXX_STANDARD}")
        set(PYBIND11_CPP_STANDARD -std=c++${CMAKE_CXX_STANDARD}
            CACHE STRING "PyBind11 CXX standard" FORCE)
    endif()
else()
    if(NOT "${PYBIND11_CPP_STANDARD}" STREQUAL "/std:c++${CMAKE_CXX_STANDARD}")
        set(PYBIND11_CPP_STANDARD /std:c++${CMAKE_CXX_STANDARD}
            CACHE STRING "PyBind11 CXX standard" FORCE)
    endif()
endif()

option(PYBIND11_INSTALL "Enable Pybind11 installation" OFF)

if(TIMEMORY_BUILD_PYTHON AND NOT TARGET pybind11)
    # checkout PyBind11 if not checked out
    checkout_git_submodule(RECURSIVE
        RELATIVE_PATH external/pybind11
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/pybind11.git
        REPO_BRANCH master)

    # add PyBind11 to project
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

add_feature(PYBIND11_CPP_STANDARD "PyBind11 C++ standard")
add_feature(PYBIND11_PYTHON_VERSION "PyBind11 Python version")

if(NOT "${TIMEMORY_PYTHON_VERSION}" MATCHES "${PYBIND11_PYTHON_VERSION}*")
    message(STATUS "TIMEMORY_PYTHON_VERSION is set to ${TIMEMORY_PYTHON_VERSION}")
    message(STATUS "PYBIND11_PYTHON_VERSION is set to ${PYBIND11_PYTHON_VERSION}")
    message(FATAL_ERROR "Mismatched 'TIMEMORY_PYTHON_VERSION' and 'PYBIND11_PYTHON_VERSION'")
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE}
    -c "import time ; print('{} {}'.format(time.ctime(), time.tzname[0]))"
    OUTPUT_VARIABLE TIMEMORY_INSTALL_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

string(REPLACE "  " " " TIMEMORY_INSTALL_DATE "${TIMEMORY_INSTALL_DATE}")

if(SKBUILD)
    set(CMAKE_INSTALL_PYTHONDIR ${CMAKE_INSTALL_PREFIX}
        CACHE PATH "Installation directory for python")
elseif(SPACK_BUILD)
    set(CMAKE_INSTALL_PYTHONDIR
        lib/python${PYBIND11_PYTHON_VERSION}/site-packages
        CACHE PATH "Installation directory for python")
else()
    string(REPLACE "\\" "/" Python3_SITEARCH "${Python3_SITEARCH}")
    set(CMAKE_INSTALL_PYTHONDIR ${Python3_SITEARCH})
    add_feature(Python3_SITEARCH "site-packages directory of python installation")
    set(_REMOVE OFF)
    # make the directory if it doesn't exist
    if(NOT EXISTS ${Python3_SITEARCH}/timemory)
        set(_REMOVE ON)
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E make_directory ${Python3_SITEARCH}/timemory
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            ERROR_QUIET)
    endif()
    # figure out if we can install to Python3_SITEARCH
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E touch ${Python3_SITEARCH}/timemory/__init__.py
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        ERROR_VARIABLE ERR_MSG
        RESULT_VARIABLE ERR_CODE)
    # remove the directory if we created it
    if(_REMOVE)
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${Python3_SITEARCH}/timemory
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            ERROR_QUIET)
    endif()
    # check the error code of the touch command
    if(ERR_CODE)
        # get the python directory name, e.g. 'python3.6' from
        # '/opt/local/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6'
        get_filename_component(PYDIR "${Python3_STDLIB}" NAME)
        ADD_FEATURE(Python3_STDLIB "standard-library directory of python installation")
        # Should not be CMAKE_INSTALL_LIBDIR! Python won't look in a lib64 folder
        set(CMAKE_INSTALL_PYTHONDIR lib/${PYDIR}/site-packages)
    endif()
endif()

if(TIMEMORY_BUILD_PYTHON OR pybind11_FOUND)
    timemory_target_compile_definitions(timemory-python INTERFACE TIMEMORY_USE_PYTHON)
    target_link_libraries(timemory-python INTERFACE ${PYTHON_LIBRARIES})
    target_include_directories(timemory-python SYSTEM INTERFACE
        ${PYTHON_INCLUDE_DIRS}
        ${PYBIND11_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${PYBIND11_INCLUDE_DIR}>)
endif()

if(APPLE)
    if(CMAKE_VERSION VERSION_LESS 3.18)
        target_link_libraries(timemory-python INTERFACE "-undefined dynamic_lookup")
    else()
        target_link_libraries(timemory-python INTERFACE
            "$<$<LINK_LANGUAGE:CXX>:-undefined dynamic_lookup>")
    endif()
endif()

if(WIN32)
    # Windows produces:
    #
    #   CMake Warning (dev) at tests/test-python-install-import.cmake:3 (SET):
    #   Syntax error in cmake code at
    #     C:/projects/timemory/build-timemory/tests/test-python-install-import.cmake:3
    #   when parsing string
    #     C:\Python36-x64\Lib\site-packages
    #   Invalid escape sequence \P
    string(REPLACE "\\" "/" INSTALL_PYTHONDIR "${CMAKE_INSTALL_PYTHONDIR}")
else()
    set(INSTALL_PYTHONDIR "${CMAKE_INSTALL_PYTHONDIR}")
endif()

configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/test-python-install-import.cmake.in
    ${PROJECT_BINARY_DIR}/tests/test-python-install-import.cmake @ONLY)
unset(INSTALL_PYTHONDIR)

add_feature(CMAKE_INSTALL_PYTHONDIR "Installation prefix of the python bindings")

set(_PYVERSION_LAST "${TIMEMORY_PYTHON_VERSION}" CACHE INTERNAL "Last version" FORCE)

find_package(PythonInterp ${TIMEMORY_PYTHON_VERSION} EXACT REQUIRED)
find_package(PythonLibs ${TIMEMORY_PYTHON_VERSION} EXACT REQUIRED)
# find_package(PythonExtensions REQUIRED)

if("${PYTHON_MODULE_EXTENSION}" STREQUAL "")
    message(WARNING "Python module extension is empty!")
endif()
