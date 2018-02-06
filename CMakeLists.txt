cmake_minimum_required(VERSION 2.8.12)

if(CMAKE_VERSION VERSION_LESS 3.1.3)
    set(GOOD_CMAKE OFF CACHE BOOL "Old version of CMake not support CMAKE_CXX_STANDARD" FORCE)
else(CMAKE_VERSION VERSION_LESS 3.1.3)
    set(GOOD_CMAKE ON CACHE BOOL "Old version of CMake not support CMAKE_CXX_STANDARD" FORCE)
endif(CMAKE_VERSION VERSION_LESS 3.1.3)

set(TIMEMORY_VERSION 1.1.8 CACHE STRING "Version of TiMemory project" FORCE)
mark_as_advanced(TIMEMORY_VERSION)

if(GOOD_CMAKE)
    cmake_policy(SET CMP0048 NEW)
    cmake_policy(SET CMP0042 NEW)
    project(TiMemory LANGUAGES C CXX VERSION ${TIMEMORY_VERSION})
else(GOOD_CMAKE)
    project(TiMemory C CXX)
    set(PROJECT_VERSION ${TIMEMORY_VERSION})
    string(REPLACE "." ";" "${PROJECT_VERSION}" _VERSION_LIST)
    list(GET _VERSION_LIST 0 PROJECT_VERSION_MAJOR)
    list(GET _VERSION_LIST 1 PROJECT_VERSION_MINOR)
    list(GET _VERSION_LIST 2 PROJECT_VERSION_PATCH)
endif(GOOD_CMAKE)

foreach(_TYPE MAJOR MINOR PATCH)
    set(TIMEMORY_VERSION_${_TYPE} ${PROJECT_VERSION_${_TYPE}} CACHE STRING "Timemory ${_TYPE} version" FORCE)
    mark_as_advanced(TIMEMORY_VERSION_${_TYPE})
endforeach(_TYPE MAJOR MINOR PATCH)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules
    ${CMAKE_MODULE_PATH})

if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set(SUBPROJECT OFF)
else("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    set(SUBPROJECT ON)
endif("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")

################################################################################
#
# The license for TiMemory allows it to be included directly in the source
# tree of another project
# By setting TIMEMORY_NAMESPACE, the project can directly
# use TiMemory in its own namespace, e.g.:
# TIMEMORY_NAMESPACE=mad
#   tim::timing_manager --> mad::timing_manager
#
################################################################################

include(Compilers)
include(MacroUtilities)

add_option(SETUP_PY "Python build from setup.py" OFF NO_FEATURE)

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

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(WIN32)
    set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ STL standard")
else(WIN32)
    set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ STL standard")
endif(WIN32)
add_feature(CMAKE_CXX_STANDARD "C++11 STL standard")

include(BuildSettings)
include(Packages)

set(CMAKE_INSTALL_PYTHONDIR
    ${CMAKE_INSTALL_LIBDIR}/python${PYBIND11_PYTHON_VERSION}/site-packages/timemory
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
    set(CMAKE_INSTALL_PYTHONDIR ${CMAKE_INSTALL_PREFIX}/timemory CACHE PATH
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

if(NOT SETUP_PY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${PROJECT_INSTALL_CMAKEDIR})
endif(NOT SETUP_PY)

set(TEST_FILES timemory_test.py simple_test.py nested_test.py array_test.py __init__.py tests.py)
foreach(_FILE ${TEST_FILES})
    # only copy *_test.py files to binary directory
    if("${_FILE}" MATCHES "_test.py")
        configure_file(${PROJECT_SOURCE_DIR}/timemory/tests/${_FILE}
            ${PROJECT_BINARY_DIR}/${_FILE} COPYONLY)
    endif("${_FILE}" MATCHES "_test.py")

    # copy to binary: timemory/tests/${_FILE}
    configure_file(${PROJECT_SOURCE_DIR}/timemory/tests/${_FILE}
        ${PROJECT_BINARY_DIR}/timemory/tests/${_FILE} COPYONLY)

    # install them though
    install(FILES ${PROJECT_BINARY_DIR}/timemory/tests/${_FILE}
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/tests)

endforeach(_FILE ${TEST_FILES})

if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/Modules/Testing.cmake")
    include(Testing)
endif(EXISTS "${CMAKE_SOURCE_DIR}/cmake/Modules/Testing.cmake")


if(NOT SUBPROJECT)
    # documentation
    set_property(GLOBAL APPEND PROPERTY BUILDTREE_INCLUDE_DIRS
        ${PROJECT_SOURCE_DIR}/source
        ${PROJECT_SOURCE_DIR}/timemory
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
