# include guard
include_guard(DIRECTORY)

##########################################################################################
#
#        timemory Package installation
#
##########################################################################################


include(CMakePackageConfigHelpers)

set(PYTHON_INSTALL_DIR      ${TIMEMORY_CONFIG_PYTHONDIR})
set(INCLUDE_INSTALL_DIR     ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR         ${CMAKE_INSTALL_LIBDIR})
foreach(_LANG C CXX CUDA)
    foreach(_TYPE COMPILE LINK)
        set(PROJECT_${_LANG}_${_TYPE}_OPTIONS
            ${${PROJECT_NAME}_${_LANG}_${_TYPE}_OPTIONS})
    endforeach()
endforeach()

set(_INSTALL_PREFIX ${TIMEMORY_INSTALL_PREFIX})
if(TIMEMORY_BUILD_PYTHON)
    execute_process(COMMAND
        ${PYTHON_EXECUTABLE} -c "import sys; print('{}'.format(sys.prefix))"
        OUTPUT_VARIABLE _INSTALL_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PROJECT_NAME}-config.cmake.in
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config.cmake
    INSTALL_DESTINATION ${TIMEMORY_INSTALL_CMAKEDIR}
    INSTALL_PREFIX ${_INSTALL_PREFIX}
    PATH_VARS
        INCLUDE_INSTALL_DIR
        LIB_INSTALL_DIR
        PYTHON_INSTALL_DIR)

# for backwards-compatibility
file(WRITE ${CMAKE_BINARY_DIR}/TiMemoryConfig.cmake
"
#
# This file exists for backwards-compatibility
#

include(\${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}-config.cmake)

")

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(
    FILES
        ${CMAKE_BINARY_DIR}/TiMemoryConfig.cmake
        ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-config.cmake
        ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-version.cmake
    DESTINATION
        ${TIMEMORY_INSTALL_CMAKEDIR})

# only if master project
if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")

    # documentation
    set_property(GLOBAL APPEND PROPERTY TIMEMORY_DOCUMENTATION_INCLUDE_DIRS
        ${PROJECT_SOURCE_DIR}/source
        ${PROJECT_SOURCE_DIR}/timemory
        ${PROJECT_SOURCE_DIR}/examples)

    set(EXCLUDE_LIST ${PROJECT_SOURCE_DIR}/external/cereal
        ${PROJECT_SOURCE_DIR}/external/pybind11)

    add_feature(TIMEMORY_COMPILED_LIBRARIES "Compiled libraries")
    # add_feature(TIMEMORY_INTERFACE_LIBRARIES "Interface libraries")

endif()

if(TIMEMORY_USE_GPERF)
    foreach(_TYPE cpu heap)
        configure_file(${PROJECT_SOURCE_DIR}/cmake/Scripts/gperf-${_TYPE}-profile.sh
            ${CMAKE_BINARY_DIR}/gperf-${_TYPE}-profile.sh COPYONLY)
    endforeach()
    configure_file(${PROJECT_SOURCE_DIR}/cmake/Scripts/gprof2dot.py
        ${CMAKE_BINARY_DIR}/gprof2dot.py COPYONLY)
endif()
