# include guard
include_guard(DIRECTORY)

##########################################################################################
#
#        TiMemory Package installation
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
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${TIMEMORY_INSTALL_CMAKEDIR}
    INSTALL_PREFIX ${_INSTALL_PREFIX}
    PATH_VARS
        INCLUDE_INSTALL_DIR
        LIB_INSTALL_DIR
        PYTHON_INSTALL_DIR)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(
    FILES
        ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION
        ${TIMEMORY_INSTALL_CMAKEDIR}
    COMPONENT
        development)

# only if master project
if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")

    # documentation
    set_property(GLOBAL APPEND PROPERTY TIMEMORY_DOCUMENTATION_INCLUDE_DIRS
        ${PROJECT_SOURCE_DIR}/source
        ${PROJECT_SOURCE_DIR}/timemory
        ${PROJECT_SOURCE_DIR}/examples)

    set(EXCLUDE_LIST ${PROJECT_SOURCE_DIR}/source/cereal
        ${PROJECT_SOURCE_DIR}/source/python/pybind11)

    include(Documentation)

    if(TIMEMORY_DOXYGEN_DOCS)
        SET(CMAKE_INSTALL_MESSAGE NEVER)
        Generate_Documentation(Doxyfile.${PROJECT_NAME})
        SET(CMAKE_INSTALL_MESSAGE LAZY)
    endif()

    add_feature(TIMEMORY_COMPILED_LIBRARIES "Compiled libraries")
    add_feature(TIMEMORY_INTERFACE_LIBRARIES "Interface libraries")
    print_features()

endif()

if(TIMEMORY_USE_GPERF)
    foreach(_TYPE cpu heap)
        configure_file(${PROJECT_SOURCE_DIR}/cmake/Scripts/gperf-${_TYPE}-profile.sh
            ${CMAKE_BINARY_DIR}/gperf-${_TYPE}-profile.sh COPYONLY)
    endforeach()
endif()
