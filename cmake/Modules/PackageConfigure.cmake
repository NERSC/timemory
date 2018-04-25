
################################################################################
#
#        TiMemory Package installation
#
################################################################################

include(CMakePackageConfigHelpers)

set(PYTHON_INSTALL_DIR      ${TIMEMORY_CONFIG_PYTHONDIR})
set(INCLUDE_INSTALL_DIR     ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR         ${CMAKE_INSTALL_LIBDIR})

set(_INSTALL_PREFIX ${TIMEMORY_INSTALL_PREFIX})
if(TIMEMORY_SETUP_PY)
    execute_process(COMMAND
        ${PYTHON_EXECUTABLE} -c "import sys; print('{}'.format(sys.prefix))"
        OUTPUT_VARIABLE _INSTALL_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif(TIMEMORY_SETUP_PY)

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

if(NOT TIMEMORY_SETUP_PY OR TIMEMORY_DEVELOPER_INSTALL)
    install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${TIMEMORY_INSTALL_CMAKEDIR}
        COMPONENT development)
endif(NOT TIMEMORY_SETUP_PY OR TIMEMORY_DEVELOPER_INSTALL)

if(NOT SUBPROJECT)
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
    endif(TIMEMORY_DOXYGEN_DOCS)

    print_features()

endif(NOT SUBPROJECT)
