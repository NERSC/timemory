
################################################################################
#
#        TiMemory Package installation
#
################################################################################

include(CMakePackageConfigHelpers)

set(PYTHON_INSTALL_DIR      ${TIMEMORY_INSTALL_PYTHONDIR})
set(INCLUDE_INSTALL_DIR     ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR         ${CMAKE_INSTALL_LIBDIR})

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${TIMEMORY_INSTALL_CMAKEDIR}
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
        DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/tests
        COMPONENT python)

endforeach(_FILE ${TEST_FILES})

if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/Modules/Testing.cmake")
    include(Testing)
endif(EXISTS "${CMAKE_SOURCE_DIR}/cmake/Modules/Testing.cmake")


if(NOT SUBPROJECT)
    # documentation
    set_property(GLOBAL APPEND PROPERTY TIMEMORY_DOCUMENTATION_INCLUDE_DIRS
        ${PROJECT_SOURCE_DIR}/source
        ${PROJECT_SOURCE_DIR}/timemory
        ${PROJECT_SOURCE_DIR}/examples)

    set(EXCLUDE_LIST ${PROJECT_SOURCE_DIR}/source/cereal)
    include(Documentation)

    if(TIMEMORY_DOXYGEN_DOCS)
        SET(CMAKE_INSTALL_MESSAGE NEVER)
        Generate_Documentation(Doxyfile.${PROJECT_NAME})
        SET(CMAKE_INSTALL_MESSAGE LAZY)
    endif(TIMEMORY_DOXYGEN_DOCS)

    print_features()

endif(NOT SUBPROJECT)
