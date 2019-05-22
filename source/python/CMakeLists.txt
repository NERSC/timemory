
################################################################################
#
#        TiMemory (Python)
#
################################################################################

unset(CMAKE_CXX_CLANG_TIDY)

set(PYLIBNAME lib${LIBNAME})
set(TIMEMORY_TLS_MODEL "global-dynamic")
add_cxx_flag_if_avail("-ftls-model=${TIMEMORY_TLS_MODEL}")

execute_process(COMMAND ${PYTHON_EXECUTABLE}
    -c "import sys; print('{}'.format(sys.version_info[0]))"
    OUTPUT_VARIABLE PYTHON_MAJOR_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
add_definitions(-D_PYTHON_MAJOR_VERSION=${PYTHON_MAJOR_VERSION})

file(GLOB_RECURSE pybind_headers ${CMAKE_CURRENT_LIST_DIR}/pybind11/include/*.h)

if(NOT BUILD_SHARED_LIBS AND WIN32)
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/${PYLIBNAME}.cpp
        PROPERTIES COMPILE_DEFINITIONS _TIMEMORY_ARCHIVE)
endif(NOT BUILD_SHARED_LIBS AND WIN32)

#----------------------------------------------------------------------------
# PyBind11
#
pybind11_add_module(${PYLIBNAME} ${CMAKE_CURRENT_LIST_DIR}/${PYLIBNAME}.cpp
    ${CMAKE_CURRENT_LIST_DIR}/${PYLIBNAME}.hpp ${pybind_headers})

target_link_libraries(${PYLIBNAME}
    PRIVATE ${LIBNAME}-cxx-library timemory-headers)

target_compile_definitions(${PYLIBNAME} PRIVATE
    ${${PROJECT_NAME}_DEFINITIONS})

target_compile_options(${PYLIBNAME} PRIVATE
    $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>)

target_include_directories(${PYLIBNAME} SYSTEM PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/pybind11/include)

target_include_directories(${PYLIBNAME} SYSTEM PRIVATE
    ${${PROJECT_NAME}_SYSTEM_INCLUDE_DIRS})

target_include_directories(${PYLIBNAME} PRIVATE
    ${${PROJECT_NAME}_TARGET_INCLUDE_DIRS})

target_include_directories(${PYLIBNAME} SYSTEM
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/pybind11/include)

add_dependencies(${PYLIBNAME} ${LIBNAME}-cxx-library)

if(TIMEMORY_BUILD_C)
    add_dependencies(${PYLIBNAME} ${LIBNAME}-c-library)
endif()

set_target_properties(${PYLIBNAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory
    PDB_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory)

foreach(_SUBMODULE plotting mpi_support util)
    set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/timemory/${_SUBMODULE})
    set(BINARY_DIR ${PROJECT_BINARY_DIR}/timemory/${_SUBMODULE})

    # copy over the python file
    if(EXISTS ${SOURCE_DIR}/${_SUBMODULE}.py)
        configure_file(${SOURCE_DIR}/${_SUBMODULE}.py
            ${BINARY_DIR}/${_SUBMODULE}.py @ONLY)
        install(FILES ${BINARY_DIR}/${_SUBMODULE}.py
            DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/${_SUBMODULE}/
            COMPONENT python)
    endif()

    # copy over the init file
    if(EXISTS ${SOURCE_DIR}/__init__.py)
        configure_file(${SOURCE_DIR}/__init__.py
            ${BINARY_DIR}/__init__.py @ONLY)
        install(FILES ${BINARY_DIR}/__init__.py
            DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/${_SUBMODULE}/
            COMPONENT python)
    endif()

    # copy over the main file
    if(EXISTS ${SOURCE_DIR}/__main__.py)
        configure_file(${SOURCE_DIR}/__main__.py
            ${BINARY_DIR}/__main__.py @ONLY)
        install(FILES ${BINARY_DIR}/__main__.py
            DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/${_SUBMODULE}/
            COMPONENT python)
    endif()

endforeach()

#----------------------------------------------------------------------------
# MPI info
#
set(MPI_EXE_INFO "MPI not supported")
set(MPI_C_INFO "MPI not supported")
set(MPI_CXX_INFO "MPI not supported")
if(TIMEMORY_USE_MPI AND MPI_FOUND)

    execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --version
        OUTPUT_VARIABLE MPI_EXE_INFO
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

    execute_process(COMMAND ${MPI_C_COMPILER} --version
        OUTPUT_VARIABLE MPI_C_INFO
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

    execute_process(COMMAND ${MPI_CXX_COMPILER} --version
        OUTPUT_VARIABLE MPI_CXX_INFO
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

endif(TIMEMORY_USE_MPI AND MPI_FOUND)

file(WRITE ${PROJECT_BINARY_DIR}/timemory/mpi_support/mpi_exe_info.txt "${MPI_EXE_INFO}")
file(WRITE ${PROJECT_BINARY_DIR}/timemory/mpi_support/mpi_c_info.txt "${MPI_C_INFO}")
file(WRITE ${PROJECT_BINARY_DIR}/timemory/mpi_support/mpi_cxx_info.txt "${MPI_CXX_INFO}")

foreach(_TYPE exe c cxx)
    install(FILES ${PROJECT_BINARY_DIR}/timemory/mpi_support/mpi_${_TYPE}_info.txt
        DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/mpi_support)
endforeach(_TYPE exe c cxx)

#----------------------------------------------------------------------------
# Installation info for Python
if(BUILD_SHARED_LIBS)
    set(TIMEMORY_INFO_FILE ${CMAKE_BINARY_DIR}/timemory/lib/timemory_pathinfo.txt)
    add_custom_target(${PROJECT_NAME}-dynamic-path ALL
        COMMAND ${CMAKE_COMMAND}
            -DOUTPUT=${TIMEMORY_INFO_FILE}
            -DLENGTH=2
            -DNAMES="timemory_shared_library\;timemory_dynamic_library"
            -DVALUES="$<TARGET_FILE:${LIBNAME}-cxx-library-shared>\;$<TARGET_FILE:${PYLIBNAME}>"
            -P ${PROJECT_SOURCE_DIR}/cmake/Scripts/generator2file.cmake
        COMMENT "Generating timemory_pathinfo.txt..."
    VERBATIM)

    install(FILES ${TIMEMORY_INFO_FILE}
        DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/lib)
endif(BUILD_SHARED_LIBS)

#----------------------------------------------------------------------------
# Install the python stuff
#
configure_file(${PROJECT_SOURCE_DIR}/timemory/__init__.py.in
    ${PROJECT_BINARY_DIR}/timemory/__init__.py @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/timemory/__init__.py
    DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}
    COMPONENT python)

foreach(PYLIB_SUBMODULE options units signals)
    configure_file(${PROJECT_SOURCE_DIR}/timemory/${PYLIB_SUBMODULE}.py
        ${PROJECT_BINARY_DIR}/timemory/${PYLIB_SUBMODULE}.py @ONLY)
    install(FILES ${PROJECT_BINARY_DIR}/timemory/${PYLIB_SUBMODULE}.py
        DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}
        COMPONENT python)
endforeach()

# Python library target (linked to static target)
install(TARGETS ${PYLIBNAME}
    DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}
    COMPONENT python)

export(TARGETS ${INSTALL_LIBRARIES} FILE ${CMAKE_BINARY_DIR}/TiMemoryBuild.cmake)

#----------------------------------------------------------------------------
# Python testing
#
set(TEST_FILES
    __init__.py
    __main__.py
    tests.py
    timemory_test.py
    simple_test.py
    nested_test.py
    array_test.py)

foreach(_FILE ${TEST_FILES})
    # only copy *_test.py files to binary directory
    if("${_FILE}" MATCHES "_test.py")
        configure_file(${PROJECT_SOURCE_DIR}/timemory/tests/${_FILE}
            ${PROJECT_BINARY_DIR}/${_FILE} @ONLY)
    endif("${_FILE}" MATCHES "_test.py")

    # copy to binary: timemory/tests/${_FILE}
    configure_file(${PROJECT_SOURCE_DIR}/timemory/tests/${_FILE}
        ${PROJECT_BINARY_DIR}/timemory/tests/${_FILE} @ONLY)

    # install them though
    install(FILES ${PROJECT_BINARY_DIR}/timemory/tests/${_FILE}
        DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/tests
        COMPONENT python)

endforeach(_FILE ${TEST_FILES})

#----------------------------------------------------------------------------
# Python tools
#
add_subdirectory(tools)
