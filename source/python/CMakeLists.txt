
################################################################################
#
#        TiMemory (Python)
#
################################################################################

execute_process(COMMAND ${PYTHON_EXECUTABLE}
    -c "import sys; print('{}'.format(sys.version_info[0]))"
    OUTPUT_VARIABLE PYTHON_MAJOR_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
add_definitions(-D_PYTHON_MAJOR_VERSION=${PYTHON_MAJOR_VERSION})

file(GLOB_RECURSE pybind_headers ${CMAKE_CURRENT_LIST_DIR}/pybind11/include/*.h)

if(NOT BUILD_SHARED_LIBS AND WIN32)
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/${LIBNAME}.cpp
        PROPERTIES COMPILE_DEFINITIONS _TIMEMORY_ARCHIVE)
endif(NOT BUILD_SHARED_LIBS AND WIN32)

#----------------------------------------------------------------------------
# PyBind11
#
pybind11_add_module(${LIBNAME} ${CMAKE_CURRENT_LIST_DIR}/${LIBNAME}.cpp
    py${LIBNAME}.hpp ${pybind_headers})
# link cxx library and external libraries
target_link_libraries(${LIBNAME} PUBLIC ${LIBNAME}-cxx-library ${EXTERNAL_LIBRARIES})
add_dependencies(${LIBNAME} ${LIBNAME}-c-library ${LIBNAME}-cxx-library)

set_target_properties(${LIBNAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory
    PDB_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory)

foreach(_FILE plotting mpi_support util)
    set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/timemory/${_FILE})
    set(BINARY_DIR ${PROJECT_BINARY_DIR}/timemory/${_FILE})

    # copy over the python file
    configure_file(${SOURCE_DIR}/${_FILE}.py
        ${BINARY_DIR}/${_FILE}.py @ONLY)

    # copy over the init file
    configure_file(${SOURCE_DIR}/__init__.py
        ${BINARY_DIR}/__init__.py @ONLY)

    # install the folder
    install(DIRECTORY ${BINARY_DIR}/
        DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/${_FILE}
        COMPONENT python)
endforeach(_FILE plotting mpi_support util)

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
            -DVALUES="$<TARGET_FILE:${LIBNAME}-cxx-library>\;$<TARGET_FILE:${LIBNAME}>"
            -P ${PROJECT_SOURCE_DIR}/cmake/Scripts/generator2file.cmake
        COMMENT "Generating timemory_pathinfo.txt..."
    VERBATIM)

    install(FILES ${TIMEMORY_INFO_FILE}
        DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/lib)
endif(BUILD_SHARED_LIBS)

#----------------------------------------------------------------------------
# Install the python stuff
#
configure_file(${PROJECT_SOURCE_DIR}/timemory/timemory.py
    ${PROJECT_BINARY_DIR}/timemory/timemory.py @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/__init__.py.in
    ${PROJECT_BINARY_DIR}/timemory/__init__.py @ONLY)

# Python library target (linked to static target)
install(TARGETS ${LIBNAME}
    DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}
    COMPONENT python)

install(FILES ${PROJECT_BINARY_DIR}/timemory/timemory.py
    DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}
    COMPONENT python)

install(FILES ${PROJECT_BINARY_DIR}/timemory/__init__.py
    DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}
    COMPONENT python)

export(TARGETS ${INSTALL_LIBRARIES} FILE ${CMAKE_BINARY_DIR}/TiMemoryBuild.cmake)

#----------------------------------------------------------------------------
# Python testing
#
set(TEST_FILES
    __init__.py
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
