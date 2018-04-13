
################################################################################
#
#        TiMemory (Python)
#
################################################################################

if(TIMEMORY_DYNAMIC_LINK)
    set(LINK_TYPE shared)
else(TIMEMORY_DYNAMIC_LINK)
    set(LINK_TYPE static)
    if(WIN32)
        set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/${LIBNAME}.cpp
            PROPERTIES COMPILE_DEFINITIONS _TIMEMORY_ARCHIVE)
    endif(WIN32)
endif(TIMEMORY_DYNAMIC_LINK)

#----------------------------------------------------------------------------
# PyBind11
#
pybind11_add_module(${LIBNAME} ${CMAKE_CURRENT_LIST_DIR}/${LIBNAME}.cpp)
# link to static library so no need for RPATH resolution
target_link_libraries(${LIBNAME} PUBLIC ${LIBNAME}-${LINK_TYPE} ${EXTERNAL_LIBRARIES})
add_dependencies(${LIBNAME} ${LIBNAME}-${LINK_TYPE})

set_target_properties(${LIBNAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory
    PDB_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory)

file(READ ${PROJECT_SOURCE_DIR}/.LICENSE LICENSE)
string(REPLACE "// " "# " LICENSE "${LICENSE}")
foreach(_FILE plotting mpi_support util)
    set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/timemory/${_FILE})
    set(BINARY_DIR ${PROJECT_BINARY_DIR}/timemory/${_FILE})

    # copy over the python file
    configure_file(${SOURCE_DIR}/${_FILE}.py
        ${BINARY_DIR}/${_FILE}.py COPYONLY)

    # copy over the init file
    configure_file(${SOURCE_DIR}/__init__.py
        ${BINARY_DIR}/__init__.py COPYONLY)

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
if(TIMEMORY_DYNAMIC_LINK)
    set(TIMEMORY_INFO_FILE ${CMAKE_BINARY_DIR}/timemory/lib/timemory_pathinfo.txt)
    add_custom_target(${PROJECT_NAME}-shared-library-path ALL
        COMMAND ${CMAKE_COMMAND}
            -DOUTPUT=${TIMEMORY_INFO_FILE}
            -DLENGTH=2
            -DNAMES="timemory_shared_library\;timemory_dynamic_library"
            -DVALUES="$<TARGET_FILE:${LIBNAME}-shared>\;$<TARGET_FILE:${LIBNAME}>"
            -P ${PROJECT_SOURCE_DIR}/cmake/Scripts/generator2file.cmake
        COMMENT "Generating timemory_pathinfo.txt..."
    VERBATIM)

    install(FILES ${TIMEMORY_INFO_FILE}
        DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/lib)
endif(TIMEMORY_DYNAMIC_LINK)

#----------------------------------------------------------------------------
# Install the python stuff
#
configure_file(${PROJECT_SOURCE_DIR}/timemory/timemory.py
    ${PROJECT_BINARY_DIR}/timemory/timemory.py COPYONLY)
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
