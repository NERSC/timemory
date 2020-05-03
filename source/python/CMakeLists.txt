
##########################################################################################
#
#        timemory (Python)
#
##########################################################################################

if(NOT TIMEMORY_BUILD_PYTHON AND NOT TIMEMORY_USE_PYTHON)
    return()
endif()

unset(CMAKE_CXX_CLANG_TIDY)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME python)

add_library(timemory-python-compile-options INTERFACE)
add_cxx_flag_if_avail("-ftls-model=global-dynamic" timemory-python-compile-options)
add_cxx_flag_if_avail("-Wno-deprecated-declarations" timemory-python-compile-options)
add_cxx_flag_if_avail("-Wno-unused-but-set-parameter" timemory-python-compile-options)

execute_process(COMMAND ${PYTHON_EXECUTABLE}
    -c "import sys; print('{}'.format(sys.version_info[0]))"
    OUTPUT_VARIABLE PYTHON_MAJOR_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
add_definitions(-D_PYTHON_MAJOR_VERSION=${PYTHON_MAJOR_VERSION})

file(GLOB_RECURSE pybind_headers ${CMAKE_CURRENT_LIST_DIR}/pybind11/include/*.h)

#----------------------------------------------------------------------------
# PyBind11
#
add_library(libpytimemory MODULE ${CMAKE_CURRENT_LIST_DIR}/libpytimemory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/libpytimemory.hpp ${pybind_headers})

target_include_directories(libpytimemory SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/pybind11/include>)

target_link_libraries(libpytimemory PRIVATE
    timemory-headers
    timemory-cxx-shared
    timemory-compile-options
    timemory-external-shared
    timemory-python-compile-options
    timemory-mpip-library
    timemory-ompt-library
    PRIVATE pybind11::module)

if(TIMEMORY_USE_CUDA)
    set(EXTRA_PROPERTIES LINKER_LANGUAGE CUDA)
endif()

set_target_properties(libpytimemory PROPERTIES
    PREFIX                      ""
    SUFFIX                      "${PYTHON_MODULE_EXTENSION}"
    LIBRARY_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/timemory
    ARCHIVE_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/timemory
    RUNTIME_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/timemory
    PDB_OUTPUT_DIRECTORY        ${CMAKE_BINARY_DIR}/timemory
    INSTALL_RPATH_USE_LINK_PATH ON
    ${EXTRA_PROPERTIES})

if(UNIX AND NOT APPLE)
    set_target_properties(libpytimemory PROPERTIES INSTALL_RPATH
        "\$ORIGIN/../../..:\$ORIGIN/../../../timemory:\$ORIGIN:${CMAKE_INSTALL_RPATH}")
endif()

set(PYTHON_SUBMODULE_FOLDERS
    ert gperftools mpi_support plotting profiler roofline util mpi bundle)

set(PYTHON_SUBMODULE_FILES
    common options signals units)

foreach(_SUBMODULE ${PYTHON_SUBMODULE_FOLDERS})
    set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/timemory/${_SUBMODULE})
    set(BINARY_DIR ${CMAKE_BINARY_DIR}/timemory/${_SUBMODULE})

    # copy over the python file
    if(EXISTS ${SOURCE_DIR}/${_SUBMODULE}.py)
        configure_file(${SOURCE_DIR}/${_SUBMODULE}.py
            ${BINARY_DIR}/${_SUBMODULE}.py @ONLY)
        install(FILES ${BINARY_DIR}/${_SUBMODULE}.py
            DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME}/${_SUBMODULE}/)
    endif()

    # copy over the init file
    if(EXISTS ${SOURCE_DIR}/__init__.py)
        configure_file(${SOURCE_DIR}/__init__.py
            ${BINARY_DIR}/__init__.py @ONLY)
        install(FILES ${BINARY_DIR}/__init__.py
            DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME}/${_SUBMODULE}/)
    endif()

    # copy over the main file
    if(EXISTS ${SOURCE_DIR}/__main__.py)
        configure_file(${SOURCE_DIR}/__main__.py
            ${BINARY_DIR}/__main__.py @ONLY)
        install(FILES ${BINARY_DIR}/__main__.py
            DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME}/${_SUBMODULE}/)
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

file(WRITE ${CMAKE_BINARY_DIR}/timemory/mpi_support/mpi_exe_info.txt "${MPI_EXE_INFO}")
file(WRITE ${CMAKE_BINARY_DIR}/timemory/mpi_support/mpi_c_info.txt "${MPI_C_INFO}")
file(WRITE ${CMAKE_BINARY_DIR}/timemory/mpi_support/mpi_cxx_info.txt "${MPI_CXX_INFO}")

foreach(_TYPE exe c cxx)
    install(FILES ${CMAKE_BINARY_DIR}/timemory/mpi_support/mpi_${_TYPE}_info.txt
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME}/mpi_support)
endforeach(_TYPE exe c cxx)


#----------------------------------------------------------------------------
# Install the python stuff
#
configure_file(${PROJECT_SOURCE_DIR}/timemory/__init__.py.in
    ${CMAKE_BINARY_DIR}/timemory/__init__.py @ONLY)

install(FILES ${CMAKE_BINARY_DIR}/timemory/__init__.py
    DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME})

foreach(PYLIB_SUBMODULE ${PYTHON_SUBMODULE_FILES})
    configure_file(${PROJECT_SOURCE_DIR}/timemory/${PYLIB_SUBMODULE}.py
        ${CMAKE_BINARY_DIR}/timemory/${PYLIB_SUBMODULE}.py @ONLY)

    install(FILES ${CMAKE_BINARY_DIR}/timemory/${PYLIB_SUBMODULE}.py
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME})
endforeach()

# Python library target
install(TARGETS libpytimemory
    DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME})

#----------------------------------------------------------------------------
# Python tools
#
add_subdirectory(tools)
