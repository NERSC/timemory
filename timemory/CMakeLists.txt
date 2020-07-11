
##########################################################################################
#
#        timemory (Python)
#
##########################################################################################

if(SKBUILD)
    set(CMAKE_INSTALL_PYTHONDIR ${CMAKE_INSTALL_PREFIX})
endif()

set(_PYLIB ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME})
if(NOT IS_ABSOLUTE "${_PYLIB}")
    set(_PYLIB ${CMAKE_INSTALL_PREFIX}/${_PYLIB})
endif()
file(RELATIVE_PATH LIB_RELPATH "${_PYLIB}"
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

#----------------------------------------------------------------------------
#
#   Always install these files since timemory has python scripts which
#   do not depend on the library being built
#
#----------------------------------------------------------------------------

set(PYTHON_SUBMODULE_FOLDERS
    api ert mpi mpi_support plotting profiler roofline settings
    util bundle component hardware_counters test trace region)

file(GLOB PYTHON_SUBMODULE_FILES ${PROJECT_SOURCE_DIR}/${PROJECT_NAME}/*.py)
string(REPLACE "${PROJECT_SOURCE_DIR}/${PROJECT_NAME}/" "" PYTHON_SUBMODULE_FILES "${PYTHON_SUBMODULE_FILES}")

foreach(_SUBMODULE ${PYTHON_SUBMODULE_FOLDERS})
    set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${_SUBMODULE})
    file(GLOB_RECURSE _FILES ${SOURCE_DIR}/*.py ${SOURCE_DIR}/*.py.in)

    # copy over the python file
    foreach(_FILE ${_FILES})
        get_filename_component(_DIR "${_FILE}" DIRECTORY)
        get_filename_component(_FILE "${_FILE}" NAME_WE)
        string(REPLACE "${PROJECT_SOURCE_DIR}/" "" _DIR "${_DIR}")
        if(EXISTS ${PROJECT_SOURCE_DIR}/${_DIR}/${_FILE}.py)
            configure_file(${PROJECT_SOURCE_DIR}/${_DIR}/${_FILE}.py
                ${CMAKE_BINARY_DIR}/${_DIR}/${_FILE}.py @ONLY)
        elseif(EXISTS ${PROJECT_SOURCE_DIR}/${_DIR}/${_FILE}.py.in)
            configure_file(${PROJECT_SOURCE_DIR}/${_DIR}/${_FILE}.py.in
                ${CMAKE_BINARY_DIR}/${_DIR}/${_FILE}.py @ONLY)
        endif()
        if(CMAKE_INSTALL_PYTHONDIR)
            install(FILES ${CMAKE_BINARY_DIR}/${_DIR}/${_FILE}.py
                DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${_DIR})
        endif()
    endforeach()
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

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/mpi_support/mpi_exe_info.txt "${MPI_EXE_INFO}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/mpi_support/mpi_c_info.txt "${MPI_C_INFO}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/mpi_support/mpi_cxx_info.txt "${MPI_CXX_INFO}")

if(CMAKE_INSTALL_PYTHONDIR)
    foreach(_TYPE exe c cxx)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mpi_support/mpi_${_TYPE}_info.txt
            DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME}/mpi_support)
    endforeach(_TYPE exe c cxx)
endif()

#----------------------------------------------------------------------------
# Install the python stuff
#
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/__init__.py @ONLY)

if(CMAKE_INSTALL_PYTHONDIR)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME})
endif()

foreach(PYLIB_SUBMODULE ${PYTHON_SUBMODULE_FILES})
    get_filename_component(PYLIB_SUBMODULE "${PYLIB_SUBMODULE}" NAME_WE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${PYLIB_SUBMODULE}.py
        ${CMAKE_CURRENT_BINARY_DIR}/${PYLIB_SUBMODULE}.py @ONLY)

    if(CMAKE_INSTALL_PYTHONDIR)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PYLIB_SUBMODULE}.py
            DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME})
    endif()
endforeach()
