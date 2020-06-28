
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
add_library(${PROJECT_NAME}::timemory-python-compile-options ALIAS
    timemory-python-compile-options)
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
file(GLOB pyheaders ${CMAKE_CURRENT_LIST_DIR}/libpytimemory*.hpp)
file(GLOB pysources ${CMAKE_CURRENT_LIST_DIR}/libpytimemory*.cpp)

if(TIMEMORY_USE_CUDA AND CMAKE_CUDA_COMPILER AND TIMEMORY_BUILD_PYTHON_CUDA)
    set_source_files_properties(${pysources} PROPERTIES
        LANGUAGE CUDA
        LINKER_LANGUAGE CUDA)
endif()

add_library(libpytimemory MODULE ${pysources} ${pyheaders} ${pybind_headers})
add_library(${PROJECT_NAME}::libpytimemory ALIAS libpytimemory)
target_compile_definitions(libpytimemory PRIVATE TIMEMORY_PYBIND11_SOURCE)

target_include_directories(libpytimemory SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/pybind11/include>)

target_link_libraries(libpytimemory PRIVATE
    pybind11::module
    timemory-python
    timemory-headers
    timemory-cxx-shared
    timemory-compile-options
    timemory-external-shared
    timemory-python-compile-options
    timemory-mpip-library
    timemory-ompt-library
    timemory-default-visibility)

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
    api ert mpi mpi_support plotting profiler roofline settings
    util bundle component hardware_counters test)

file(GLOB PYTHON_SUBMODULE_FILES ${PROJECT_SOURCE_DIR}/${PROJECT_NAME}/*.py)
string(REPLACE "${PROJECT_SOURCE_DIR}/${PROJECT_NAME}/" "" PYTHON_SUBMODULE_FILES "${PYTHON_SUBMODULE_FILES}")

foreach(_SUBMODULE ${PYTHON_SUBMODULE_FOLDERS})
    set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/timemory/${_SUBMODULE})
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
        install(FILES ${CMAKE_BINARY_DIR}/${_DIR}/${_FILE}.py
            DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${_DIR})
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
    get_filename_component(PYLIB_SUBMODULE "${PYLIB_SUBMODULE}" NAME_WE)
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
