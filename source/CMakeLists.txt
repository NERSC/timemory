
include_directories(${CMAKE_CURRENT_LIST_DIR})

################################################################################
#
#        Cereal
#
################################################################################

# cereal options
add_option(WITH_WERROR "Compile with '-Werror' C++ compiler flag" OFF NO_FEATURE)
add_option(THREAD_SAFE "Compile Cereal with THREAD_SAFE option" ON NO_FEATURE)
add_option(JUST_INSTALL_CEREAL "Skip testing of Cereal" ON NO_FEATURE)
add_option(SKIP_PORTABILITY_TEST "Skip Cereal portability test" ON NO_FEATURE)

set(DEV_WARNINGS ${CMAKE_SUPPRESS_DEVELOPER_WARNINGS})
# this gets annoying
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)

# add cereal
add_subdirectory(cereal)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ${DEV_WARNINGS} CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)

include_directories(${CMAKE_CURRENT_LIST_DIR}/cereal/include)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# - headers are included so they will show up in IDEs
file(GLOB sources ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB headers ${CMAKE_CURRENT_LIST_DIR}/timemory/*.hpp)

add_library(${LIBNAME}-object OBJECT ${sources} ${headers})
add_library(${LIBNAME}-shared SHARED $<TARGET_OBJECTS:${LIBNAME}-object>)
add_library(${LIBNAME}-static STATIC $<TARGET_OBJECTS:${LIBNAME}-object>)
target_link_libraries(${LIBNAME}-shared ${EXTERNAL_LIBRARIES})
target_link_libraries(${LIBNAME}-static ${EXTERNAL_LIBRARIES})

set_target_properties(${LIBNAME}-shared PROPERTIES
    OUTPUT_NAME ${LIBNAME}
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
set_target_properties(${LIBNAME}-static PROPERTIES
    OUTPUT_NAME ${LIBNAME}
    VERSION ${PROJECT_VERSION})

#----------------------------------------------------------------------------
# PyBind11
#
message(STATUS "External libraries: ${EXTERNAL_LIBRARIES}")
pybind11_add_module(${LIBNAME} ${CMAKE_CURRENT_LIST_DIR}/${LIBNAME}.cc)
# link to static library so no need for RPATH resolution
target_link_libraries(${LIBNAME} PRIVATE ${LIBNAME}-static ${EXTERNAL_LIBRARIES})

foreach(_FILE plot options)
    configure_file(${PROJECT_SOURCE_DIR}/python/${_FILE}.py
        ${PROJECT_BINARY_DIR}/timemory-supp/${_FILE}.py COPYONLY)
endforeach(_FILE plot options)

file(WRITE ${PROJECT_BINARY_DIR}/timemory-supp/__init__.py
"#!/usr/bin/env python

from .plot import *
from .options import *
from .mpi_support import *

")

#----------------------------------------------------------------------------
# MPI info
#

set(MPI_EXE_INFO "MPI not supported")
set(MPI_C_INFO "MPI not supported")
set(MPI_CXX_INFO "MPI not supported")
if(USE_MPI)

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

endif(USE_MPI)

file(WRITE ${PROJECT_BINARY_DIR}/timemory-supp/mpi_exe_info.txt "${MPI_EXE_INFO}")
file(WRITE ${PROJECT_BINARY_DIR}/timemory-supp/mpi_c_info.txt "${MPI_C_INFO}")
file(WRITE ${PROJECT_BINARY_DIR}/timemory-supp/mpi_cxx_info.txt "${MPI_CXX_INFO}")

file(WRITE ${PROJECT_BINARY_DIR}/timemory-supp/mpi_support.py
"#!/usr/bin/env python

import os

def _path_to_file(pfile):
    import timemory
    return '{}/timemory-supp/{}'.format(os.path.dirname(timemory.__file__), pfile)

def mpi_exe_info():
    return print(open(_path_to_file('mpi_exe_info.txt'), 'r').read())

def mpi_c_info():
    return print(open(_path_to_file('mpi_c_info.txt'), 'r').read())

def mpi_cxx_info():
    return print(open(_path_to_file('mpi_cxx_info.txt'), 'r').read())

")

#----------------------------------------------------------------------------
# Install the targets and export
#

# C++ compiled libraries
install(TARGETS ${LIBNAME}-shared ${LIBNAME}-static
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    EXPORT ${PROJECT_NAME}LibraryDepends)

# C++ development headers
install(FILES ${headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/timemory)

# Python library target (linked to static target)
install(TARGETS ${LIBNAME}
    DESTINATION ${CMAKE_INSTALL_PYTHONDIR})

# Python supplementary files
install(FILES
    ${PROJECT_BINARY_DIR}/timemory-supp/__init__.py
    ${PROJECT_BINARY_DIR}/timemory-supp/plot.py
    ${PROJECT_BINARY_DIR}/timemory-supp/options.py
    ${PROJECT_BINARY_DIR}/timemory-supp/mpi_support.py
    # MPI info files
    ${PROJECT_BINARY_DIR}/timemory-supp/mpi_exe_info.txt
    ${PROJECT_BINARY_DIR}/timemory-supp/mpi_c_info.txt
    ${PROJECT_BINARY_DIR}/timemory-supp/mpi_cxx_info.txt
    # ${PROJECT_BINARY_DIR}/timemory-supp/decorator.py
    DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/timemory-supp)

# Install the export set for use with the install-tree
install(EXPORT ${PROJECT_NAME}LibraryDepends
    DESTINATION ${PROJECT_INSTALL_CMAKEDIR}
    COMPONENT dev)
