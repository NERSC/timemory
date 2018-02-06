
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
if(NOT SETUP_PY)
    add_subdirectory(cereal)
endif(NOT SETUP_PY)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ${DEV_WARNINGS} CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)
#message(STATUS "Dev warnings: ${DEV_WARNINGS}")
include_directories(${CMAKE_CURRENT_LIST_DIR}/cereal/include)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# - headers are included so they will show up in IDEs
file(GLOB sources ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB headers ${CMAKE_CURRENT_LIST_DIR}/timemory/*.hpp)

# libraries to install
set(INSTALL_LIBRARIES )

# macro to build a library of type: shared, static
macro(_BUILD_LIBRARY type output_name)
    string(TOUPPER "${type}" LIB_TYPE)
    add_library(${LIBNAME}-${type} ${LIB_TYPE} ${ARGN})
    target_link_libraries(${LIBNAME}-${type} ${EXTERNAL_LIBRARIES})
    set_target_properties(${LIBNAME}-${type} PROPERTIES
        OUTPUT_NAME ${output_name}
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})
    list(APPEND INSTALL_LIBRARIES ${LIBNAME}-${type})
endmacro(_BUILD_LIBRARY TYPE)

# library type to link pybind11 module to
set(LINK_TYPE static)

if(WIN32)
    #_build_library(shared ${LIBNAME}-shared ${sources} ${headers})
    _build_library(static ${LIBNAME}-archive ${sources} ${headers})
    set(LINK_TYPE static)
else(WIN32)
    message(STATUS "")
    message(STATUS "Building ${PROJECT_NAME} with object library...")
    message(STATUS "")
    add_library(${LIBNAME}-object OBJECT ${sources} ${headers})
    _build_library(shared ${LIBNAME} $<TARGET_OBJECTS:${LIBNAME}-object>)
    _build_library(static ${LIBNAME} $<TARGET_OBJECTS:${LIBNAME}-object>)
    set(LINK_TYPE static)
endif(WIN32)



#----------------------------------------------------------------------------
# PyBind11
#
message(STATUS "External libraries: ${EXTERNAL_LIBRARIES}")
pybind11_add_module(${LIBNAME} ${CMAKE_CURRENT_LIST_DIR}/${LIBNAME}.cc)
# link to static library so no need for RPATH resolution
target_link_libraries(${LIBNAME} PRIVATE ${LIBNAME}-${LINK_TYPE} ${EXTERNAL_LIBRARIES})

set_target_properties(${LIBNAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory)

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
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${_FILE})
endforeach(_FILE plotting mpi_support util)

#----------------------------------------------------------------------------
# MPI info
#
set(MPI_EXE_INFO "MPI not supported")
set(MPI_C_INFO "MPI not supported")
set(MPI_CXX_INFO "MPI not supported")
if(USE_MPI AND MPI_FOUND)

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

endif(USE_MPI AND MPI_FOUND)

file(WRITE ${PROJECT_BINARY_DIR}/timemory/mpi_support/mpi_exe_info.txt "${MPI_EXE_INFO}")
file(WRITE ${PROJECT_BINARY_DIR}/timemory/mpi_support/mpi_c_info.txt "${MPI_C_INFO}")
file(WRITE ${PROJECT_BINARY_DIR}/timemory/mpi_support/mpi_cxx_info.txt "${MPI_CXX_INFO}")

foreach(_TYPE exe c cxx)
    install(FILES ${PROJECT_BINARY_DIR}/timemory/mpi_support/mpi_${_TYPE}_info.txt
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/mpi_support)
endforeach(_TYPE exe c cxx)

#----------------------------------------------------------------------------
# Install the targets and export
#

if(NOT SETUP_PY)
    # C++ compiled libraries
    install(TARGETS ${INSTALL_LIBRARIES}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        EXPORT ${PROJECT_NAME}LibraryDepends)

    # C++ development headers
    install(FILES ${headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/timemory)

    # Install the export set for use with the install-tree
    install(EXPORT ${PROJECT_NAME}LibraryDepends
        DESTINATION ${PROJECT_INSTALL_CMAKEDIR}
        COMPONENT dev)

endif(NOT SETUP_PY)

# Python library target (linked to static target)
install(TARGETS ${LIBNAME}
    DESTINATION ${CMAKE_INSTALL_PYTHONDIR})

# it is in a folder that gets installed but still do it explicitly
#install(TARGETS ${LIBNAME}
#    DESTINATION ${CMAKE_INSTALL_PREFIX})

install(TARGETS ${LIBNAME}
    DESTINATION ${CMAKE_INSTALL_PYTHONDIR})

configure_file(${PROJECT_SOURCE_DIR}/timemory/timemory.py
    ${PROJECT_BINARY_DIR}/timemory/timemory.py COPYONLY)

install(FILES ${PROJECT_BINARY_DIR}/timemory/timemory.py
    DESTINATION ${CMAKE_INSTALL_PYTHONDIR})

configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/__init__.py.in
    ${PROJECT_BINARY_DIR}/timemory/__init__.py @ONLY)

install(FILES ${PROJECT_BINARY_DIR}/timemory/__init__.py
    DESTINATION ${CMAKE_INSTALL_PYTHONDIR})
