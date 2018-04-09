
include_directories(${CMAKE_CURRENT_LIST_DIR})

#if(SHARED_PYTHON_LINKING)
#    set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_FULL_LIBDIR})
#    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
#endif(SHARED_PYTHON_LINKING)

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
macro(_BUILD_LIBRARY TYPE OUTPUT_NAME)
    string(TOUPPER "${TYPE}" LIB_TYPE)
    add_library(${OUTPUT_NAME}-${TYPE} ${LIB_TYPE} ${ARGN})
    target_link_libraries(${OUTPUT_NAME}-${TYPE} PUBLIC ${EXTERNAL_LIBRARIES})
	if(WIN32)
	    set_target_properties(${OUTPUT_NAME}-${TYPE} PROPERTIES
		    OUTPUT_NAME ${OUTPUT_NAME}
			COMPILE_DEFINITIONS TIMEMORY_DLL
			LANGUAGE CXX
            # C++ shared library fallback installation
            LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory/lib
            POSITION_INDEPENDENT_CODE ON)
		target_compile_options(${OUTPUT_NAME}-${TYPE} PRIVATE /MP /bigobj /Tp)
	else(WIN32)
        if(XCODE)
            # xcode has issue with "cmake_symlink_library"
            set_target_properties(${OUTPUT_NAME}-${TYPE} PROPERTIES
                OUTPUT_NAME ${OUTPUT_NAME}
                # C++ shared library fallback installation
                LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory/lib
                POSITION_INDEPENDENT_CODE ON)
        else(XCODE)
            set_target_properties(${OUTPUT_NAME}-${TYPE} PROPERTIES
                OUTPUT_NAME ${OUTPUT_NAME}
                VERSION ${PROJECT_VERSION}
                SOVERSION ${PROJECT_VERSION_MAJOR}
                # C++ shared library fallback installation
                LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/timemory/lib
                POSITION_INDEPENDENT_CODE ON)
        endif(XCODE)
	endif(WIN32)
    list(APPEND INSTALL_LIBRARIES ${LIBNAME}-${TYPE})
endmacro(_BUILD_LIBRARY TYPE OUTPUT_NAME)

# this includes compile definitions for headers
add_subdirectory(timemory)

message(STATUS "")
message(STATUS "Building ${PROJECT_NAME} with object library...")
message(STATUS "")
if(NOT XCODE)
    add_library(${LIBNAME}-object OBJECT ${sources} ${headers})
    _build_library(shared ${LIBNAME} $<TARGET_OBJECTS:${LIBNAME}-object>)
    _build_library(static ${LIBNAME} $<TARGET_OBJECTS:${LIBNAME}-object>)
else(NOT XCODE)
    _build_library(shared ${LIBNAME} ${sources} ${headers})
    _build_library(static ${LIBNAME} ${sources} ${headers})
endif(NOT XCODE)

# library type to link pybind11 module to
add_option(SHARED_PYTHON_LINKING
    "Link the TiMemory PyBind11 module to the shared library" OFF)
set(LINK_TYPE static)

if(SHARED_PYTHON_LINKING)
    set(LINK_TYPE shared)
    install(TARGETS ${LIBNAME}-shared
        DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/lib
        COMPONENT python)
endif(SHARED_PYTHON_LINKING)

#----------------------------------------------------------------------------
# PyBind11
#
message(STATUS "External libraries: ${EXTERNAL_LIBRARIES}")
pybind11_add_module(${LIBNAME} ${CMAKE_CURRENT_LIST_DIR}/${LIBNAME}.cc)
# link to static library so no need for RPATH resolution
target_link_libraries(${LIBNAME} PUBLIC ${LIBNAME}-${LINK_TYPE} ${EXTERNAL_LIBRARIES})
add_dependencies(${LIBNAME} ${LIBNAME}-${LINK_TYPE})

if(WIN32)
	set_target_properties(${OUTPUT_NAME} PROPERTIES
		COMPILE_FLAGS "/MP /bigobj /Tp"
		LANGUAGE CXX
		POSITION_INDEPENDENT_CODE ON)
endif(WIN32)

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
        DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/${_FILE}
        COMPONENT python)
endforeach(_FILE plotting mpi_support util)

# install the plotting.py file as an executable
configure_file(${PROJECT_BINARY_DIR}/timemory/plotting/plotting.py
    ${PROJECT_BINARY_DIR}/timemory-plotter COPYONLY)

install(FILES ${PROJECT_BINARY_DIR}/timemory-plotter
    DESTINATION ${TIMEMORY_INSTALL_BINDIR}
    COMPONENT python
    PERMISSIONS
    OWNER_EXECUTE OWNER_READ OWNER_WRITE
    GROUP_EXECUTE GROUP_READ
    WORLD_EXECUTE WORLD_READ)

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
        DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/mpi_support)
endforeach(_TYPE exe c cxx)

#----------------------------------------------------------------------------
# Install the targets and export
#

if(NOT SETUP_PY OR PYTHON_DEVELOPER_INSTALL)
    # C++ compiled libraries
    install(TARGETS ${INSTALL_LIBRARIES}
        DESTINATION ${TIMEMORY_INSTALL_LIBDIR}
        EXPORT ${PROJECT_NAME}LibraryDepends
        COMPONENT development)

    # C++ development headers
    install(FILES ${headers}
        DESTINATION ${TIMEMORY_INSTALL_INCLUDEDIR}/timemory
        COMPONENT development)

    # Install the export set for use with the install-tree
    install(EXPORT ${PROJECT_NAME}LibraryDepends
        DESTINATION ${TIMEMORY_INSTALL_CMAKEDIR}
        COMPONENT development)

endif(NOT SETUP_PY OR PYTHON_DEVELOPER_INSTALL)

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
