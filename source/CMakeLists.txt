
include(CMakeParseArguments)
include_directories(${CMAKE_CURRENT_LIST_DIR})

################################################################################
#
#        Cereal
#
################################################################################

set(DEV_WARNINGS ${CMAKE_SUPPRESS_DEVELOPER_WARNINGS})
# this gets annoying
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)

# add cereal
if(NOT TIMEMORY_SETUP_PY OR TIMEMORY_DEVELOPER_INSTALL)
    add_subdirectory(cereal)
endif(NOT TIMEMORY_SETUP_PY OR TIMEMORY_DEVELOPER_INSTALL)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ${DEV_WARNINGS} CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)

include_directories(SYSTEM ${CMAKE_CURRENT_LIST_DIR}/cereal/include)
file(GLOB_RECURSE cereal_headers ${CMAKE_CURRENT_LIST_DIR}/cereal/include/*.hpp)

################################################################################
#
#        TiMemory (C++)
#
################################################################################

#------------------------------------------------------------------------------#
# Locate sources and headers for this project
# - headers are included so they will show up in IDEs
file(GLOB cxx_sources ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB cxx_headers ${CMAKE_CURRENT_LIST_DIR}/timemory/*.hpp)
file(GLOB c_sources ${CMAKE_CURRENT_LIST_DIR}/*.c)
file(GLOB c_headers ${CMAKE_CURRENT_LIST_DIR}/timemory/*.h)
# this is only needed for windows
file(GLOB pysources ${CMAKE_CURRENT_LIST_DIR}/python/*.cpp)
# set the languages
set_source_files_properties(${c_sources} PROPERTIES LANGUAGE C)
set_source_files_properties(${cxx_sources} PROPERTIES LANGUAGE CXX)

# libraries to install
set(INSTALL_LIBRARIES )


#------------------------------------------------------------------------------#
# macro to build a library of type: shared, static, object
#
macro(BUILD_LIBRARY)

    set(_options    PIC)                    # options
    set(_onevalue   TYPE                    # single-value
                    OUTPUT_NAME
                    TARGET_NAME
                    OUTPUT_DIR
                    LANGUAGE
                    LINKER_LANGUAGE)
    set(_multival   SOURCES                 # multi-value
                    LINK_LIBRARIES
                    COMPILE_DEFINITIONS)

    cmake_parse_arguments(
        LIBRARY "${_options}" "${_onevalue}" "${_multival}" ${ARGN})

    add_library(
        ${LIBRARY_TARGET_NAME} ${LIBRARY_TYPE} ${LIBRARY_SOURCES})
    target_link_libraries(
        ${LIBRARY_TARGET_NAME} PUBLIC ${LIBRARY_LINK_LIBRARIES})

    # Windows
	if(WIN32)

        set_target_properties(
            ${LIBRARY_TARGET_NAME}          PROPERTIES
            OUTPUT_NAME                     lib${LIBRARY_OUTPUT_NAME}
            COMPILE_DEFINITIONS             ${LIBRARY_COMPILE_DEFINITIONS}
            LANGUAGE                        ${LIBRARY_LANGUAGE}
            LINKER_LANGUAGE                 ${LIBRARY_LINKER_LANGUAGE}
            POSITION_INDEPENDENT_CODE       ${LIBRARY_PIC})

	else(WIN32)

        if(XCODE)

            # xcode has issue with "cmake_symlink_library"
            set_target_properties(
                ${LIBRARY_TARGET_NAME}      PROPERTIES
                OUTPUT_NAME                 ${LIBRARY_OUTPUT_NAME}
                LIBRARY_OUTPUT_DIRECTORY    ${LIBRARY_OUTPUT_DIR}
                LANGUAGE                    ${LIBRARY_LANGUAGE}
                LINKER_LANGUAGE             ${LIBRARY_LINKER_LANGUAGE}
                POSITION_INDEPENDENT_CODE   ${LIBRARY_PIC})

        else(XCODE)

            set_target_properties(
                ${LIBRARY_TARGET_NAME}      PROPERTIES
                OUTPUT_NAME                 ${LIBRARY_OUTPUT_NAME}
                VERSION                     ${PROJECT_VERSION}
                SOVERSION                   ${PROJECT_VERSION_MAJOR}
                LIBRARY_OUTPUT_DIRECTORY    ${LIBRARY_OUTPUT_DIR}
                LANGUAGE                    ${LIBRARY_LANGUAGE}
                LINKER_LANGUAGE             ${LIBRARY_LINKER_LANGUAGE}
                POSITION_INDEPENDENT_CODE   ${LIBRARY_PIC})

        endif(XCODE)

    endif(WIN32)

    list(APPEND INSTALL_LIBRARIES ${LIBRARY_TARGET_NAME})

endmacro(BUILD_LIBRARY)


#------------------------------------------------------------------------------#
# this includes compile definitions for headers
#
add_subdirectory(timemory)


#------------------------------------------------------------------------------#
# build library setup
#
set(_COMPILE_DEFS )
set(_OUTPUT_DIR )
# directly compile sources
set(C_LIBRARY_SOURCES   ${c_sources}   ${c_headers})
set(CXX_LIBRARY_SOURCES ${cxx_sources} ${cxx_headers} ${cereal_headers})

# if Windows, set _COMPILE_DEFS and possibly _ARCHIVE_SUFFIX
if(WIN32)
    if(BUILD_SHARED_LIBS)
        set(_COMPILE_DEFS       _TIMEMORY_DLL)
    else(BUILD_SHARED_LIBS)
        set(_COMPILE_DEFS       _TIMEMORY_ARCHIVE)
    endif(BUILD_SHARED_LIBS)
else(WIN32)
    set(_OUTPUT_DIR ${PROJECT_BINARY_DIR}/timemory/lib)
endif(WIN32)

# Set source file compile definitions
if(NOT "${_COMPILE_DEFS}" STREQUAL "")
    set_source_files_properties(
        ${C_LIBRARY_SOURCES} ${CXX_LIBRARY_SOURCES} ${pysources}
        PROPERTIES COMPILE_DEFINITIONS ${_COMPILE_DEFS})
endif(NOT "${_COMPILE_DEFS}" STREQUAL "")


#------------------------------------------------------------------------------#
# build the libraries
#
if(BUILD_SHARED_LIBS)
    set(_TYPE SHARED)
else(BUILD_SHARED_LIBS)
    set(_TYPE STATIC)
endif(BUILD_SHARED_LIBS)

build_library(
    PIC
    TYPE                ${_TYPE}
    TARGET_NAME         ${LIBNAME}-cxx-library
    OUTPUT_NAME         ${LIBNAME}
    LANGUAGE            CXX
    LINKER_LANGUAGE     CXX
    OUTPUT_DIR          ${_OUTPUT_DIR} # ignored on Windows
    COMPILE_DEFINITIONS ${_COMPILE_DEFS}
    SOURCES             ${CXX_LIBRARY_SOURCES}
    LINK_LIBRARIES      ${EXTERNAL_LIBRARIES})

build_library(
    PIC
    TYPE                ${_TYPE}
    TARGET_NAME         ${LIBNAME}-c-library
    OUTPUT_NAME         c${LIBNAME}
    LANGUAGE            C
    LINKER_LANGUAGE     CXX
    OUTPUT_DIR          ${CMAKE_BINARY_DIR} # ignored on Windows
    COMPILE_DEFINITIONS ${_COMPILE_DEFS}
    SOURCES             ${C_LIBRARY_SOURCES}
    LINK_LIBRARIES      ${LIBNAME}-cxx-library)


#------------------------------------------------------------------------------#
# Install the targets and export
#

# C/C++ compiled libraries
install(TARGETS ${INSTALL_LIBRARIES}
    DESTINATION ${TIMEMORY_INSTALL_LIBDIR}
    EXPORT ${PROJECT_NAME}LibraryDepends
    COMPONENT development)

# C/C++ development headers
install(FILES ${cxx_headers} ${c_headers}
    DESTINATION ${TIMEMORY_INSTALL_INCLUDEDIR}/timemory
    COMPONENT development)

# Install the export set for use with the install-tree
install(EXPORT ${PROJECT_NAME}LibraryDepends
    DESTINATION ${TIMEMORY_INSTALL_CMAKEDIR}
    COMPONENT development)

export(TARGETS ${INSTALL_LIBRARIES}
    FILE ${CMAKE_BINARY_DIR}/TiMemoryBuild.cmake)


#------------------------------------------------------------------------------#
# timem wrapper tool
#
add_subdirectory(tools)


#------------------------------------------------------------------------------#
# Python bindings
#
if(TIMEMORY_USE_PYTHON_BINDING)

    add_subdirectory(python)

    if(BUILD_SHARED_LIBS)
        set(LINK_TYPE shared)
        install(TARGETS ${LIBNAME}-cxx-library
            DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/lib
            COMPONENT python)
    endif(BUILD_SHARED_LIBS)

endif(TIMEMORY_USE_PYTHON_BINDING)


#------------------------------------------------------------------------------#
# install the plotting.py module as a Python executable
# named 'timemory-plotter' as C++ JSON outputs can use this
# to generate plots
#
configure_file(${PROJECT_SOURCE_DIR}/timemory/plotting/plotting.py
    ${PROJECT_BINARY_DIR}/timemory-plotter COPYONLY)

install(FILES ${PROJECT_BINARY_DIR}/timemory-plotter
    DESTINATION ${TIMEMORY_INSTALL_BINDIR}
    COMPONENT development
    PERMISSIONS
    OWNER_EXECUTE OWNER_READ OWNER_WRITE
    GROUP_EXECUTE GROUP_READ
    WORLD_EXECUTE WORLD_READ)
