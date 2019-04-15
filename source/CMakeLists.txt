
include(CMakeParseArguments)
list(APPEND ${PROJECT_NAME}_TARGET_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/cereal/include)

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
endif()

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ${DEV_WARNINGS} CACHE BOOL
    "Suppress Warnings that are meant for the author of the CMakeLists.txt files"
    FORCE)

file(GLOB_RECURSE cereal_headers ${CMAKE_CURRENT_LIST_DIR}/cereal/include/*.hpp)

################################################################################
#
#        TiMemory (C++)
#
################################################################################

#------------------------------------------------------------------------------#
# Locate sources and headers for this project
# - headers are included so they will show up in IDEs
file(GLOB c_headers ${CMAKE_CURRENT_LIST_DIR}/timemory/*.h)
file(GLOB c_sources ctimemory.c ctimemory.cpp)
file(GLOB cxx_headers ${CMAKE_CURRENT_LIST_DIR}/timemory/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/timemory/*.icpp)
file(GLOB cxx_sources manager.cpp)
# this is only needed for windows
file(GLOB pyheaders ${CMAKE_CURRENT_LIST_DIR}/python/*.hpp)
file(GLOB pysources ${CMAKE_CURRENT_LIST_DIR}/python/*.cpp)

# libraries to install
set(INSTALL_LIBRARIES )


#------------------------------------------------------------------------------#
# macro to build a library of type: shared, static, object
#
macro(BUILD_LIBRARY)

    # options
    set(_options    PIC)
    # single-value
    set(_onevalue   TYPE
                    OUTPUT_NAME
                    TARGET_NAME
                    OUTPUT_DIR
                    LANGUAGE
                    LINKER_LANGUAGE)
    # multi-value
    set(_multival   SOURCES
                    LINK_LIBRARIES
                    COMPILE_DEFINITIONS
                    INCLUDE_DIRECTORIES
                    EXTRA_PROPERTIES)

    cmake_parse_arguments(
        LIBRARY "${_options}" "${_onevalue}" "${_multival}" ${ARGN})

    if(NOT WIN32 AND NOT XCODE)
        list(APPEND LIBRARY_EXTRA_PROPERTIES
            VERSION                     ${PROJECT_VERSION}
            SOVERSION                   ${PROJECT_VERSION_MAJOR})
    endif()

    if(NOT WIN32)
        set(LIB_PREFIX )
        list(APPEND LIBRARY_EXTRA_PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY    ${LIBRARY_OUTPUT_DIR})
    else()
        set(LIB_PREFIX lib)
    endif()

    add_library(${LIBRARY_TARGET_NAME}
        ${LIBRARY_TYPE} ${LIBRARY_SOURCES})
    target_include_directories(${LIBRARY_TARGET_NAME}
        PUBLIC ${EXTERNAL_INCLUDE_DIRS}
        PRIVATE ${${PROJECT_NAME}_TARGET_INCLUDE_DIRS})
    target_link_libraries(${LIBRARY_TARGET_NAME}
        PUBLIC ${LIBRARY_LINK_LIBRARIES})
    target_compile_definitions(${LIBRARY_TARGET_NAME}
        PUBLIC ${LIBRARY_COMPILE_DEFINITIONS})
    target_compile_options(${LIBRARY_TARGET_NAME}
        PRIVATE
            $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
            $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>)

    set_target_properties(
        ${LIBRARY_TARGET_NAME}          PROPERTIES
        OUTPUT_NAME                     ${LIB_PREFIX}${LIBRARY_OUTPUT_NAME}
        LANGUAGE                        ${LIBRARY_LANGUAGE}
        LINKER_LANGUAGE                 ${LIBRARY_LINKER_LANGUAGE}
        POSITION_INDEPENDENT_CODE       ${LIBRARY_PIC}
        ${LIBRARY_EXTRA_PROPERTIES})

    list(APPEND INSTALL_LIBRARIES ${LIBRARY_TARGET_NAME})

endmacro(BUILD_LIBRARY)


#------------------------------------------------------------------------------#
# this includes compile definitions for headers
#
add_subdirectory(timemory)


#------------------------------------------------------------------------------#
# build library setup
#
set(_OUTPUT_DIR )
# directly compile sources
set(C_LIBRARY_SOURCES   ${c_sources}   ${c_headers})
set(CXX_LIBRARY_SOURCES ${cxx_sources} ${cxx_headers} ${cereal_headers})

# if Windows, set _COMPILE_DEFS and possibly _ARCHIVE_SUFFIX
if(WIN32)
    if(BUILD_SHARED_LIBS)
        set(PRIVATE_COMPILE_DEFINITIONS         _TIMEMORY_DLL)
    else()
        list(APPEND ${PROJECT_NAME}_DEFINITIONS _TIMEMORY_ARCHIVE)
        set(PRIVATE_COMPILE_DEFINITIONS         _TIMEMORY_ARCHIVE)
    endif()

    # Set source file compile definitions
    set_source_files_properties(
        ${c_sources} ${CXX_LIBRARY_SOURCES}
        PROPERTIES COMPILE_DEFINITIONS ${PRIVATE_COMPILE_DEFINITIONS})
else()
    set(_OUTPUT_DIR ${PROJECT_BINARY_DIR}/timemory/lib)
endif()



#------------------------------------------------------------------------------#
# build the libraries
#
set(_TYPE SHARED) # default
if(NOT BUILD_SHARED_LIBS)
    set(_TYPE STATIC)
endif()

build_library(
    PIC
    TYPE                SHARED
    TARGET_NAME         ${LIBNAME}-cxx-library
    OUTPUT_NAME         ${LIBNAME}
    LANGUAGE            CXX
    LINKER_LANGUAGE     CXX
    OUTPUT_DIR          ${_OUTPUT_DIR} # ignored on Windows
    COMPILE_DEFINITIONS ${${PROJECT_NAME}_DEFINITIONS}
    SOURCES             ${CXX_LIBRARY_SOURCES}
    LINK_LIBRARIES      ${EXTERNAL_LIBRARIES}
    INCLUDE_DIRECTORIES ${${PROJECT_NAME}_TARGET_INCLUDE_DIRS})

if(BUILD_SHARED_LIBS)
    build_library(
        PIC
        TYPE                STATIC
        TARGET_NAME         ${LIBNAME}-cxx-library-static
        OUTPUT_NAME         ${LIBNAME}
        LANGUAGE            CXX
        LINKER_LANGUAGE     CXX
        OUTPUT_DIR          ${_OUTPUT_DIR} # ignored on Windows
        COMPILE_DEFINITIONS ${${PROJECT_NAME}_DEFINITIONS}
        SOURCES             ${CXX_LIBRARY_SOURCES}
        LINK_LIBRARIES      ${EXTERNAL_LIBRARIES}
        INCLUDE_DIRECTORIES ${${PROJECT_NAME}_TARGET_INCLUDE_DIRS})
else()
    add_library(${LIBNAME}-cxx-library-static ALIAS ${LIBNAME}-cxx-library)
endif()

if(TIMEMORY_BUILD_C)
    build_library(
        PIC
        TYPE                ${_TYPE}
        TARGET_NAME         ${LIBNAME}-c-library
        OUTPUT_NAME         c${LIBNAME}
        LANGUAGE            C
        LINKER_LANGUAGE     CXX
        OUTPUT_DIR          ${_OUTPUT_DIR} # ignored on Windows
        COMPILE_DEFINITIONS ${${PROJECT_NAME}_DEFINITIONS}
        SOURCES             ${C_LIBRARY_SOURCES}
        LINK_LIBRARIES      ${LIBNAME}-cxx-library
        INCLUDE_DIRECTORIES ${${PROJECT_NAME}_TARGET_INCLUDE_DIRS})
endif()

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
if(TIMEMORY_BUILD_PYTHON)

    add_subdirectory(python)

    if(BUILD_SHARED_LIBS)
        set(LINK_TYPE shared)
        install(TARGETS ${LIBNAME}-cxx-library
            DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/lib
            COMPONENT python)
    endif(BUILD_SHARED_LIBS)

endif()


#------------------------------------------------------------------------------#
# install the plotting.py module as a Python executable
# named 'timemory-plotter' as C++ JSON outputs can use this
# to generate plots
#
configure_file(${PROJECT_SOURCE_DIR}/timemory/plotting/__main__.py
    ${PROJECT_BINARY_DIR}/timemory-plotter @ONLY)

install(FILES ${PROJECT_BINARY_DIR}/timemory-plotter
    DESTINATION ${TIMEMORY_INSTALL_BINDIR}
    COMPONENT development
    PERMISSIONS
    OWNER_EXECUTE OWNER_READ OWNER_WRITE
    GROUP_EXECUTE GROUP_READ
    WORLD_EXECUTE WORLD_READ)
