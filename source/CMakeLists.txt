
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
file(GLOB c_sources ctimemory.c)
file(GLOB cxx_headers ${CMAKE_CURRENT_LIST_DIR}/timemory/*.hpp)
file(GLOB cxx_headers_impl ${CMAKE_CURRENT_LIST_DIR}/timemory/impl/*.icpp)
file(GLOB cxx_sources ctimemory.cpp)
# this is only needed for windows
file(GLOB pyheaders ${CMAKE_CURRENT_LIST_DIR}/python/*.hpp)
file(GLOB pysources ${CMAKE_CURRENT_LIST_DIR}/python/*.cpp)

# libraries to install
set(INSTALL_LIBRARIES )

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
set(CXX_LIBRARY_SOURCES ${cxx_sources} ${cxx_headers} ${cxx_headers_impl} ${cereal_headers})

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
elseif(XCODE)
    set(_OUTPUT_DIR ${PROJECT_BINARY_DIR})
else()
    set(_OUTPUT_DIR ${PROJECT_BINARY_DIR}/timemory/lib)
endif()



#------------------------------------------------------------------------------#
# build the C++ libraries
#

if(XCODE)
    set(CXX_OBJECT_SOURCES ${CXX_LIBRARY_SOURCES})
else()
    build_library(
        PIC
        TYPE                OBJECT
        TARGET_NAME         ${LIBNAME}-cxx-library-object
        OUTPUT_NAME         ${LIBNAME}
        LANGUAGE            CXX
        LINKER_LANGUAGE     CXX
        OUTPUT_DIR          ${_OUTPUT_DIR} # ignored on Windows
        COMPILE_DEFINITIONS ${${PROJECT_NAME}_DEFINITIONS}
        SOURCES             ${CXX_LIBRARY_SOURCES}
        LINK_LIBRARIES      ${EXTERNAL_LIBRARIES}
        INCLUDE_DIRECTORIES ${${PROJECT_NAME}_TARGET_INCLUDE_DIRS})
    set(CXX_OBJECT_SOURCES $<TARGET_OBJECTS:${LIBNAME}-cxx-library-object>)
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
    SOURCES             ${CXX_OBJECT_SOURCES}
    LINK_LIBRARIES      ${EXTERNAL_LIBRARIES}
    INCLUDE_DIRECTORIES ${${PROJECT_NAME}_TARGET_INCLUDE_DIRS})

build_library(
    PIC
    TYPE                STATIC
    TARGET_NAME         ${LIBNAME}-cxx-library-static
    OUTPUT_NAME         ${LIBNAME}
    LANGUAGE            CXX
    LINKER_LANGUAGE     CXX
    OUTPUT_DIR          ${_OUTPUT_DIR} # ignored on Windows
    COMPILE_DEFINITIONS ${${PROJECT_NAME}_DEFINITIONS}
    SOURCES             ${CXX_OBJECT_SOURCES}
    LINK_LIBRARIES      ${EXTERNAL_LIBRARIES}
    INCLUDE_DIRECTORIES ${${PROJECT_NAME}_TARGET_INCLUDE_DIRS})


#------------------------------------------------------------------------------#
# build the C libraries
#
if(TIMEMORY_BUILD_C)

    if(XCODE)
        set(C_OBJECT_SOURCES ${C_LIBRARY_SOURCES})
    else()
        build_library(
            PIC
            TYPE                OBJECT
            TARGET_NAME         ${LIBNAME}-c-library-object
            OUTPUT_NAME         c${LIBNAME}
            LANGUAGE            C
            LINKER_LANGUAGE     CXX
            OUTPUT_DIR          ${_OUTPUT_DIR} # ignored on Windows
            COMPILE_DEFINITIONS ${${PROJECT_NAME}_DEFINITIONS}
            SOURCES             ${C_LIBRARY_SOURCES}
            LINK_LIBRARIES      ${LIBNAME}-cxx-library
            INCLUDE_DIRECTORIES ${${PROJECT_NAME}_TARGET_INCLUDE_DIRS})

        set(C_OBJECT_SOURCES   $<TARGET_OBJECTS:${LIBNAME}-c-library-object>)
    endif()

    build_library(
        PIC
        TYPE                SHARED
        TARGET_NAME         ${LIBNAME}-c-library
        OUTPUT_NAME         c${LIBNAME}
        LANGUAGE            C
        LINKER_LANGUAGE     CXX
        OUTPUT_DIR          ${_OUTPUT_DIR} # ignored on Windows
        COMPILE_DEFINITIONS ${${PROJECT_NAME}_DEFINITIONS}
        SOURCES             ${C_OBJECT_SOURCES}
        LINK_LIBRARIES      ${LIBNAME}-cxx-library
        INCLUDE_DIRECTORIES ${${PROJECT_NAME}_TARGET_INCLUDE_DIRS})

    build_library(
        PIC
        TYPE                STATIC
        TARGET_NAME         ${LIBNAME}-c-library-static
        OUTPUT_NAME         c${LIBNAME}
        LANGUAGE            C
        LINKER_LANGUAGE     CXX
        OUTPUT_DIR          ${_OUTPUT_DIR} # ignored on Windows
        COMPILE_DEFINITIONS ${${PROJECT_NAME}_DEFINITIONS}
        SOURCES             ${C_OBJECT_SOURCES}
        LINK_LIBRARIES      ${LIBNAME}-cxx-library-static
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

install(FILES ${cxx_headers_impl}
    DESTINATION ${TIMEMORY_INSTALL_INCLUDEDIR}/timemory/impl
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
