
include(CMakeParseArguments)


#----------------------------------------------------------------------------------------#
# Locate sources and headers for this project
# - headers are included so they will show up in IDEs
#
file(GLOB c_headers ${CMAKE_CURRENT_LIST_DIR}/timemory/*.h)
file(GLOB c_sources ctimemory.c)
file(GLOB cxx_headers ${CMAKE_CURRENT_LIST_DIR}/timemory/*.hpp)
file(GLOB cxx_headers_impl ${CMAKE_CURRENT_LIST_DIR}/timemory/impl/*.icpp)
file(GLOB cxx_sources ctimemory.cpp)
file(GLOB_RECURSE cereal_headers ${CMAKE_CURRENT_LIST_DIR}/cereal/include/*.hpp)
# this is only needed for windows
file(GLOB pyheaders ${CMAKE_CURRENT_LIST_DIR}/python/*.hpp)
file(GLOB pysources ${CMAKE_CURRENT_LIST_DIR}/python/*.cpp)


#----------------------------------------------------------------------------------------#
# this includes compile definitions for headers
#
add_subdirectory(timemory)


#----------------------------------------------------------------------------------------#
# build library setup
#
set(_OUTPUT_DIR ${PROJECT_BINARY_DIR}/timemory/lib)
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
    set_source_files_properties(${c_sources} ${cxx_sources}
        PROPERTIES COMPILE_DEFINITIONS ${PRIVATE_COMPILE_DEFINITIONS})
elseif(XCODE)
    set(_OUTPUT_DIR ${PROJECT_BINARY_DIR})
endif()



#----------------------------------------------------------------------------------------#
# build the C++ libraries
#

build_library(
    PIC
    TYPE                SHARED
    TARGET_NAME         ${LIBNAME}-cxx-library-shared
    OUTPUT_NAME         ${LIBNAME}
    LANGUAGE            CXX
    LINKER_LANGUAGE     CXX
    OUTPUT_DIR          ${_OUTPUT_DIR} # ignored on Windows
    SOURCES             ${CXX_LIBRARY_SOURCES}
    LINK_LIBRARIES      ${EXTERNAL_LIBRARIES})

build_library(
    PIC
    TYPE                STATIC
    TARGET_NAME         ${LIBNAME}-cxx-library-static
    OUTPUT_NAME         ${LIBNAME}
    LANGUAGE            CXX
    LINKER_LANGUAGE     CXX
    OUTPUT_DIR          ${_OUTPUT_DIR} # ignored on Windows
    SOURCES             ${CXX_LIBRARY_SOURCES}
    LINK_LIBRARIES      ${EXTERNAL_LIBRARIES})

add_library(${LIBNAME}-cxx-library INTERFACE)
target_link_libraries(${LIBNAME}-cxx-library INTERFACE ${LIBNAME}-cxx-library-shared)
list(APPEND INSTALL_LIBRARIES ${LIBNAME}-cxx-library)

target_link_libraries(timemory-extern-templates INTERFACE timemory-cxx-library)
target_link_libraries(timemory-shared-extern-templates INTERFACE timemory-cxx-library-shared)
target_link_libraries(timemory-static-extern-templates INTERFACE timemory-cxx-library-static)

if(TIMEMORY_BUILD_EXTERN_TEMPLATES)
    target_compile_definitions(${LIBNAME}-cxx-library-shared    PRIVATE     TIMEMORY_BUILD_EXTERN_TEMPLATES)
    target_compile_definitions(${LIBNAME}-cxx-library-static    PRIVATE     TIMEMORY_BUILD_EXTERN_TEMPLATES)
    target_compile_definitions(timemory-extern-templates        INTERFACE   TIMEMORY_EXTERN_TEMPLATES)
    target_compile_definitions(timemory-shared-extern-templates INTERFACE   TIMEMORY_EXTERN_TEMPLATES)
    target_compile_definitions(timemory-static-extern-templates INTERFACE   TIMEMORY_EXTERN_TEMPLATES)
endif()

#----------------------------------------------------------------------------------------#
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
            SOURCES             ${C_LIBRARY_SOURCES}
            LINK_LIBRARIES      ${LIBNAME}-cxx-library-shared timemory-cereal)

        set(C_OBJECT_SOURCES   $<TARGET_OBJECTS:${LIBNAME}-c-library-object>)
    endif()

    build_library(
        PIC
        TYPE                SHARED
        TARGET_NAME         ${LIBNAME}-c-library-shared
        OUTPUT_NAME         c${LIBNAME}
        LANGUAGE            C
        LINKER_LANGUAGE     CXX
        OUTPUT_DIR          ${_OUTPUT_DIR} # ignored on Windows
        SOURCES             ${C_OBJECT_SOURCES}
        LINK_LIBRARIES      ${LIBNAME}-cxx-library-shared)

    build_library(
        PIC
        TYPE                STATIC
        TARGET_NAME         ${LIBNAME}-c-library-static
        OUTPUT_NAME         c${LIBNAME}
        LANGUAGE            C
        LINKER_LANGUAGE     CXX
        OUTPUT_DIR          ${_OUTPUT_DIR} # ignored on Windows
        SOURCES             ${C_OBJECT_SOURCES}
        LINK_LIBRARIES      ${LIBNAME}-cxx-library-static)

    add_library(${LIBNAME}-c-library INTERFACE)
    target_link_libraries(${LIBNAME}-c-library INTERFACE ${LIBNAME}-c-library-shared)
    list(APPEND INSTALL_LIBRARIES ${LIBNAME}-c-library)

endif()


#----------------------------------------------------------------------------------------#
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


#----------------------------------------------------------------------------------------#
# timem wrapper tool
#
add_subdirectory(tools)


#----------------------------------------------------------------------------------------#
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


#----------------------------------------------------------------------------------------#
# install the plotting.py module as a Python executable
# named 'timemory-plotter' as C++ JSON outputs can use this
# to generate plots
#
if(TIMEMORY_BUILD_PYTHON)
    configure_file(${PROJECT_SOURCE_DIR}/timemory/plotting/__main__.py
        ${PROJECT_BINARY_DIR}/timemory-plotter @ONLY)

    install(FILES ${PROJECT_BINARY_DIR}/timemory-plotter
        DESTINATION ${TIMEMORY_INSTALL_BINDIR}
        COMPONENT development
        PERMISSIONS
        OWNER_EXECUTE OWNER_READ OWNER_WRITE
        GROUP_EXECUTE GROUP_READ
        WORLD_EXECUTE WORLD_READ)
endif()
