
#----------------------------------------------------------------------------------------#
# Locate sources and headers for this project
# - headers are included so they will show up in IDEs
#
file(GLOB c_headers                 ${CMAKE_CURRENT_LIST_DIR}/timemory/*.h)
file(GLOB cxx_headers               ${CMAKE_CURRENT_LIST_DIR}/timemory/*.hpp)
file(GLOB cxx_headers_variadic      ${CMAKE_CURRENT_LIST_DIR}/timemory/variadic/*.hpp)
file(GLOB cxx_headers_backends      ${CMAKE_CURRENT_LIST_DIR}/timemory/backends/*.hpp)
file(GLOB cxx_headers_ert           ${CMAKE_CURRENT_LIST_DIR}/timemory/ert/*.hpp)
file(GLOB cxx_headers_details       ${CMAKE_CURRENT_LIST_DIR}/timemory/details/*.hpp)
file(GLOB cxx_headers_components    ${CMAKE_CURRENT_LIST_DIR}/timemory/components/*.hpp
                                    ${CMAKE_CURRENT_LIST_DIR}/timemory/components/derived/*.hpp)
file(GLOB cxx_headers_mpl           ${CMAKE_CURRENT_LIST_DIR}/timemory/mpl/*.hpp)
file(GLOB cxx_headers_extern        ${CMAKE_CURRENT_LIST_DIR}/timemory/extern/*.hpp)
file(GLOB cxx_headers_data          ${CMAKE_CURRENT_LIST_DIR}/timemory/data/*.hpp)
file(GLOB cxx_headers_utility       ${CMAKE_CURRENT_LIST_DIR}/timemory/utility/*.hpp)
file(GLOB cxx_headers_bits          ${CMAKE_CURRENT_LIST_DIR}/timemory/bits/*.hpp
                                    ${CMAKE_CURRENT_LIST_DIR}/timemory/bits/*.h
                                    ${CMAKE_CURRENT_LIST_DIR}/timemory/bits/*.icpp)
file(GLOB compat_headers            ${CMAKE_CURRENT_LIST_DIR}/timemory/compat/*.h
                                    ${CMAKE_CURRENT_LIST_DIR}/timemory/compat/*.hpp)
file(GLOB_RECURSE cereal_headers    ${CMAKE_CURRENT_LIST_DIR}/cereal/include/*.hpp
                                    ${CMAKE_CURRENT_LIST_DIR}/cereal/include/*.h
                                    ${CMAKE_CURRENT_LIST_DIR}/cereal/include/*.hh)
file(GLOB c_sources                 ${CMAKE_CURRENT_LIST_DIR}/*.c)
file(GLOB cxx_sources               ${CMAKE_CURRENT_LIST_DIR}/*.cpp
                                    ${CMAKE_CURRENT_LIST_DIR}/extern/*.cpp)

# bits folder
set(cxx_headers_category_bits)
foreach(_CATEGORY backends components ert mpl variadic utility)
    file(GLOB cxx_headers_${_CATEGORY}_bits
        ${CMAKE_CURRENT_LIST_DIR}/timemory/${_CATEGORY}/bits/*.hpp
        ${CMAKE_CURRENT_LIST_DIR}/timemory/${_CATEGORY}/bits/*.icpp)
    list(APPEND cxx_headers_category_bits ${cxx_headers_${_CATEGORY}_bits})
endforeach()

# all the headers
set(timemory_headers
    ${c_headers} ${cxx_headers} ${cxx_headers_variadic}
    ${cxx_headers_backends} ${cxx_headers_ert} ${cxx_headers_details}
    ${cxx_headers_components} ${cxx_headers_mpl} ${cxx_headers_extern}
    ${cxx_headers_utility} ${cxx_headers_bits} ${cxx_headers_data}
    ${cxx_headers_category_bits} ${compat_headers})

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
# directly compile sources
set(C_LIBRARY_SOURCES    ${c_sources}   ${c_headers})
set(CXX_LIBRARY_SOURCES  ${cxx_sources} ${timemory_headers} ${cereal_headers})

set(_ARCH_LIBRARY )
if(TIMEMORY_USE_ARCH)
    set(_ARCH_LIBRARY timemory-arch)
endif()

#----------------------------------------------------------------------------------------#
# build the C++ libraries
#
set(_EXTERN_INIT TIMEMORY_EXTERN_INIT)
set(_EXTERN_TEMP TIMEMORY_EXTERN_TEMPLATES)
if(WIN32)
    set(_EXTERN_INIT )
    set(_EXTERN_TEMP )
endif()

set(_LINKER_LANGUAGE CXX)
#if(TIMEMORY_USE_CUDA OR TIMEMORY_USE_CUPTI)
#    set(_LINKER_LANGUAGE CUDA)
#endif()

if(_BUILD_SHARED_CXX)
    build_library(
        PIC
        TYPE                SHARED
        TARGET_NAME         timemory-cxx-shared
        OUTPUT_NAME         timemory
        LANGUAGE            CXX
        LINKER_LANGUAGE     ${_LINKER_LANGUAGE}
        OUTPUT_DIR          ${PROJECT_BINARY_DIR}
        SOURCES             ${CXX_LIBRARY_SOURCES}
        CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS}
        COMPILE_DEFINITIONS ${_EXTERN_INIT} ${_EXTERN_TEMP})

    target_link_libraries(timemory-cxx-shared PUBLIC
        timemory-headers
        timemory-vector
        timemory-external-shared)

    target_link_libraries(timemory-cxx-shared PRIVATE
        timemory-compile-options
        timemory-develop-options
        timemory-analysis-tools
        ${_ARCH_LIBRARY})

    if(TIMEMORY_BUILD_C)
        target_compile_definitions(timemory-cxx-shared PRIVATE TIMEMORY_BUILD_C)
    endif()

    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        target_compile_definitions(timemory-cxx-shared PRIVATE DEBUG)
    endif()
endif()

if(_BUILD_STATIC_CXX)

    build_library(
        TYPE                STATIC
        TARGET_NAME         timemory-cxx-static
        OUTPUT_NAME         timemory
        LANGUAGE            CXX
        LINKER_LANGUAGE     ${_LINKER_LANGUAGE}
        OUTPUT_DIR          ${PROJECT_BINARY_DIR}
        SOURCES             ${CXX_LIBRARY_SOURCES}
        CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS}
        COMPILE_DEFINITIONS ${_EXTERN_INIT} ${_EXTERN_TEMP})

    target_link_libraries(timemory-cxx-static PUBLIC
        timemory-headers
        timemory-vector
        timemory-external-static)

    target_link_libraries(timemory-cxx-static PRIVATE
        timemory-compile-options
        timemory-develop-options
        timemory-analysis-tools
        ${_ARCH_LIBRARY})

    if(TIMEMORY_BUILD_C)
        target_compile_definitions(timemory-cxx-static PRIVATE TIMEMORY_BUILD_C)
    endif()

    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        target_compile_definitions(timemory-cxx-static PRIVATE DEBUG)
    endif()
endif()

if(WIN32)
    if(_BUILD_SHARED_CXX AND _BUILD_STATIC_CXX)
        # make sure on Windows, static is built first. There is something weird
        # happening here when built in parallel...
        add_dependencies(timemory-cxx-shared timemory-cxx-static)
    endif()
    if(_BUILD_SHARED_CXX)
        target_compile_definitions(timemory-cxx-shared PRIVATE _TIMEMORY_DLL PUBLIC _TIMEMORY_LINK_LIBRARY)
    endif()
    if(_BUILD_STATIC_CXX)
        target_compile_definitions(timemory-cxx-static PUBLIC  _TIMEMORY_ARCHIVE)
    endif()
endif()


#----------------------------------------------------------------------------------------#
# build the C libraries
#
if(TIMEMORY_BUILD_C)

    if(_BUILD_SHARED_CXX)
        build_library(
            PIC
            TYPE                SHARED
            TARGET_NAME         timemory-c-shared
            OUTPUT_NAME         ctimemory
            LANGUAGE            C
            LINKER_LANGUAGE     ${_LINKER_LANGUAGE}
            OUTPUT_DIR          ${PROJECT_BINARY_DIR}
            SOURCES             ${C_LIBRARY_SOURCES}
            C_COMPILE_OPTIONS   ${${PROJECT_NAME}_C_COMPILE_OPTIONS}
            LINK_LIBRARIES      timemory-headers
                                PRIVATE
                                    timemory-compile-options
                                    timemory-develop-options
                                    timemory-cxx-shared
                                    timemory-external-shared
                                    timemory-analysis-tools
                                    ${_ARCH_LIBRARY})
        if(WIN32)
            target_compile_definitions(timemory-c-shared PRIVATE _TIMEMORY_DLL)
        endif()
    endif()

    if(_BUILD_STATIC_CXX)

        build_library(
            TYPE                STATIC
            TARGET_NAME         timemory-c-static
            OUTPUT_NAME         ctimemory
            LANGUAGE            C
            LINKER_LANGUAGE     ${_LINKER_LANGUAGE}
            OUTPUT_DIR          ${PROJECT_BINARY_DIR}
            SOURCES             ${C_LIBRARY_SOURCES}
            C_COMPILE_OPTIONS   ${${PROJECT_NAME}_C_COMPILE_OPTIONS}
            LINK_LIBRARIES      timemory-headers
                                PRIVATE
                                    timemory-compile-options
                                    timemory-develop-options
                                    timemory-cxx-static
                                    timemory-external-static
                                    timemory-analysis-tools
                                    ${_ARCH_LIBRARY})
        if(WIN32)
            target_compile_definitions(timemory-c-static PUBLIC  _TIMEMORY_ARCHIVE)
        endif()
    endif()

    if(WIN32 AND _BUILD_SHARED_CXX AND _BUILD_STATIC_CXX)
        add_dependencies(timemory-c-shared timemory-c-static)
    endif()
endif()


#----------------------------------------------------------------------------------------#
# Install the targets and export
#

# C/C++ compiled libraries
install(
    TARGETS ${TIMEMORY_COMPILED_LIBRARIES}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    EXPORT ${PROJECT_NAME}-library-depends)

# C/C++ development headers
macro(INSTALL_HEADER_FILES)
    foreach(_header ${ARGN})
        file(RELATIVE_PATH _relative ${CMAKE_CURRENT_LIST_DIR} ${_header})
        get_filename_component(_destpath ${_relative} DIRECTORY)
        install(FILES ${_header} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${_destpath})
    endforeach()
endmacro()

install_header_files(${timemory_headers})

install(FILES ${cxx_headers_backend}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/timemory/backends
    COMPONENT development)

# Install the export set for use with the install-tree
install(EXPORT ${PROJECT_NAME}-library-depends
    DESTINATION ${CMAKE_INSTALL_CONFIGDIR}
    NAMESPACE ${PROJECT_NAME}::)

export(
    TARGETS
        ${TIMEMORY_COMPILED_LIBRARIES}
        ${TIMEMORY_INTERFACE_LIBRARIES}
        ${TIMEMORY_ADDITIONAL_EXPORT_TARGETS}
    NAMESPACE
        ${PROJECT_NAME}::
    FILE
        ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-build.cmake)


#----------------------------------------------------------------------------------------#
# timem wrapper tool, mpip, etc.
#
add_subdirectory(tools)


#----------------------------------------------------------------------------------------#
# Python bindings
#
if(TIMEMORY_BUILD_PYTHON)
    add_subdirectory(python)
endif()

#----------------------------------------------------------------------------------------#
# Preload library tool
#
# add_subdirectory(preload)

#----------------------------------------------------------------------------------------#
# install the plotting.py module as a Python executable
# named 'timemory-plotter' as C++ JSON outputs can use this
# to generate plots
#
if(TIMEMORY_BUILD_PYTHON AND NOT SKBUILD)
    configure_file(${PROJECT_SOURCE_DIR}/timemory/plotting/__main__.py
        ${PROJECT_BINARY_DIR}/timemory-plotter @ONLY)

    install(FILES ${PROJECT_BINARY_DIR}/timemory-plotter
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        PERMISSIONS
        OWNER_EXECUTE OWNER_READ OWNER_WRITE
        GROUP_EXECUTE GROUP_READ
        WORLD_EXECUTE WORLD_READ)
endif()

#----------------------------------------------------------------------------------------#
#   Package setup
#----------------------------------------------------------------------------------------#

include(PackageConfigure)

#----------------------------------------------------------------------------------------#
#   Unit tests
#----------------------------------------------------------------------------------------#

add_subdirectory(tests)
