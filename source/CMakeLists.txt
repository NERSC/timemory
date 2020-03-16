
include_directories(BEFORE ${CMAKE_CURRENT_LIST_DIR})

#----------------------------------------------------------------------------------------#
# Locate sources and headers for this project
# - headers are included so they will show up in IDEs
#
file(GLOB c_headers                     ${CMAKE_CURRENT_LIST_DIR}/timemory/*.h)
file(GLOB cxx_headers                   ${CMAKE_CURRENT_LIST_DIR}/timemory/*.hpp)
file(GLOB cxx_headers_variadic          ${CMAKE_CURRENT_LIST_DIR}/timemory/variadic/*.hpp)
file(GLOB_RECURSE cxx_headers_backends  ${CMAKE_CURRENT_LIST_DIR}/timemory/backends/*.hpp)
file(GLOB cxx_headers_ert               ${CMAKE_CURRENT_LIST_DIR}/timemory/ert/*.hpp)
file(GLOB_RECURSE
    cxx_headers_components          ${CMAKE_CURRENT_LIST_DIR}/timemory/components/*.hpp)
file(GLOB cxx_headers_mpl           ${CMAKE_CURRENT_LIST_DIR}/timemory/mpl/*.hpp)
file(GLOB cxx_headers_extern        ${CMAKE_CURRENT_LIST_DIR}/timemory/extern/*.hpp)
file(GLOB cxx_headers_general       ${CMAKE_CURRENT_LIST_DIR}/timemory/general/*.hpp)
file(GLOB cxx_headers_runtime       ${CMAKE_CURRENT_LIST_DIR}/timemory/runtime/*.hpp)
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
file(GLOB cxx_sources               ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB_RECURSE
    cxx_extern_sources              ${CMAKE_CURRENT_LIST_DIR}/extern/*.cpp)

set(_CATEGORIES backends mpl utility variadic)
# bits folder
set(cxx_headers_category_bits)
foreach(_CATEGORY ${_CATEGORIES})
    file(GLOB cxx_headers_${_CATEGORY}_bits
        ${CMAKE_CURRENT_LIST_DIR}/timemory/${_CATEGORY}/bits/*.h
        ${CMAKE_CURRENT_LIST_DIR}/timemory/${_CATEGORY}/bits/*.hpp
        ${CMAKE_CURRENT_LIST_DIR}/timemory/${_CATEGORY}/bits/*.icpp)
    list(APPEND cxx_headers_category_bits ${cxx_headers_${_CATEGORY}_bits})
endforeach()

# all the headers
set(timemory_headers
    ${c_headers} ${cxx_headers} ${cxx_headers_variadic}
    ${cxx_headers_backends} ${cxx_headers_ert}
    ${cxx_headers_components} ${cxx_headers_mpl} ${cxx_headers_extern}
    ${cxx_headers_utility} ${cxx_headers_bits} ${cxx_headers_data}
    ${cxx_headers_category_bits}
    ${cxx_headers_general} ${cxx_headers_runtime} ${compat_headers})

if(TIMEMORY_SOURCE_GROUP)
    source_group(TREE ${CMAKE_CURRENT_LIST_DIR} FILES ${timemory_headers})
endif()

# this is only needed for windows
file(GLOB pyheaders ${CMAKE_CURRENT_LIST_DIR}/python/*.hpp)
file(GLOB pysources ${CMAKE_CURRENT_LIST_DIR}/python/*.cpp)

#----------------------------------------------------------------------------------------#
# this includes compile definitions for headers
#
add_subdirectory(timemory)

message(STATUS "Enabled  extern components libraries: ${TIMEMORY_ENABLED_COMPONENTS}")
message(STATUS "Disabled extern components libraries: ${TIMEMORY_DISABLED_COMPONENTS}")

set(_USE_EXTERN TIMEMORY_USE_EXTERN)
if(WIN32)
    set(_USE_EXTERN )
endif()

# need to compile these as CUDA or else roofline and ert does not work
if(TIMEMORY_USE_CUDA AND CMAKE_CUDA_COMPILER)
    set_source_files_properties(${cxx_extern_sources} PROPERTIES
        LANGUAGE        CUDA
        LINKER_LANGUAGE CUDA)
endif()

#----------------------------------------------------------------------------------------#
# build library setup
#
# directly compile sources
set(C_LIBRARY_SOURCES        ${c_sources}   ${c_headers})
# routinely have problems with OBJECT libraries on Windows
if(WIN32)
    set(CXX_EXTERN_SOURCES   )
    set(CXX_EXTERN_TARGETS   "")
    set(CXX_LIBRARY_SOURCES  ${cxx_sources} ${cxx_extern_sources} ${timemory_headers}
                             ${cereal_headers})
else()
    set(CXX_EXTERN_SOURCES   ${cxx_extern_sources})
    set(CXX_EXTERN_TARGETS   "$<TARGET_OBJECTS:timemory-cxx-object>")
    set(CXX_LIBRARY_SOURCES  ${cxx_sources} ${timemory_headers} ${cereal_headers})
endif()

set(_ARCH_LIBRARY )
if(TIMEMORY_USE_ARCH)
    set(_ARCH_LIBRARY timemory-arch)
endif()

#----------------------------------------------------------------------------------------#
# build the C++ libraries
#
set(_PRIVATE_INTERFACES )

set(_ANALYSIS_TOOLS timemory-analysis-tools)
if(NOT _GPERF_IN_LIBRARY)
    set(_ANALYSIS_TOOLS)
endif()

set(_LINKER_LANGUAGE CXX)

function(TIMEMORY_GET_INTERNAL_DEPENDS VAR)
    # set the depends before creating the library so it does not
    # link to itself
    set(DEPENDS)
    foreach(DEP ${ARGN})
        if(TARGET ${DEP})
            list(APPEND DEPENDS ${DEP})
        endif()
        if(TARGET ${DEP}-${LINK})
            list(APPEND DEPENDS ${DEP}-${LINK})
        endif()
        if(TARGET ${DEP}-component-${LINK})
            list(APPEND DEPENDS ${DEP}-component-${LINK})
        endif()
        if(TARGET timemory-${DEP}-${LINK})
            list(APPEND DEPENDS timemory-${DEP}-${LINK})
        endif()
        if(TARGET timemory-${DEP}-component-${LINK})
            list(APPEND DEPENDS timemory-${DEP}-component-${LINK})
        endif()
    endforeach()
    set(${VAR} "${DEPENDS}" PARENT_SCOPE)
endfunction()

#----------------------------------------------------------------------------------------#
#
#                   GLOBAL libraries which are basically used everywhere
#
#----------------------------------------------------------------------------------------#

set(TIMEMORY_SHARED_GLOBAL_LIBRARIES)
set(TIMEMORY_STATIC_GLOBAL_LIBRARIES)

foreach(COMP ${TIMEMORY_GLOBAL})

    if(WIN32 OR "${${COMP}_SOURCES}" STREQUAL "")
        continue()
    endif()

    string(TOUPPER "${COMP}" UPP_COMP)

    set(_LIB_TYPES)
    if(_BUILD_SHARED_CXX)
        list(APPEND _LIB_TYPES shared)
        set(shared_OPTIONS PIC TYPE SHARED)
        list(APPEND TIMEMORY_SHARED_GLOBAL_LIBRARIES timemory-${COMP}-shared)
    endif()

    if(_BUILD_STATIC_CXX)
        list(APPEND _LIB_TYPES static)
        set(static_OPTIONS TYPE STATIC)
        list(APPEND TIMEMORY_STATIC_GLOBAL_LIBRARIES timemory-${COMP}-static)
    endif()

    foreach(LINK ${_LIB_TYPES})
        string(TOUPPER "${LINK}" UPP_LINK)
        set(TARGET_NAME timemory-${COMP}-${LINK})

        timemory_get_internal_depends(DEPENDS ${${COMP}_DEPENDS})

        list(APPEND timemory_headers ${${COMP}_HEADERS})

        build_library(${${LINK}_OPTIONS}
            TARGET_NAME         ${TARGET_NAME}
            OUTPUT_NAME         timemory-${COMP}
            LANGUAGE            CXX
            LINKER_LANGUAGE     ${_LINKER_LANGUAGE}
            OUTPUT_DIR          ${PROJECT_BINARY_DIR}/global
            SOURCES             ${${COMP}_SOURCES}
                                ${${COMP}_HEADERS}
            CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS})

        target_link_libraries(${TARGET_NAME} PUBLIC
            timemory-headers
            timemory-vector
            timemory-external-${LINK}
            ${DEPENDS})

        target_link_libraries(${TARGET_NAME} PRIVATE
            timemory-compile-options
            timemory-develop-options
            ${_ANALYSIS_TOOLS}
            ${_ARCH_LIBRARY})

        if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            target_compile_definitions(${TARGET_NAME} PRIVATE DEBUG)
        endif()

        target_compile_definitions(${TARGET_NAME} PUBLIC  ${_USE_EXTERN})
        target_compile_definitions(${TARGET_NAME} PRIVATE
            TIMEMORY_SOURCE
            TIMEMORY_GLOBAL_SOURCE
            TIMEMORY_${UPP_COMP}_SOURCE)

        if(NOT WIN32)
            target_compile_definitions(${TARGET_NAME} PUBLIC
                TIMEMORY_USE_GLOBAL_EXTERN
                TIMEMORY_USE_${UPP_COMP}_EXTERN)
        endif()

    endforeach()

endforeach()

#----------------------------------------------------------------------------------------#
#
#                   OBJECT libraries
#
#----------------------------------------------------------------------------------------#

# routinely have problems with OBJECT libraries on Windows
if(NOT WIN32 AND (_BUILD_SHARED_CXX OR _BUILD_STATIC_CXX))
    build_library(
        PIC
        TYPE                OBJECT
        TARGET_NAME         timemory-cxx-object
        OUTPUT_NAME         timemory
        LANGUAGE            CXX
        LINKER_LANGUAGE     ${_LINKER_LANGUAGE}
        SOURCES             ${CXX_EXTERN_SOURCES}
        CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS}
        COMPILE_DEFINITIONS ${_USE_EXTERN})

    target_link_libraries(timemory-cxx-object PUBLIC
        timemory-headers
        timemory-vector
        timemory-external-shared
        timemory-external-static)

    target_link_libraries(timemory-cxx-object PRIVATE
        timemory-compile-options
        timemory-develop-options
        ${_ANALYSIS_TOOLS}
        ${_ARCH_LIBRARY})

    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        target_compile_definitions(timemory-cxx-object PRIVATE DEBUG)
    endif()

endif()

#----------------------------------------------------------------------------------------#
#
#                   COMPONENT libraries
#
#----------------------------------------------------------------------------------------#

set(TIMEMORY_SHARED_COMPONENT_LIBRARIES )
set(TIMEMORY_STATIC_COMPONENT_LIBRARIES )

foreach(COMP ${TIMEMORY_ENABLED_COMPONENTS})

    if(WIN32 OR "${${COMP}_SOURCES}" STREQUAL "")
        continue()
    endif()

    string(TOUPPER "${COMP}" UPP_COMP)

    set(_LIB_TYPES)
    if(_BUILD_SHARED_CXX)
        list(APPEND _LIB_TYPES shared)
        set(shared_OPTIONS PIC TYPE SHARED)
        list(APPEND TIMEMORY_SHARED_COMPONENT_LIBRARIES timemory-${COMP}-component-shared)
    endif()

    if(_BUILD_STATIC_CXX)
        list(APPEND _LIB_TYPES static)
        set(static_OPTIONS TYPE STATIC)
        list(APPEND TIMEMORY_STATIC_COMPONENT_LIBRARIES timemory-${COMP}-component-static)
    endif()

    set(_SOURCES ${${COMP}_SOURCES} ${${COMP}_HEADERS})

    foreach(LINK ${_LIB_TYPES})

        string(TOUPPER "${LINK}" UPP_LINK)
        set(TARGET_NAME timemory-${COMP}-component-${LINK})

        # set the depends before creating the library so it does not
        # link to itself
        timemory_get_internal_depends(DEPENDS ${${COMP}_DEPENDS})

        list(APPEND timemory_headers ${${COMP}_HEADERS})

        build_library(${${LINK}_OPTIONS}
            TARGET_NAME         ${TARGET_NAME}
            OUTPUT_NAME         timemory-${COMP}-component
            LANGUAGE            CXX
            LINKER_LANGUAGE     ${_LINKER_LANGUAGE}
            OUTPUT_DIR          ${PROJECT_BINARY_DIR}/components
            SOURCES             ${_SOURCES}
            CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS})

        target_link_libraries(${TARGET_NAME} PUBLIC
            timemory-headers
            timemory-vector
            timemory-external-${LINK}
            ${DEPENDS}
            ${TIMEMORY_${UPP_LINK}_GLOBAL_LIBRARIES})

        target_link_libraries(${TARGET_NAME} PRIVATE
            timemory-compile-options
            timemory-develop-options
            ${_ANALYSIS_TOOLS}
            ${_ARCH_LIBRARY})

        if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            target_compile_definitions(${TARGET_NAME} PRIVATE DEBUG)
        endif()

        target_compile_definitions(${TARGET_NAME} PUBLIC  ${_USE_EXTERN})
        target_compile_definitions(${TARGET_NAME} PRIVATE
            TIMEMORY_SOURCE
            TIMEMORY_COMPONENT_SOURCE
            TIMEMORY_${UPP_COMP}_SOURCE)

        if(NOT WIN32)
            target_compile_definitions(${TARGET_NAME} PUBLIC
                TIMEMORY_USE_COMPONENT_EXTERN
                TIMEMORY_USE_${UPP_COMP}_EXTERN)
        endif()
    endforeach()

endforeach()

set(TIMEMORY_COMPONENT_LIBRARIES
    ${TIMEMORY_SHARED_COMPONENT_LIBRARIES} ${TIMEMORY_STATIC_COMPONENT_LIBRARIES})

message(STATUS "TIMEMORY_COMPONENT_LIBRARIES: ${TIMEMORY_COMPONENT_LIBRARIES}")

#----------------------------------------------------------------------------------------#
#
#               EXTERN libraries which are have no subsequent dependencies
#
#----------------------------------------------------------------------------------------#

set(TIMEMORY_SHARED_EXTERN_LIBRARIES)
set(TIMEMORY_STATIC_EXTERN_LIBRARIES)

foreach(COMP ${TIMEMORY_EXTERN})

    if(WIN32 OR "${${COMP}_SOURCES}" STREQUAL "")
        continue()
    endif()

    string(TOUPPER "${COMP}" UPP_COMP)

    set(_LIB_TYPES)
    if(_BUILD_SHARED_CXX)
        list(APPEND _LIB_TYPES shared)
        set(shared_OPTIONS PIC TYPE SHARED)
        list(APPEND TIMEMORY_SHARED_EXTERN_LIBRARIES timemory-${COMP}-shared)
    endif()

    if(_BUILD_STATIC_CXX)
        list(APPEND _LIB_TYPES static)
        set(static_OPTIONS TYPE STATIC)
        list(APPEND TIMEMORY_STATIC_EXTERN_LIBRARIES timemory-${COMP}-static)
    endif()

    foreach(LINK ${_LIB_TYPES})
        string(TOUPPER "${LINK}" UPP_LINK)
        set(TARGET_NAME timemory-${COMP}-${LINK})

        timemory_get_internal_depends(DEPENDS ${${COMP}_DEPENDS})

        list(APPEND timemory_headers ${${COMP}_HEADERS})

        build_library(${${LINK}_OPTIONS}
            TARGET_NAME         ${TARGET_NAME}
            OUTPUT_NAME         timemory-${COMP}
            LANGUAGE            CXX
            LINKER_LANGUAGE     ${_LINKER_LANGUAGE}
            OUTPUT_DIR          ${PROJECT_BINARY_DIR}/extern
            SOURCES             ${${COMP}_SOURCES}
                                ${${COMP}_HEADERS}
            CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS})

        target_link_libraries(${TARGET_NAME} PUBLIC
            timemory-headers
            timemory-vector
            timemory-external-${LINK}
            ${TIMEMORY_${UPP_LINK}_COMPONENT_LIBRARIES}
            ${DEPENDS})

        target_link_libraries(${TARGET_NAME} PRIVATE
            timemory-compile-options
            timemory-develop-options
            ${_ANALYSIS_TOOLS}
            ${_ARCH_LIBRARY})

        if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            target_compile_definitions(${TARGET_NAME} PRIVATE DEBUG)
        endif()

        target_compile_definitions(${TARGET_NAME} PUBLIC  ${_USE_EXTERN})
        target_compile_definitions(${TARGET_NAME} PRIVATE
            TIMEMORY_SOURCE
            TIMEMORY_EXTERN_SOURCE
            TIMEMORY_${UPP_COMP}_SOURCE)

    endforeach()

endforeach()

#----------------------------------------------------------------------------------------#
#
#                       EXTERN CXX libraries
#
#----------------------------------------------------------------------------------------#

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
                            ${CXX_EXTERN_TARGETS}
        CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS}
        COMPILE_DEFINITIONS ${_USE_EXTERN})

    target_link_libraries(timemory-cxx-shared PUBLIC
        timemory-headers
        timemory-vector
        timemory-external-shared
        ${TIMEMORY_SHARED_COMPONENT_LIBRARIES}
        ${TIMEMORY_SHARED_EXTERN_LIBRARIES})

    target_link_libraries(timemory-cxx-shared PRIVATE
        timemory-compile-options
        timemory-develop-options
        ${_ANALYSIS_TOOLS}
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
                            ${CXX_EXTERN_TARGETS}
        CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS}
        COMPILE_DEFINITIONS ${_USE_EXTERN})

    target_link_libraries(timemory-cxx-static PUBLIC
        timemory-headers
        timemory-vector
        timemory-external-static
        ${TIMEMORY_STATIC_COMPONENT_LIBRARIES}
        ${TIMEMORY_STATIC_EXTERN_LIBRARIES})

    target_link_libraries(timemory-cxx-static PRIVATE
        timemory-compile-options
        timemory-develop-options
        ${_ANALYSIS_TOOLS}
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
#
#                       EXTERN C libraries
#
#----------------------------------------------------------------------------------------#

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
                                    ${_ANALYSIS_TOOLS}
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
                                    ${_ANALYSIS_TOOLS}
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
#
#                       TOOL executables and libraries
#
#----------------------------------------------------------------------------------------#
#
# timem wrapper tool, mpip, etc.
#
add_subdirectory(tools)

add_library(timemory-mpip-library INTERFACE)
if(TARGET timemory-mpip)
    target_compile_definitions(timemory-mpip-library INTERFACE TIMEMORY_USE_MPI_P)
    target_link_libraries(timemory-mpip-library INTERFACE timemory-mpip)
    list(APPEND TIMEMORY_COMPILED_LIBRARIES timemory-mpip)
endif()


#----------------------------------------------------------------------------------------#
#
#                       INSTALL and EXPORT
#
#----------------------------------------------------------------------------------------#

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

list(REMOVE_DUPLICATES timemory_headers)

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
#
#                       PYTHON Bindings
#
#----------------------------------------------------------------------------------------#
#
# Python bindings
#
add_subdirectory(python)

#
# install the plotting.py module as a Python executable
# named 'timemory-plotter' as C++ JSON outputs can use this
# to generate plots
#
if((TIMEMORY_BUILD_PYTHON OR TIMEMORY_USE_PYTHON) AND NOT SKBUILD)
    configure_file(${PROJECT_SOURCE_DIR}/timemory/plotting/__main__.py
        ${PROJECT_BINARY_DIR}/timemory-plotter @ONLY)

    configure_file(${PROJECT_SOURCE_DIR}/timemory/roofline/__main__.py
        ${PROJECT_BINARY_DIR}/timemory-roofline @ONLY)

    install(
        FILES
            ${PROJECT_BINARY_DIR}/timemory-plotter
            ${PROJECT_BINARY_DIR}/timemory-roofline
        DESTINATION
            ${CMAKE_INSTALL_BINDIR}
        PERMISSIONS
            OWNER_EXECUTE OWNER_READ OWNER_WRITE
            GROUP_EXECUTE GROUP_READ
            WORLD_EXECUTE WORLD_READ)
endif()

#----------------------------------------------------------------------------------------#
#
#                       Package setup
#
#----------------------------------------------------------------------------------------#

include(PackageConfigure)

#----------------------------------------------------------------------------------------#
#
#                       TESTING -- Unit tests
#
#----------------------------------------------------------------------------------------#

add_subdirectory(tests)

set(TIMEMORY_MAKEFILE_COMPILED_LIBRARIES
    ${TIMEMORY_COMPILED_LIBRARIES}
    PARENT_SCOPE)

set(TIMEMORY_MAKEFILE_INTERFACE_LIBRARIES
    ${TIMEMORY_INTERFACE_LIBRARIES}
    ${${PROJECT_NAME_UC}_INTERFACE_LIBRARIES}
    PARENT_SCOPE)
