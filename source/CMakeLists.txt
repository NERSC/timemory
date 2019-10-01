
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
file(GLOB cxx_headers_components    ${CMAKE_CURRENT_LIST_DIR}/timemory/components/*.hpp)
file(GLOB cxx_headers_mpl           ${CMAKE_CURRENT_LIST_DIR}/timemory/mpl/*.hpp)
file(GLOB cxx_headers_templates     ${CMAKE_CURRENT_LIST_DIR}/timemory/templates/*.hpp)
file(GLOB cxx_headers_utility       ${CMAKE_CURRENT_LIST_DIR}/timemory/utility/*.hpp)
file(GLOB cxx_headers_bits          ${CMAKE_CURRENT_LIST_DIR}/timemory/bits/*.hpp
                                    ${CMAKE_CURRENT_LIST_DIR}/timemory/bits/*.h)
file(GLOB_RECURSE cereal_headers    ${CMAKE_CURRENT_LIST_DIR}/cereal/include/*.hpp
                                    ${CMAKE_CURRENT_LIST_DIR}/cereal/include/*.h
                                    ${CMAKE_CURRENT_LIST_DIR}/cereal/include/*.hh)
file(GLOB c_sources                 ${CMAKE_CURRENT_LIST_DIR}/ctimemory.c)
file(GLOB cxx_sources               ${CMAKE_CURRENT_LIST_DIR}/ctimemory.cpp
                                    ${CMAKE_CURRENT_LIST_DIR}/library.cpp
                                    ${CMAKE_CURRENT_LIST_DIR}/init.cpp
                                    ${CMAKE_CURRENT_LIST_DIR}/auto_timer_extern.cpp
                                    ${CMAKE_CURRENT_LIST_DIR}/native_extern.cpp)

# all the headers
set(timemory_headers
    ${c_headers} ${cxx_headers} ${cxx_headers_variadic}
    ${cxx_headers_backends} ${cxx_headers_ert} ${cxx_headers_details}
    ${cxx_headers_components} ${cxx_headers_mpl} ${cxx_headers_templates}
    ${cxx_headers_utility} ${cxx_headers_bits})
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

if(_BUILD_SHARED_CXX)
    build_library(
        PIC
        TYPE                SHARED
        TARGET_NAME         timemory-cxx-shared
        OUTPUT_NAME         timemory
        LANGUAGE            CXX
        LINKER_LANGUAGE     CXX
        OUTPUT_DIR          ${PROJECT_BINARY_DIR}
        SOURCES             ${CXX_LIBRARY_SOURCES}
        CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS}
        COMPILE_DEFINITIONS TIMEMORY_EXTERN_INIT
                            PRIVATE TIMEMORY_EXTERN_AUTO_TIMER_TEMPLATES
                            TIMEMORY_EXTERN_NATIVE_TEMPLATES
        LINK_LIBRARIES      timemory-headers
                            timemory-vector
                            timemory-external-shared
                            PRIVATE
                                timemory-compile-options
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
        LINKER_LANGUAGE     CXX
        OUTPUT_DIR          ${PROJECT_BINARY_DIR}
        SOURCES             ${CXX_LIBRARY_SOURCES}
        CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS}
        COMPILE_DEFINITIONS TIMEMORY_EXTERN_INIT
                            PRIVATE TIMEMORY_EXTERN_AUTO_TIMER_TEMPLATES
                            TIMEMORY_EXTERN_NATIVE_TEMPLATES
        LINK_LIBRARIES      timemory-headers
                            timemory-vector
                            timemory-external-static
                            PRIVATE
                                timemory-compile-options
                                timemory-analysis-tools
                                ${_ARCH_LIBRARY})

    if(TIMEMORY_BUILD_C)
        target_compile_definitions(timemory-cxx-static PRIVATE TIMEMORY_BUILD_C)
    endif()

    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        target_compile_definitions(timemory-cxx-static PRIVATE DEBUG)
    endif()
endif()

if(WIN32 AND _BUILD_SHARED_CXX AND _BUILD_STATIC_CXX)
    # make sure on Windows, static is built first. There is something weird
    # happening here when built in parallel...
    add_dependencies(timemory-cxx-shared timemory-cxx-static)
    target_compile_definitions(timemory-cxx-shared PRIVATE _TIMEMORY_DLL PUBLIC _TIMEMORY_LINK_LIBRARY)
    target_compile_definitions(timemory-cxx-static PUBLIC  _TIMEMORY_ARCHIVE)
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
            LINKER_LANGUAGE     CXX
            OUTPUT_DIR          ${PROJECT_BINARY_DIR}
            SOURCES             ${C_LIBRARY_SOURCES}
            C_COMPILE_OPTIONS   ${${PROJECT_NAME}_C_COMPILE_OPTIONS}
            LINK_LIBRARIES      timemory-headers
                                PRIVATE
                                    timemory-compile-options
                                    timemory-cxx-shared
                                    timemory-external-shared
                                    timemory-analysis-tools
                                    ${_ARCH_LIBRARY})
    endif()

    if(_BUILD_STATIC_CXX)

        build_library(
            TYPE                STATIC
            TARGET_NAME         timemory-c-static
            OUTPUT_NAME         ctimemory
            LANGUAGE            C
            LINKER_LANGUAGE     CXX
            OUTPUT_DIR          ${PROJECT_BINARY_DIR}
            SOURCES             ${C_LIBRARY_SOURCES}
            C_COMPILE_OPTIONS   ${${PROJECT_NAME}_C_COMPILE_OPTIONS}
            LINK_LIBRARIES      timemory-headers
                                PRIVATE
                                    timemory-compile-options
                                    timemory-cxx-static
                                    timemory-external-static
                                    timemory-analysis-tools
                                    ${_ARCH_LIBRARY})
    endif()

    if(WIN32 AND _BUILD_SHARED_CXX AND _BUILD_STATIC_CXX)
        add_dependencies(timemory-c-shared timemory-c-static)
        target_compile_definitions(timemory-c-shared PRIVATE _TIMEMORY_DLL)
        target_compile_definitions(timemory-c-static PUBLIC  _TIMEMORY_ARCHIVE)
    endif()
endif()

#----------------------------------------------------------------------------------------#
# Extern initialization
#
file(GLOB extern_init_sources       ${CMAKE_CURRENT_LIST_DIR}/init.cpp)
file(GLOB extern_native_sources     ${CMAKE_CURRENT_LIST_DIR}/native_extern.cpp)
file(GLOB extern_cuda_sources       ${CMAKE_CURRENT_LIST_DIR}/cuda_extern.cpp)
file(GLOB extern_auto_timer_sources ${CMAKE_CURRENT_LIST_DIR}/auto_timer_extern.cpp)

set(extern_init_link_libs_shared       timemory-external-shared)
set(extern_cuda_link_libs_shared       timemory-cuda # timemory-cupti
                                       timemory-cudart # timemory-cudart-device
                                       )
set(extern_native_link_libs_shared     )
set(extern_auto_timer_link_libs_shared )

set(extern_init_link_libs_static       timemory-external-static)
set(extern_cuda_link_libs_static       timemory-cuda # timemory-cupti
                                       timemory-cudart-static # timemory-cudart-device
                                       )
set(extern_native_link_libs_static     )
set(extern_auto_timer_link_libs_static )

set(extern_init_definitions       TIMEMORY_EXTERN_INIT)
set(extern_cuda_definitions       TIMEMORY_EXTERN_CUDA_TEMPLATES)
set(extern_native_definitions     TIMEMORY_EXTERN_NATIVE_TEMPLATES)
set(extern_auto_timer_definitions TIMEMORY_EXTERN_AUTO_TIMER_TEMPLATES)

# message(STATUS "BUILD CXX SHARED: ${_BUILD_SHARED_CXX}")
# message(STATUS "BUILD CXX STATIC: ${_BUILD_STATIC_CXX}")

set(TIMEMORY_EXTERN_LIBRARIES extern-native extern-cuda extern-auto-timer)
foreach(_EXTERN_LIB init native cuda auto-timer)
    foreach(_TYPE shared static)
        # Windows makes all this crap too complicated
        if(WIN32 OR
                ("${_TYPE}" STREQUAL "shared" AND NOT _BUILD_SHARED_CXX) OR
                ("${_TYPE}" STREQUAL "static" AND NOT _BUILD_STATIC_CXX))
            add_interface_library(timemory-extern-${_EXTERN_LIB}-${_TYPE})
            continue()
        endif()

        # determine linker language
        set(_LINK_LANGUAGE CXX)
        # don't build cuda
        if(NOT TIMEMORY_USE_CUDA AND "${_EXTERN_LIB}" STREQUAL "cuda")
            continue()
        elseif(TIMEMORY_USE_CUDA AND "${_EXTERN_LIB}" STREQUAL "cuda")
            set(_LINK_LANGUAGE CUDA)
        endif()

        string(REPLACE "-" "_" _NAME "${_EXTERN_LIB}")

        string(TOUPPER "${_TYPE}" _LIB_TYPE)
        #------------------------------------------------------------------------------------#
        # build the extern libraries
        #
        build_library(
            PIC
            TYPE                ${_LIB_TYPE}
            TARGET_NAME         timemory-extern-${_EXTERN_LIB}-${_TYPE}
            OUTPUT_NAME         timemory-extern-${_EXTERN_LIB}
            LANGUAGE            CXX
            LINKER_LANGUAGE     ${_LINK_LANGUAGE}
            OUTPUT_DIR          ${PROJECT_BINARY_DIR}
            SOURCES             ${extern_${_NAME}_sources}
            COMPILE_DEFINITIONS PUBLIC
                                    ${extern_${_NAME}_definitions}
            LINK_LIBRARIES      timemory-headers
                                ${extern_${_NAME}_link_libs_${_TYPE}}
                                PRIVATE
                                    timemory-compile-options
                                    timemory-arch)
        #
        # suffix when shared is empty and static is "-static
        #
        set(_SUFFIX )
        if("${_TYPE}" STREQUAL "static")
            set(_SUFFIX "-static")
        endif()

        if("${_EXTERN_LIB}" STREQUAL "init")
            if("${_TYPE}" STREQUAL "shared")
                target_link_libraries(timemory-extern-init INTERFACE timemory-extern-${_EXTERN_LIB}-shared)
            endif()
            #target_link_libraries(timemory-cxx-${_TYPE} PRIVATE timemory-extern-init-${_TYPE})
        elseif("${_EXTERN_LIB}" STREQUAL "auto-timer")
            target_link_libraries(timemory-extern-templates${_SUFFIX} INTERFACE timemory-extern-${_EXTERN_LIB}-${_TYPE})
            #target_link_libraries(timemory-cxx-${_TYPE}               PRIVATE   timemory-extern-${_EXTERN_LIB}-${_TYPE})
        elseif(NOT "${_EXTERN_LIB}" STREQUAL "cuda")
            target_link_libraries(timemory-extern-templates${_SUFFIX} INTERFACE timemory-extern-${_EXTERN_LIB}-${_TYPE})
        endif()

    endforeach()

endforeach()


#----------------------------------------------------------------------------------------#
# Install the targets and export
#

# C/C++ compiled libraries
install(
    TARGETS ${TIMEMORY_COMPILED_LIBRARIES}
    DESTINATION ${TIMEMORY_INSTALL_LIBDIR}
    EXPORT ${PROJECT_NAME}LibraryDepends)

# C/C++ development headers
macro(INSTALL_HEADER_FILES)
    foreach(_header ${ARGN})
        file(RELATIVE_PATH _relative ${CMAKE_CURRENT_LIST_DIR} ${_header})
        get_filename_component(_destpath ${_relative} DIRECTORY)
        install(FILES ${_header} DESTINATION ${TIMEMORY_INSTALL_INCLUDEDIR}/${_destpath})
    endforeach()
endmacro()

install_header_files(${timemory_headers})

install(FILES ${cxx_headers_backend}
    DESTINATION ${TIMEMORY_INSTALL_INCLUDEDIR}/timemory/backends
    COMPONENT development)

# Install the export set for use with the install-tree
install(EXPORT ${PROJECT_NAME}LibraryDepends
    DESTINATION ${TIMEMORY_INSTALL_CMAKEDIR})

export(
    TARGETS
        ${TIMEMORY_COMPILED_LIBRARIES}
        ${TIMEMORY_INTERFACE_LIBRARIES}
        ${TIMEMORY_ADDITIONAL_EXPORT_TARGETS}
    FILE
        ${CMAKE_BINARY_DIR}/${PROJECT_NAME}-build.cmake)


#----------------------------------------------------------------------------------------#
# timem wrapper tool
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
        DESTINATION ${TIMEMORY_INSTALL_BINDIR}
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
