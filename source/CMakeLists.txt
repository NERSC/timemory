
#----------------------------------------------------------------------------------------#
# Locate sources and headers for this project
# - headers are included so they will show up in IDEs
#
file(GLOB c_headers                 ${CMAKE_CURRENT_LIST_DIR}/timemory/*.h)
file(GLOB c_sources                 ${CMAKE_CURRENT_LIST_DIR}/ctimemory.c)
file(GLOB cxx_headers               ${CMAKE_CURRENT_LIST_DIR}/timemory/*.hpp)
file(GLOB cxx_headers_ert           ${CMAKE_CURRENT_LIST_DIR}/timemory/ert/*.hpp)
file(GLOB cxx_headers_impl          ${CMAKE_CURRENT_LIST_DIR}/timemory/impl/*.icpp)
file(GLOB cxx_headers_comp          ${CMAKE_CURRENT_LIST_DIR}/timemory/components/*.hpp)
file(GLOB cxx_headers_backend       ${CMAKE_CURRENT_LIST_DIR}/timemory/backends/*.hpp)
file(GLOB cxx_headers_templates     ${CMAKE_CURRENT_LIST_DIR}/timemory/templates/*.hpp)
file(GLOB cxx_sources               ${CMAKE_CURRENT_LIST_DIR}/ctimemory.cpp
                                    ${CMAKE_CURRENT_LIST_DIR}/cxx.cpp
                                    ${CMAKE_CURRENT_LIST_DIR}/init.cpp)
file(GLOB cuda_sources              ${CMAKE_CURRENT_LIST_DIR}/cuda.cpp)
file(GLOB_RECURSE cereal_headers    ${CMAKE_CURRENT_LIST_DIR}/cereal/include/*.hpp
                                    ${CMAKE_CURRENT_LIST_DIR}/cereal/include/*.h
                                    ${CMAKE_CURRENT_LIST_DIR}/cereal/include/*.hh)

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
set(C_LIBRARY_SOURCES    ${c_sources}        ${c_headers})
set(CXX_LIBRARY_SOURCES  ${cxx_sources}      ${cxx_headers}      ${cxx_headers_templates}
                         ${cxx_headers_comp} ${cxx_headers_impl} ${cxx_headers_backend}
                         ${cxx_headers_ert}  ${cereal_headers})
set(CUDA_LIBRARY_SOURCES ${cuda_sources})


#----------------------------------------------------------------------------------------#
# build the C++ libraries
#
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
    COMPILE_DEFINITIONS
    LINK_LIBRARIES      timemory-headers)

build_library(
    TYPE                STATIC
    TARGET_NAME         timemory-cxx-static
    OUTPUT_NAME         timemory
    LANGUAGE            CXX
    LINKER_LANGUAGE     CXX
    OUTPUT_DIR          ${PROJECT_BINARY_DIR}
    SOURCES             ${CXX_LIBRARY_SOURCES}
    CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS}
    LINK_LIBRARIES      timemory-headers)

if(WIN32)
    # make sure on Windows, static is built first. There is something weird
    # happening here when built in parallel...
    add_dependencies(timemory-cxx-shared timemory-cxx-static)
endif()

add_interface_library(timemory-cxx)
target_link_libraries(timemory-cxx INTERFACE timemory-cxx-shared)

# private compile options
target_link_libraries(timemory-cxx-shared PRIVATE timemory-compile-options timemory-analysis-tools)
target_link_libraries(timemory-cxx-static PRIVATE timemory-compile-options timemory-analysis-tools)

if(TIMEMORY_BUILD_EXTERN_TEMPLATES)
    if(NOT WIN32)
        target_compile_definitions(timemory-cxx-shared PRIVATE TIMEMORY_BUILD_EXTERN_TEMPLATES)
        target_compile_definitions(timemory-cxx-static PRIVATE TIMEMORY_BUILD_EXTERN_TEMPLATES)
    else()
        message(WARNING "Building extern templates is not currently supported on Windows")
        set(TIMEMORY_BUILD_EXTERN_TEMPLATES OFF)
    endif()
endif()

if(WIN32)
    target_compile_definitions(timemory-cxx-shared PRIVATE _TIMEMORY_DLL)
    target_compile_definitions(timemory-cxx-static PUBLIC  _TIMEMORY_ARCHIVE)
endif()


#----------------------------------------------------------------------------------------#
# build the CUDA libraries
#
if(TIMEMORY_USE_CUDA)

    build_library(
        PIC
        TYPE                SHARED
        TARGET_NAME         timemory-cuda-shared
        OUTPUT_NAME         cutimemory
        LANGUAGE            CXX
        LINKER_LANGUAGE     CUDA
        OUTPUT_DIR          ${PROJECT_BINARY_DIR}
        SOURCES             ${CUDA_LIBRARY_SOURCES}
        CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS}
        COMPILE_DEFINITIONS
        LINK_LIBRARIES      timemory-headers timemory-cuda timemory-cudart)

    build_library(
        TYPE                STATIC
        TARGET_NAME         timemory-cuda-static
        OUTPUT_NAME         cutimemory
        LANGUAGE            CXX
        LINKER_LANGUAGE     CUDA
        OUTPUT_DIR          ${PROJECT_BINARY_DIR}
        SOURCES             ${CUDA_LIBRARY_SOURCES}
        CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS}
        LINK_LIBRARIES      timemory-headers timemory-cuda timemory-cudart-static)

    if(WIN32)
        # make sure on Windows, static is built first. There is something weird
        # happening here when built in parallel...
        add_dependencies(timemory-cuda-shared timemory-cuda-static)
    endif()

    # private compile options
    target_link_libraries(timemory-cuda-shared PRIVATE timemory-compile-options timemory-analysis-tools)
    target_link_libraries(timemory-cuda-static PRIVATE timemory-compile-options timemory-analysis-tools)

    if(TIMEMORY_BUILD_EXTERN_TEMPLATES)
        if(NOT WIN32)
            target_compile_definitions(timemory-cuda-shared PRIVATE TIMEMORY_BUILD_EXTERN_TEMPLATES)
            target_compile_definitions(timemory-cuda-static PRIVATE TIMEMORY_BUILD_EXTERN_TEMPLATES)
        else()
            message(WARNING "Building extern templates is not currently supported on Windows")
            set(TIMEMORY_BUILD_EXTERN_TEMPLATES OFF)
        endif()
    endif()

    if(WIN32)
        target_compile_definitions(timemory-cuda-shared PRIVATE _TIMEMORY_DLL)
        target_compile_definitions(timemory-cuda-static PUBLIC  _TIMEMORY_ARCHIVE)
    endif()

endif()

#----------------------------------------------------------------------------------------#
# build the C libraries
#
if(TIMEMORY_BUILD_C)

    target_compile_definitions(timemory-cxx-shared PRIVATE TIMEMORY_BUILD_C)
    target_compile_definitions(timemory-cxx-static PRIVATE TIMEMORY_BUILD_C)

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
        LINK_LIBRARIES      timemory-headers timemory-compile-options timemory-cxx-shared)

    build_library(
        TYPE                STATIC
        TARGET_NAME         timemory-c-static
        OUTPUT_NAME         ctimemory
        LANGUAGE            C
        LINKER_LANGUAGE     CXX
        OUTPUT_DIR          ${PROJECT_BINARY_DIR}
        SOURCES             ${C_LIBRARY_SOURCES}
        C_COMPILE_OPTIONS   ${${PROJECT_NAME}_C_COMPILE_OPTIONS}
        LINK_LIBRARIES      timemory-headers timemory-compile-options timemory-cxx-static)

    add_interface_library(timemory-c)
    target_link_libraries(timemory-c INTERFACE timemory-c-shared)

    target_link_libraries(timemory-c-shared PRIVATE timemory-analysis-tools)
    target_link_libraries(timemory-c-static PRIVATE timemory-analysis-tools)

    if(WIN32)
        target_compile_definitions(timemory-c-shared PRIVATE _TIMEMORY_DLL)
        target_compile_definitions(timemory-c-static PUBLIC  _TIMEMORY_ARCHIVE)
    endif()
endif()


#----------------------------------------------------------------------------------------#
# Install the targets and export
#

# C/C++ compiled libraries
install(
    TARGETS ${TIMEMORY_COMPILED_LIBRARIES} ${TIMEMORY_INTERFACE_LIBRARIES}
    DESTINATION ${TIMEMORY_INSTALL_LIBDIR}
    EXPORT ${PROJECT_NAME}LibraryDepends)

# C/C++ development headers
install(FILES ${cxx_headers} ${c_headers}
    DESTINATION ${TIMEMORY_INSTALL_INCLUDEDIR}/timemory)

install(FILES ${cxx_headers_impl}
    DESTINATION ${TIMEMORY_INSTALL_INCLUDEDIR}/timemory/impl)

install(FILES ${cxx_headers_comp}
    DESTINATION ${TIMEMORY_INSTALL_INCLUDEDIR}/timemory/components)

install(FILES ${cxx_headers_ert}
    DESTINATION ${TIMEMORY_INSTALL_INCLUDEDIR}/timemory/ert)

install(FILES ${cxx_headers_backend}
    DESTINATION ${TIMEMORY_INSTALL_INCLUDEDIR}/timemory/backends)

install(FILES ${cxx_headers_templates}
    DESTINATION ${TIMEMORY_INSTALL_INCLUDEDIR}/timemory/templates)

# Install the export set for use with the install-tree
install(EXPORT ${PROJECT_NAME}LibraryDepends
    DESTINATION ${TIMEMORY_INSTALL_CMAKEDIR})

export(
    TARGETS ${TIMEMORY_COMPILED_LIBRARIES} ${TIMEMORY_INTERFACE_LIBRARIES}
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
endif()


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
