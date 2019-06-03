
include(CMakeParseArguments)

#----------------------------------------------------------------------------------------#
# Locate sources and headers for this project
# - headers are included so they will show up in IDEs
#
file(GLOB c_headers                 ${CMAKE_CURRENT_LIST_DIR}/timemory/*.h)
file(GLOB c_sources                 ${CMAKE_CURRENT_LIST_DIR}/ctimemory.c)
file(GLOB cxx_headers               ${CMAKE_CURRENT_LIST_DIR}/timemory/*.hpp)
file(GLOB cxx_headers_impl          ${CMAKE_CURRENT_LIST_DIR}/timemory/impl/*.icpp)
file(GLOB cxx_sources               ${CMAKE_CURRENT_LIST_DIR}/ctimemory.cpp)
file(GLOB cuda_sources              ${CMAKE_CURRENT_LIST_DIR}/ctimemory.cu)
file(GLOB_RECURSE cereal_headers    ${CMAKE_CURRENT_LIST_DIR}/cereal/include/*.hpp)

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
set(C_LIBRARY_SOURCES    ${c_sources}    ${c_headers})
set(CXX_LIBRARY_SOURCES  ${cxx_sources}  ${cxx_headers} ${cxx_headers_impl} ${cereal_headers})
set(CUDA_LIBRARY_SOURCES ${cuda_sources} )

set(_OUTPUT_DIR ${PROJECT_BINARY_DIR}/timemory/lib)
if(XCODE)
    set(_OUTPUT_DIR ${PROJECT_BINARY_DIR})
endif()


#------------------------------------------------------------------------------#
# build the CUDA library
#
if(TIMEMORY_USE_CUDA AND TIMEMORY_USE_CUPTI)

    build_library(
        PIC
        TYPE                    SHARED
        TARGET_NAME             timemory-cupti-shared
        OUTPUT_NAME             cutimemory
        LANGUAGE                CUDA
        LINKER_LANGUAGE         CUDA
        OUTPUT_DIR              ${_OUTPUT_DIR}
        COMPILE_DEFINITIONS     ${${PROJECT_NAME}_DEFINITIONS} TIMEMORY_USE_CUPTI
        CXX_COMPILE_OPTIONS     ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS}
        CUDA_COMPILE_OPTIONS    ${${PROJECT_NAME}_CUDA_COMPILE_OPTIONS}
        SOURCES                 ${CUDA_LIBRARY_SOURCES}
        LINK_LIBRARIES          timemory-headers timemory-extensions timemory-cuda
                                ${CUDA_cupti_LIBRARY} ${CUDA_cuda_LIBRARY}
        INCLUDE_DIRECTORIES     ${${PROJECT_NAME}_TARGET_INCLUDE_DIRS}
                                ${CUDA_cupti_INCLUDE_DIR})

    build_library(
        PIC
        TYPE                    STATIC
        TARGET_NAME             timemory-cupti-static
        OUTPUT_NAME             cutimemory
        LANGUAGE                CUDA
        LINKER_LANGUAGE         CUDA
        OUTPUT_DIR              ${_OUTPUT_DIR}
        COMPILE_DEFINITIONS     ${${PROJECT_NAME}_DEFINITIONS} TIMEMORY_USE_CUPTI
        SOURCES                 ${CUDA_LIBRARY_SOURCES}
        CXX_COMPILE_OPTIONS     ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS}
        CUDA_COMPILE_OPTIONS    ${${PROJECT_NAME}_CUDA_COMPILE_OPTIONS}
        LINK_LIBRARIES          timemory-headers timemory-extensions timemory-cuda
                                ${CUDA_cupti_LIBRARY} ${CUDA_cuda_LIBRARY}
        INCLUDE_DIRECTORIES     ${${PROJECT_NAME}_TARGET_INCLUDE_DIRS}
                                ${CUDA_cupti_INCLUDE_DIR})

    target_link_libraries(timemory-cupti-shared INTERFACE timemory-cuda)
    target_link_libraries(timemory-cupti-static INTERFACE timemory-cuda)

    if(WIN32)
        target_compile_definitions(timemory-cupti-shared PRIVATE _TIMEMORY_DLL)
        target_compile_definitions(timemory-cupti-static PUBLIC  _TIMEMORY_ARCHIVE)
    endif()

elseif(TIMEMORY_USE_CUDA)
    add_interface_library(timemory-cupti-shared)
    add_interface_library(timemory-cupti-static)
endif()


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
    OUTPUT_DIR          ${_OUTPUT_DIR}
    SOURCES             ${CXX_LIBRARY_SOURCES}
    CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS}
    COMPILE_DEFINITIONS
    LINK_LIBRARIES      timemory-cxx-compile-options timemory-headers timemory-extensions)

build_library(
    TYPE                STATIC
    TARGET_NAME         timemory-cxx-static
    OUTPUT_NAME         timemory
    LANGUAGE            CXX
    LINKER_LANGUAGE     CXX
    OUTPUT_DIR          ${_OUTPUT_DIR}
    SOURCES             ${CXX_LIBRARY_SOURCES}
    CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS}
    LINK_LIBRARIES      timemory-headers timemory-extensions)

add_interface_library(timemory-cxx)
target_link_libraries(timemory-cxx INTERFACE timemory-cxx-shared)

if(TIMEMORY_BUILD_EXTERN_TEMPLATES)
    target_compile_definitions(timemory-cxx-shared PRIVATE TIMEMORY_BUILD_EXTERN_TEMPLATES)
    target_compile_definitions(timemory-cxx-static PRIVATE TIMEMORY_BUILD_EXTERN_TEMPLATES)
endif()

if(WIN32)
    target_compile_definitions(timemory-cxx-shared PRIVATE _TIMEMORY_DLL)
    target_compile_definitions(timemory-cxx-static PUBLIC  _TIMEMORY_ARCHIVE)
endif()


#----------------------------------------------------------------------------------------#
# build the C libraries
#
if(TIMEMORY_BUILD_C)

    build_library(
        PIC
        TYPE                SHARED
        TARGET_NAME         timemory-c-shared
        OUTPUT_NAME         ctimemory
        LANGUAGE            C
        LINKER_LANGUAGE     CXX
        OUTPUT_DIR          ${_OUTPUT_DIR}
        SOURCES             ${C_LIBRARY_SOURCES}
        C_COMPILE_OPTIONS   ${${PROJECT_NAME}_C_COMPILE_OPTIONS}
        LINK_LIBRARIES      timemory-headers timemory-extensions timemory-cxx-shared)

    build_library(
        TYPE                STATIC
        TARGET_NAME         timemory-c-static
        OUTPUT_NAME         ctimemory
        LANGUAGE            C
        LINKER_LANGUAGE     CXX
        OUTPUT_DIR          ${_OUTPUT_DIR}
        SOURCES             ${C_LIBRARY_SOURCES}
        C_COMPILE_OPTIONS   ${${PROJECT_NAME}_C_COMPILE_OPTIONS}
        LINK_LIBRARIES      timemory-headers timemory-extensions timemory-cxx-static)

    add_interface_library(timemory-c)
    target_link_libraries(timemory-c INTERFACE timemory-c-shared)

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
    TARGETS ${TIMEMORY_COMPILED_LIBRARIES} ${TIMEMORY_INTERFACE_LIBRARIES} ${TIMEMORY_EXTERNAL_LIBRARIES}
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

export(
    TARGETS ${TIMEMORY_COMPILED_LIBRARIES} ${TIMEMORY_INTERFACE_LIBRARIES} ${TIMEMORY_EXTERNAL_LIBRARIES}
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
        install(TARGETS timemory-cxx
            DESTINATION ${TIMEMORY_INSTALL_PYTHONDIR}/lib
            COMPONENT python)
    endif()

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
