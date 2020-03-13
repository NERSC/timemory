
##########################################################################################
#
#        timemory tool (executables)
#
##########################################################################################

# Windows not supported
if(WIN32)
    return()
endif()

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME tools)

set(_DYNINST OFF)
if(TIMEMORY_BUILD_TOOLS AND TIMEMORY_USE_DYNINST)
    set(_DYNINST ON)
endif()

set(_MPIP ${TIMEMORY_USE_MPI})
if(_MPIP AND NOT TARGET timemory-cxx-shared)
    set(_MPIP OFF)
endif()

add_option(TIMEMORY_BUILD_AVAIL "Build the timemory-avail tool" ${TIMEMORY_BUILD_TOOLS})
add_option(TIMEMORY_BUILD_TIMEM "Build the timem tool" ${TIMEMORY_BUILD_TOOLS})
add_option(TIMEMORY_BUILD_KOKKOS_TOOLS "Build the kokkos-tools libraries" OFF) # still dev
add_option(TIMEMORY_BUILD_DYNINST_TOOLS "Build the timemory-run dynamic instrumentation tool" ${_DYNINST})

if(TIMEMORY_USE_MPI)
    add_option(TIMEMORY_BUILD_MPIP "Build the mpiP library" ${_MPIP})
endif()

if(TIMEMORY_BUILD_MPIP AND NOT TARGET timemory-cxx-shared)
    message(AUTHOR_WARNING "Missing CMake target 'timemory-cxx-shared' for timemory-mpip. Disabling TIMEMORY_BUILD_MPIP...")
    set(TIMEMORY_BUILD_MPIP OFF PARENT_SCOPE)
endif()

#----------------------------------------------------------------------------------------#
# Build and install timemory-avail tool
#
if(TIMEMORY_BUILD_AVAIL)
    add_executable(timemory-avail
        ${CMAKE_CURRENT_LIST_DIR}/available.cpp
        ${CMAKE_CURRENT_LIST_DIR}/available.hpp)
    target_include_directories(timemory-avail PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    target_link_libraries(timemory-avail PRIVATE timemory-extensions timemory-headers)
    set_target_properties(timemory-avail PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
    install(TARGETS timemory-avail DESTINATION bin)
endif()


#----------------------------------------------------------------------------------------#
# Build and install timem tool
#
if(TIMEMORY_BUILD_TIMEM)
    add_library(timem-libexplain INTERFACE)

    add_option(TIMEMORY_BUILD_TOOLS_LIBEXPECT "Enable using libexpect to diagnose errors" OFF)
    if(TIMEMORY_BUILD_TOOLS_LIBEXPECT)

        find_path(LIBEXPLAIN_INCLUDE_DIR
            NAMES libexplain/execvp.h
            PATH_SUFFIXES include)

        find_library(LIBEXPLAIN_LIBRARY
            NAMES explain
            PATH_SUFFIXES lib lib64 lib32)

        if(LIBEXPLAIN_INCLUDE_DIR AND LIBEXPLAIN_LIBRARY)
            target_include_directories(timem-libexplain INTERFACE ${LIBEXPLAIN_INCLUDE_DIR})
            target_compile_definitions(timem-libexplain INTERFACE TIMEMORY_USE_LIBEXPLAIN)
            target_link_libraries(timem-libexplain INTERFACE ${LIBEXPLAIN_LIBRARY})
        else()
            set(TIMEMORY_BUILD_TOOLS_LIBEXPECT OFF)
        endif()

    endif()

    add_executable(timem timem.cpp timem.hpp)

    target_link_libraries(timem PRIVATE
        timemory-compile-options
        timemory-arch
        timemory-vector
        timemory-headers
        # timemory-papi
        timem-libexplain)

    set_target_properties(timem PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH ON)

    # timem installation
    install(TARGETS timem DESTINATION bin)

    # add_executable(timem-sampler timem-sampler.cpp timem.hpp)
    #
    # target_link_libraries(timem-sampler PRIVATE
    #    timemory-compile-options
    #    timemory-arch
    #    timemory-vector
    #    timemory-headers
    #    # timemory-papi
    #    timem-libexplain)
    #
    # set_target_properties(timem-sampler PROPERTIES
    #    INSTALL_RPATH_USE_LINK_PATH ON)
    #
    # timem-sampler installation
    # install(TARGETS timem-sampler DESTINATION bin)

endif()


#----------------------------------------------------------------------------------------#
# Build and install timemory-run
#
if(TIMEMORY_USE_DYNINST)

    find_library(DYNINST_API_RT dyninstAPI_RT
        HINTS ${Dyninst_DIR}
        PATHS ${Dyninst_DIR}
        PATH_SUFFIXES lib)

    add_executable(timemory-run timemory-run.cpp)

    if(DYNINST_API_RT)
        target_compile_definitions(timemory-run PRIVATE
            DYNINST_API_RT="${DYNINST_API_RT}")
    endif()

    target_link_libraries(timemory-run PRIVATE
        timemory-compile-options
        timemory-arch
        timemory-vector
        timemory-headers
        timemory-dyninst
        timemory-cxx-shared)

    set_target_properties(timemory-run PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH ON
        # EXCLUDE_FROM_ALL ON
    )

    install(TARGETS timemory-run DESTINATION bin)
endif()


#----------------------------------------------------------------------------------------#
# Build and install timemory-mpip library
#
if(TIMEMORY_BUILD_MPIP)
    add_subdirectory(mpip)
endif()

#----------------------------------------------------------------------------------------#
# Build and install timemory-connector libraries for kokkos
#
if(TIMEMORY_BUILD_KOKKOS_TOOLS)
    add_subdirectory(kokkos-connector)
endif()

#----------------------------------------------------------------------------------------#
# Build and install templight-xml tool
#
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/templight-xml")
    option(TIMEMORY_BUILD_TEMPLIGHT_XML "Build templight XML summarizer" OFF)
    mark_as_advanced(TIMEMORY_BUILD_TEMPLIGHT_XML)
    if(TIMEMORY_BUILD_TEMPLIGHT_XML)
        add_subdirectory(templight-xml)
    endif()
endif()
