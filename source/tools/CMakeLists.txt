
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
if(_MPIP AND (NOT TARGET timemory-cxx-shared OR NOT TIMEMORY_USE_GOTCHA))
    set(_MPIP OFF)
endif()

set(_OMPT ${TIMEMORY_USE_OMPT})
if(_OMPT AND NOT TARGET timemory-cxx-shared)
    set(_OMPT OFF)
endif()

add_option(TIMEMORY_BUILD_AVAIL "Build the timemory-avail tool" ${TIMEMORY_BUILD_TOOLS})
add_option(TIMEMORY_BUILD_TIMEM "Build the timem tool" ${TIMEMORY_BUILD_TOOLS})
add_option(TIMEMORY_BUILD_KOKKOS_TOOLS "Build the kokkos-tools libraries" OFF) # still dev
add_option(TIMEMORY_BUILD_DYNINST_TOOLS "Build the timemory-run dynamic instrumentation tool" ${_DYNINST})
add_option(TIMEMORY_BUILD_MPIP_LIBRARY "Build the mpiP library" ${_MPIP})
add_option(TIMEMORY_BUILD_OMPT_LIBRARY "Build the OMPT library" ${_OMPT})

if(TIMEMORY_BUILD_MPIP_LIBRARY AND (NOT TARGET timemory-cxx-shared OR
    NOT TIMEMORY_USE_MPI OR NOT TIMEMORY_USE_GOTCHA))
    message(AUTHOR_WARNING
        "TIMEMORY_BUILD_MPIP_LIBRARY requires BUILD_SHARED_LIBS=ON, TIMEMORY_USE_MPI=ON, and TIMEMORY_USE_GOTCHA=ON...")
    set(TIMEMORY_BUILD_MPIP_LIBRARY OFF CACHE BOOL "Build the mpiP library" FORCE)
endif()

if(TIMEMORY_BUILD_OMPT_LIBRARY AND NOT TIMEMORY_USE_OMPT)
    message(AUTHOR_WARNING
        "TIMEMORY_BUILD_OMPT_LIBRARY requires BUILD_SHARED_LIBS=ON and TIMEMORY_USE_OMPT=ON...")
    set(TIMEMORY_BUILD_OMPT_LIBRARY OFF CACHE BOOL "Build the OMPT library" FORCE)
endif()

if(NOT TARGET timemory-cxx-shared AND TIMEMORY_BUILD_KOKKOS_TOOLS)
    message(AUTHOR_WARNING
        "TIMEMORY_BUILD_KOKKOS_TOOLS requires BUILD_SHARED_LIBS=ON...")
    set(TIMEMORY_BUILD_KOKKOS_TOOLS OFF CACHE BOOL "Build the kokkos-tools libraries" FORCE)
endif()

#----------------------------------------------------------------------------------------#
# Build and install timemory-avail tool
#
message(STATUS "Adding source/tools/timemory-avail...")
add_subdirectory(timemory-avail)

#----------------------------------------------------------------------------------------#
# Build and install timemory-pid tool
#
message(STATUS "Adding source/tools/timemory-pid...")
add_subdirectory(timemory-pid)

#----------------------------------------------------------------------------------------#
# Build and install timem tool
#
message(STATUS "Adding source/tools/timem...")
add_subdirectory(timem)

#----------------------------------------------------------------------------------------#
# Build and install timemory-run
#
message(STATUS "Adding source/tools/timemory-run...")
add_subdirectory(timemory-run)

#----------------------------------------------------------------------------------------#
# Build and install timemory-mpip library
#
message(STATUS "Adding source/tools/timemory-mpip...")
add_subdirectory(timemory-mpip)

#----------------------------------------------------------------------------------------#
# Build and install timemory-ompt library
#
message(STATUS "Adding source/tools/timemory-ompt...")
add_subdirectory(timemory-ompt)

#----------------------------------------------------------------------------------------#
# Build and install timemory-connector libraries for kokkos
#
message(STATUS "Adding source/tools/kokkos-connector...")
add_subdirectory(kokkos-connector)
