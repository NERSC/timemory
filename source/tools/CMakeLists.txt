
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
add_option(TIMEMORY_BUILD_MPIP "Build the mpiP library" ${_MPIP})

if(TIMEMORY_BUILD_MPIP AND (NOT TARGET timemory-cxx-shared OR NOT TIMEMORY_USE_MPI))
    message(AUTHOR_WARNING "TIMEMORY_BUILD_MPIP requires BUILD_SHARED_LIBS=ON and TIMEMORY_USE_MPI=ON...")
    set(TIMEMORY_BUILD_MPIP OFF PARENT_SCOPE)
    set(TIMEMORY_BUILD_MPIP OFF)
endif()

if(NOT TARGET timemory-cxx-shared AND TIMEMORY_BUILD_KOKKOS_TOOLS)
    message(AUTHOR_WARNING "TIMEMORY_BUILD_KOKKOS_TOOLS requires BUILD_SHARED_LIBS=ON...")
    set(TIMEMORY_BUILD_KOKKOS_TOOLS OFF CACHE BOOL "Build the kokkos-tools libraries" FORCE)
endif()

#----------------------------------------------------------------------------------------#
# Build and install timemory-avail tool
#
add_subdirectory(timemory-avail)

#----------------------------------------------------------------------------------------#
# Build and install timem tool
#
add_subdirectory(timem)

#----------------------------------------------------------------------------------------#
# Build and install timemory-run
#
add_subdirectory(timemory-run)

#----------------------------------------------------------------------------------------#
# Build and install timemory-mpip library
#
add_subdirectory(timemory-mpip)

#----------------------------------------------------------------------------------------#
# Build and install timemory-ompt library
#
add_subdirectory(timemory-ompt)

#----------------------------------------------------------------------------------------#
# Build and install timemory-connector libraries for kokkos
#
add_subdirectory(kokkos-connector)
