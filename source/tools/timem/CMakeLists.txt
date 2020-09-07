
# no windows support
if(WIN32)
    return()
endif()

#----------------------------------------------------------------------------------------#
# Build and install timem tool
#
if(NOT TIMEMORY_BUILD_TIMEM)
    # if not enabled as option, make it available to be built and installed
    set(_EXCLUDE EXCLUDE_FROM_ALL)
    set(_OPTIONAL OPTIONAL)
endif()

set(_STATIC_LIB)
foreach(_TARG core timing-component rusage-component)
    if(TARGET timemory-${_TARG}-static AND NOT TIMEMORY_USE_COVERAGE)
        list(APPEND _STATIC_LIB timemory-${_TARG}-static)
    endif()
endforeach()

add_library(timem-libexplain INTERFACE)

add_option(TIMEMORY_BUILD_TOOLS_LIBEXPECT "Enable using libexpect to diagnose errors" OFF NO_FEATURE)
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

# non-MPI version
add_executable(timem ${_EXCLUDE} timem.cpp timem.hpp)

target_link_libraries(timem PRIVATE
    timemory-compile-options
    timemory-threading
    timemory-headers
    timemory-cereal
    timemory-arch
    timemory-papi
    timem-libexplain)

add_target_flag_if_avail(timem "-g")

set_target_properties(timem PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH ON)

# timem installation
install(TARGETS timem
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT   tools
    ${_OPTIONAL})

# do not build timem-mpi if not using mpi
if(NOT TIMEMORY_USE_MPI)
    return()
endif()
  
add_executable(timem-mpi ${_EXCLUDE} timem.cpp timem.hpp)

target_link_libraries(timem-mpi PRIVATE
    timemory-compile-options
    timemory-threading
    timemory-headers
    timemory-cereal
    timemory-arch
    timemory-papi
    timemory-mpi
    timem-libexplain
    ${_STATIC_LIB})

add_target_flag_if_avail(timem-mpi "-g")

set_target_properties(timem-mpi PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH ON)

# ensure timemory-pid is built
if(TARGET timemory-pid)
    add_dependencies(timem-mpi timemory-pid)
endif()

# timem installation
install(TARGETS timem-mpi
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT   tools
    ${_OPTIONAL})
